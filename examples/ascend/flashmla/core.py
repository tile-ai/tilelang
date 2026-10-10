"""FlashMLA V4.1 sparse attention, explicit AIC/AIV pipeline, no AutoSchedule.

The schedule, gather2 and skip-scale algorithm follow FlashMLA PR #229.
All loops, buffer ownership, flags, and SIMD calculations are expressed in
TileLang. manual_intrinsics.h only adapts individual CANN instructions and SS
buffer access; it contains no attention kernel or scheduling loop.
"""

from pathlib import Path
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from examples.ascend.flashmla.quantization import dequantize

PASS_CONFIG = {
    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    # Explicit index masks and zero-burst DMA protect every masked gather.
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}


def rolled(n):
    """Emit pragma unroll 1, matching the reference rolled loops."""
    return T.unroll(n, annotations={"pragma_unroll_factor": 1})


def sparse_attention(
    nq,
    nk,
    topk,
    *,
    sink=True,
    variable_lengths=False,
    scale=512**-0.5,
    q_strides=None,
    index_stride=None,
    kv_format="bf16",
    extra_topk=0,
    extra_format="fp8",
    extra_index_stride=None,
    variable_extra_lengths=False,
    kv_storage_bytes=1,
    extra_storage_bytes=1,
    cache_hint=0,
):
    if topk <= 0 or topk % 64:
        raise ValueError("topk must be a positive multiple of 64")
    if kv_format not in ("bf16", "fp8") or extra_format not in ("fp8", "fp4"):
        raise ValueError("Supported formats are BF16 prefill and FP8 main / FP8 or FP4 extra decode")
    if extra_topk < 0 or extra_topk % 64:
        raise ValueError("extra_topk must be a nonnegative multiple of 64")
    if max(kv_storage_bytes, extra_storage_bytes) >= 2**32:
        raise ValueError("The manual gather uses 32-bit byte offsets per KV cache")
    decode = kv_format != "bf16"
    fp4 = kv_format == "fp4"
    extra_fp4 = extra_format == "fp4"
    record_bytes = 288 if fp4 else 528
    extra_record_bytes = 288 if extra_fp4 else 528
    valid_limit = T.uint32(0x80000000) if decode else nk
    extra_index_stride = extra_index_stride or max(extra_topk, 1)
    if not decode and nk >= 2**21:
        raise ValueError("This gather2 implementation uses signed 32-bit byte offsets")
    q_strides = q_strides or (64 * 512, 512, 1)
    index_stride = index_stride or topk
    nb0 = topk // 64
    nb1 = (extra_topk + 63) // 64
    nb = nb0 + nb1
    ib = 640 if decode else (topk if topk <= 640 and topk % 128 == 0 else 128)
    groups0 = (nb0 + ib // 64 - 1) // (ib // 64)
    groups = groups0 + (nb1 + ib // 64 - 1) // (ib // 64)
    # A full 8/10-block index group needs only two live versions. This keeps
    # dynamic UB below 248 KiB, leaving CANN its kernel-reserved 8 KiB.
    index_versions = 2 if not decode and ib >= 512 and nb % (ib // 64) == 0 else 4
    source = Path(__file__).with_name("manual_intrinsics.h").read_text()

    @T.macro
    def cset(pipe, flag):
        T.ascend_cross_core_set_flag(4, pipe, flag)
        T.ascend_cross_core_set_flag(4, pipe, flag + 16)

    @T.macro
    def cwait(pipe, flag):
        T.ascend_cross_core_wait_flag(4, pipe, flag)
        T.ascend_cross_core_wait_flag(4, pipe, flag + 16)

    @T.macro
    def load_q(job, bx, Q, l1):
        qidx = bx + job * 32
        T.ascend_wait_flag("MTE1_MTE2", job % 2)
        T.call_extern("void", "fm_q_load", T.access_ptr(l1[(job % 2) * 16384], "w"), T.access_ptr(Q[qidx, 0, 0], "r"), q_strides[1] * 2)
        T.call_extern(
            "void", "fm_q_load", T.access_ptr(l1[229376 + (job % 2) * 16384], "w"), T.access_ptr(Q[qidx, 32, 0], "r"), q_strides[1] * 2
        )
        T.ascend_set_flag("MTE2_MTE1", job % 2)

    @T.macro
    def qk(job, k, acounter, l1, a, b, c, p):
        T.call_extern("void", "asc_set_mmad_direction_m")
        slot = (job * nb + k) % 4
        for d in T.unroll(4):
            ai = (acounter + d) % 4
            if d == 0 and k == 0:
                T.ascend_wait_flag("MTE2_MTE1", job % 2)
            T.ascend_get_buf("PIPE_MTE1", ai)
            T.call_extern("void", "fm_a_load", T.access_ptr(a[ai * 8192], "w"), T.access_ptr(l1[(job % 2) * 16384], "r"), d)
            T.call_extern("void", "fm_a_load", T.access_ptr(a[ai * 8192 + 512], "w"), T.access_ptr(l1[229376 + (job % 2) * 16384], "r"), d)
            T.ascend_rls_buf("PIPE_MTE1", ai)
            if d == 3 and k == nb - 1:
                T.ascend_set_flag("MTE1_MTE2", job % 2)
            if d == 0:
                cwait("PIPE_MTE1", 6 + slot)
            T.ascend_get_buf("PIPE_MTE1", 4 + d)
            T.call_extern("void", "fm_k_load", T.access_ptr(b[d * 8192], "w"), T.access_ptr(l1[32768 + slot * 16384], "r"), d)
            T.call_extern("void", "fm_k_load", T.access_ptr(b[d * 8192 + 512], "w"), T.access_ptr(l1[147456 + slot * 16384], "r"), d)
            T.ascend_rls_buf("PIPE_MTE1", 4 + d)
            T.ascend_get_buf("PIPE_M", ai)
            T.ascend_get_buf("PIPE_M", 4 + d)
            T.call_extern(
                "void",
                "fm_mmad",
                T.access_ptr(c[32768], "rw"),
                T.access_ptr(a[ai * 8192], "r"),
                T.access_ptr(b[d * 8192], "r"),
                64,
                128,
                3 if d == 3 else 2,
                d == 0,
            )
            T.ascend_rls_buf("PIPE_M", ai)
            T.ascend_rls_buf("PIPE_M", 4 + d)
        cwait("PIPE_FIX", 0)
        T.call_extern("void", "fm_p_store", T.access_ptr(p[0], "w"), T.access_ptr(c[32768], "r"))
        cset("PIPE_FIX", 0)

    @T.macro
    def copy_o(last, c, out):
        if not last:
            T.ascend_set_flag("M_FIX", 0)
            T.ascend_wait_flag("M_FIX", 0)
        for d in T.unroll(4):
            cwait("PIPE_FIX", 4 + d % 2)
            T.call_extern("void", "fm_o_store", T.access_ptr(out[d % 2, 0], "w"), T.access_ptr(c[d * 8192], "r"), 3 if last else 0)
            cset("PIPE_FIX", 4 + d % 2)
        if not last:
            T.ascend_set_flag("FIX_M", 0)
            T.ascend_wait_flag("FIX_M", 0)

    @T.macro
    def pv(job, k, ai, l1, a, b, c, out):
        T.call_extern("void", "asc_set_mmad_direction_n")
        slot = (job * nb + k) % 4
        clear = T.alloc_var("int32", init=0)
        for d in T.unroll(4):
            T.ascend_get_buf("PIPE_MTE1", 4 + d)
            T.call_extern("void", "fm_v_load", T.access_ptr(b[d * 8192], "w"), T.access_ptr(l1[32768 + slot * 16384], "r"), d)
            T.call_extern("void", "fm_v_load", T.access_ptr(b[d * 8192 + 4096], "w"), T.access_ptr(l1[147456 + slot * 16384], "r"), d)
            T.ascend_rls_buf("PIPE_MTE1", 4 + d)
            if d == 3:
                cset("PIPE_MTE1", 6 + slot)
            if d == 0:
                cwait("PIPE_MTE1", 1)
                T.ascend_get_buf("PIPE_MTE1", ai)
                T.call_extern("void", "fm_s_load", T.access_ptr(a[ai * 8192], "w"), T.access_ptr(l1[129024], "r"))
                T.ascend_rls_buf("PIPE_MTE1", ai)
                cset("PIPE_MTE1", 1)
                T.ascend_get_buf("PIPE_M", ai)
                if k > 0:
                    cwait("PIPE_S", 2)
                    cset("PIPE_S", 2)
                    clear = T.call_extern("int32", "fm_ss_get", (job % 8) * 8 + k % 8)
                    if clear:
                        copy_o(False, c, out)
            T.ascend_get_buf("PIPE_M", 4 + d)
            T.call_extern(
                "void",
                "fm_mmad",
                T.access_ptr(c[d * 8192], "rw"),
                T.access_ptr(a[ai * 8192], "r"),
                T.access_ptr(b[d * 8192], "r"),
                128,
                64,
                T.if_then_else(k == nb - 1, 3, T.if_then_else(k == 0, 2, 0)),
                (k == 0) | (clear != 0),
            )
            T.ascend_rls_buf("PIPE_M", 4 + d)
        T.ascend_rls_buf("PIPE_M", ai)
        if k == nb - 1:
            copy_o(True, c, out)

    @T.macro
    def softmax_vf(first, job4, k4, idx_slot, idx_off, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta):
        with T.SimdVF():
            full = T.simd.pset(32)
            eight = T.simd.pset(32, "PAT_VL8")
            one = T.simd.pset(32, "PAT_VL1")
            # Paired indices are sorted by gather2; repeat 16 columns.
            masks = T.alloc_local((4,), "boolx256")
            for col in T.unroll(4):
                lo = T.simd.vld(smaller[idx_slot, idx_off + col * 8], "BLK")
                hi = T.simd.vld(larger[idx_slot, idx_off + col * 8], "BLK")
                paired, ignored = T.simd.vintlv(lo, hi)
                masks[col] = T.simd.vcmps(paired, valid_limit, op="lt")
            old_sum = T.simd.vld(sums[job4 % 2, 0])
            old_scale_max = T.simd.vld(scaling_max[job4 % 2, 0])
            for r in rolled(8):
                rm = T.alloc_var("float32x64")
                rm = T.simd.vdup(-3.4028234663852886e38, "float32")
                if not first:
                    rm = T.simd.vld(maxima[job4 % 2, r * 8], "E2B_B32")
                max_ptr = T.simd.make_ubuf_ptr(T.access_ptr(p[r * 64], "r", extent=1537), "float32")
                for col in T.unroll(4):
                    x, max_ptr = T.simd.vld(max_ptr, post_inc=512)
                    rm = T.simd.vmax(rm, x, masks[col], mode="MODE_MERGING")
                reduced = T.simd.vcgmax(rm)
                even, odd = T.simd.vdintlv(reduced, reduced)
                rmax = T.simd.vmax(even, odd)
                repeated, ignored2 = T.simd.vintlv(rmax, rmax)
                T.simd.vsts(maxima[job4 % 2, r * 8], repeated, eight)
            T.simd.mem_bar("VST_VLD")
            new_max = T.simd.vld(maxima[job4 % 2, 0])
            chosen_max = T.alloc_var("float32x64")
            factor = T.alloc_var("float32x64")
            chosen_max = new_max
            factor = T.simd.vdup(0.0, "float32")
            if not first:
                diff = T.simd.vsub(new_max, old_scale_max)
                greatest = T.simd.vcmax(diff)
                broadcast = T.simd.vdupv(greatest)
                rescale = T.simd.vcmps(broadcast, 6.0 / scale, op="gt")
                chosen_max = T.simd.vsel(new_max, old_scale_max, rescale)
                factor = T.simd.vexpdif(T.simd.vmuls(old_scale_max, scale), T.simd.vmuls(chosen_max, scale))
                T.simd.vsts(delta[job4 % 2, k4 % 2, 0], greatest, one)
            T.simd.vsts(factors[job4 % 2, k4 % 2, 0], factor)
            T.simd.vsts(scaling_max[job4 % 2, 0], chosen_max)
            T.simd.mem_bar("VST_VLD")
            pp = T.simd.make_ubuf_ptr(T.access_ptr(p[0], "r", extent=2048), "float32")
            for r in rolled(4):
                rmax2 = T.alloc_local((2,), "float32x64")
                rsum2 = T.alloc_local((2,), "float32x64")
                packed2 = T.alloc_local((2, 2), "bfloat16x128")
                for rr in T.unroll(2):
                    rmax2[rr] = T.simd.vmuls(T.simd.vld(scaling_max[job4 % 2, r * 16 + rr * 8], "E2B_B32"), scale)
                for pair in T.unroll(2):
                    for cc in T.unroll(2):
                        for rr in T.unroll(2):
                            increment = T.if_then_else(rr == 0, 64, T.if_then_else(pair == 1 and cc == 1, -1472, 448))
                            x, pp = T.simd.vld(pp, post_inc=increment)
                            prob = T.simd.vexpdif(T.simd.vmuls(x, scale), rmax2[rr], masks[pair * 2 + cc])
                            if pair == 0 and cc == 0:
                                rsum2[rr] = prob
                            else:
                                rsum2[rr] = T.simd.vadd(rsum2[rr], prob)
                            if cc == 0:
                                packed2[rr, cc] = T.simd.vcvt(prob, "bfloat16", full, part=0)
                            else:
                                packed2[rr, cc] = T.simd.vcvt(prob, "bfloat16", full, part=1)
                    merged0 = T.simd.vadd(packed2[0, 0], packed2[0, 1])
                    merged1 = T.simd.vadd(packed2[1, 0], packed2[1, 1])
                    col0, col1 = T.simd.vdintlv(merged0, merged1)
                    T.simd.vsts(s[pair * 2048 + r * 128], col0)
                    T.simd.vsts(s[pair * 2048 + 1024 + r * 128], col1)
                for rr in T.unroll(2):
                    reduced_sum = T.simd.vcpadd(T.simd.vcgadd(rsum2[rr]))
                    repeated_sum, ignored3 = T.simd.vintlv(reduced_sum, reduced_sum)
                    T.simd.vsts(sums[job4 % 2, r * 16 + rr * 8], repeated_sum, eight)
            T.simd.mem_bar("VST_VLD")
            if not first:
                current_sum = T.simd.vld(sums[job4 % 2, 0])
                updated_sum = T.simd.vadd(T.simd.vmul(old_sum, factor), current_sum)
                T.simd.vsts(sums[job4 % 2, 0], updated_sum)

    @T.macro
    def scale_output_vf(add, d, job5, k5, factors, out, accum):
        with T.SimdVF():
            for r in rolled(32):
                factor5 = T.simd.vld(factors[job5 % 2, k5 % 2, r * 2], "BRC_B32")
                for j in T.unroll(2):
                    cur = T.alloc_var("float32x64", init=T.simd.vld(out[d % 2, r * 128 + j * 64]))
                    if add:
                        old = T.simd.vld(accum[r * 512 + d * 128 + j * 64])
                        cur = T.simd.vadd(old, cur)
                    T.simd.vsts(accum[r * 512 + d * 128 + j * 64], T.simd.vmul(cur, factor5))

    @T.macro
    def final_output_vf(add, d, job5, denorm, out, accum, final):
        with T.SimdVF():
            mask16 = T.simd.pset(16)
            mask32 = T.simd.pset(32)
            pp = T.simd.make_ubuf_ptr(T.access_ptr(out[d % 2, 0], "r", extent=4096), "float32")
            for r in rolled(16):
                packed = T.alloc_local((2, 2), "bfloat16x128")
                for rr in T.unroll(2):
                    norm = T.simd.vld(denorm[job5 % 2, r * 2 + rr], "BRC_B32")
                    for cc in T.unroll(2):
                        x, pp = T.simd.vld(pp, post_inc=64)
                        if add:
                            old = T.simd.vld(accum[(r * 2 + rr) * 512 + d * 128 + cc * 64])
                            x = T.simd.vadd(x, old, mask32)
                        scaled = T.simd.vmul(x, norm, mask32)
                        if rr == 0:
                            packed[rr, cc] = T.simd.vcvt(scaled, "bfloat16", mask32, part=0)
                        else:
                            packed[rr, cc] = T.simd.vcvt(scaled, "bfloat16", mask32, part=1)
                merged0 = T.simd.vadd(packed[0, 0], packed[1, 0], mask16)
                merged1 = T.simd.vadd(packed[0, 1], packed[1, 1], mask16)
                row0, row1 = T.simd.vdintlv(merged0, merged1)
                T.simd.vsts(final[r * 2048 + d * 256], row0, mask16)
                T.simd.vsts(final[r * 2048 + 1024 + d * 256], row1, mask16)

    @T.prim_func
    def main(
        Q: T.StridedTensor((nq, 64, 512), q_strides, "bfloat16"),
        KV: T.Tensor((kv_storage_bytes,) if decode else (nk, 512), "uint8" if decode else "bfloat16"),
        Indices: T.StridedTensor((nq, topk), (index_stride, 1), "int32"),
        Sink: T.Tensor((64,), "float32"),
        Lengths: T.Tensor((nq,), "int32"),
        ExtraKV: T.Tensor((extra_storage_bytes,), "uint8"),
        ExtraIndices: T.StridedTensor((nq, max(extra_topk, 1)), (extra_index_stride, 1), "int32"),
        ExtraLengths: T.Tensor((nq,), "int32"),
        O: T.Tensor((nq, 64, 512), "bfloat16"),
        Max: T.Tensor((nq, 64), "float32"),
        LSE: T.Tensor((nq, 64), "float32"),
    ):
        with T.Kernel(min(nq, 32)) as bx:
            T.import_source(source)
            # Explicit physical L1 plan balances both 256 KiB bank groups.
            l1 = T.alloc_l1((262144,), "bfloat16")
            a = T.alloc_l0a((32768,), "bfloat16")
            b = T.alloc_l0b((32768,), "bfloat16")
            c = T.alloc_l0c((36864,), "float32", layout=False)
            kv = T.alloc_shared((3, 32, 1 if decode else 512), "bfloat16")
            raw8 = T.alloc_shared((3, 32, 544 if decode else 1), "uint8")
            raw4 = T.alloc_shared((3, 32, 288 if decode else 1), "uint8")
            scales = T.alloc_shared((1024 if decode else 1,), "bfloat16")
            nz = T.alloc_shared((33 * 512,), "bfloat16")
            accum = T.alloc_shared((32 * 512,), "float32")
            out = T.alloc_shared((2, 32 * 128), "float32")
            p = T.alloc_shared((32 * 64,), "float32")
            s = tvm.tirx.decl_buffer((32 * 128,), "bfloat16", data=p.data, scope="shared.dyn")
            final = tvm.tirx.decl_buffer((32 * 1024,), "bfloat16", data=accum.data, scope="shared.dyn")
            maxima = T.alloc_shared((2, 64), "float32")
            scaling_max = T.alloc_shared((2, 64), "float32")
            sums = T.alloc_shared((2, 64), "float32")
            denorm = T.alloc_shared((2, 64), "float32")
            factors = T.alloc_shared((2, 2, 64), "float32")
            delta = T.alloc_shared((2, 2, 64), "float32")
            attn_sink = T.alloc_shared((64,), "float32")
            indices = T.alloc_shared((ib,), "int32")
            smaller = T.alloc_shared((index_versions, ib // 2), "uint32")
            larger = T.alloc_shared((index_versions, ib // 2), "uint32")
            offsets = T.alloc_shared((ib // 2,), "uint32")
            strides = T.alloc_shared((ib // 2,), "uint32")
            T.call_extern("void", "fm_init")

            with T.Cube():
                cset("PIPE_S", 1)
                for i in T.unroll(4):
                    cset("PIPE_S", 6 + i)
                for i in T.unroll(2):
                    T.ascend_set_flag("MTE1_MTE2", i)
                acounter = T.alloc_var("int32", init=0)
                jobs = T.ceildiv(nq - bx, 32)
                load_q(0, bx, Q, l1)
                qk(0, 0, acounter, l1, a, b, c, p)
                for job in rolled(jobs):
                    if job + 1 < jobs:
                        load_q(job + 1, bx, Q, l1)
                    for k in rolled(nb - 1):
                        qk(job, k + 1, acounter, l1, a, b, c, p)
                        pv(job, k, acounter, l1, a, b, c, out)
                        acounter = (acounter + 1) % 4
                    if job + 1 < jobs:
                        qk(job + 1, 0, acounter, l1, a, b, c, p)
                    pv(job, nb - 1, acounter, l1, a, b, c, out)
                    acounter = (acounter + 1) % 4
                T.ascend_pipe_barrier("PIPE_ALL")

            with T.Vector() as sid:
                if sink:
                    T.copy(Sink[sid * 32 : sid * 32 + 32], attn_sink[:32])
                    T.ascend_set_flag("MTE2_V", 2)
                T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                for i in T.unroll(2):
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 4 + i)
                for i in rolled(3):
                    T.ascend_get_buf("PIPE_V", 8 + i)
                    with T.SimdVF():
                        if decode:
                            zero = T.simd.vdup(0, "uint8")
                            for j in rolled(68):
                                T.simd.vsts(raw8[i, j * 256 // 544, j * 256 % 544], zero)
                            for j in rolled(36):
                                T.simd.vsts(raw4[i, j * 256 // 288, j * 256 % 288], zero)
                        else:
                            zero = T.simd.vdup(0.0, "bfloat16")
                            for j in rolled(128):
                                T.simd.vsts(kv[i, j // 4, j % 4 * 128], zero)
                    T.ascend_rls_buf("PIPE_V", 8 + i)
                emitted = T.alloc_var("int32", init=0)
                total = T.ceildiv(nq - bx, 32) * nb
                for step in rolled(total + 5):
                    if not decode:  # noqa: SIM102 - compile-time format dispatch
                        # Stage 0: fetch the next index group.
                        if step < total:
                            job0 = step // nb
                            k0 = step % nb
                            q0 = bx + job0 * 32
                            extra0 = k0 >= nb0
                            rel0 = T.if_then_else(extra0, k0 - nb0, k0)
                            if rel0 % (ib // 64) == 0:
                                T.ascend_get_buf("PIPE_MTE2", 0)
                                if extra_topk > 0 and extra0:
                                    length0 = T.min(ib, extra_topk - rel0 * 64)
                                    T.copy(ExtraIndices[q0, rel0 * 64 : rel0 * 64 + length0], indices[:length0])
                                else:
                                    length0 = T.min(ib, topk - rel0 * 64)
                                    T.copy(Indices[q0, rel0 * 64 : rel0 * 64 + length0], indices[:length0])
                                T.ascend_rls_buf("PIPE_MTE2", 0)
                    # Stage 2: ND -> compact NZ, then publish a KV slot to AIC.
                    if step >= 2 and step - 2 < total:
                        t2 = step - 2
                        slot2 = T.alloc_var("int32", init=t2 % 3)
                        T.ascend_get_buf("PIPE_V", 8 + slot2)
                        T.ascend_get_buf("PIPE_V", 11)
                        if decode:
                            if extra_topk > 0 and t2 % nb >= nb0:
                                if extra_fp4:
                                    dequantize(raw4, scales, nz, slot2, True)
                                else:
                                    dequantize(raw8, scales, nz, slot2, False)
                            else:
                                if fp4:
                                    dequantize(raw4, scales, nz, slot2, True)
                                else:
                                    dequantize(raw8, scales, nz, slot2, False)
                        else:
                            with T.SimdVF():
                                nd_mask = T.simd.pset(16)
                                src_ptr = T.simd.make_ubuf_ptr(T.access_ptr(kv[slot2, 0, 0], "r", extent=16384), "bfloat16")
                                dst0 = T.simd.make_ubuf_ptr(T.access_ptr(nz[0], "w", extent=16896), "bfloat16")
                                dst1 = T.simd.make_ubuf_ptr(T.access_ptr(nz[4224], "w", extent=12672), "bfloat16")
                                for _col in rolled(2):
                                    for row in T.unroll(32):
                                        x0, src_ptr = T.simd.vld(src_ptr, post_inc=128)
                                        x1, src_ptr = T.simd.vld(src_ptr, post_inc=T.if_then_else(row == 31, -15744, 384))
                                        config = (33 << 16) | T.if_then_else(row == 31, 497, 1)
                                        dst0 = T.simd.vsstb(x0, dst0, config, nd_mask, update=True)
                                        dst1 = T.simd.vsstb(x1, dst1, config, nd_mask, update=True)
                        T.ascend_rls_buf("PIPE_V", 11)
                        T.ascend_rls_buf("PIPE_V", 8 + slot2)
                        T.ascend_get_buf("PIPE_MTE3", 11)
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 6 + t2 % 4)
                        T.call_extern(
                            "void",
                            "fm_kv_push",
                            T.access_ptr(l1[T.if_then_else(sid == 0, 32768, 147456) + t2 % 4 * 16384], "w"),
                            T.access_ptr(nz[0], "r"),
                        )
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 6 + t2 % 4)
                        T.ascend_rls_buf("PIPE_MTE3", 11)
                    # Stage 4: online softmax, with a threshold of 6 for rescaling.
                    if step >= 4 and step - 4 < total:
                        t4 = step - 4
                        job4 = T.alloc_var("int32", init=t4 // nb)
                        k4 = T.alloc_var("int32", init=t4 % nb)
                        extra4 = k4 >= nb0
                        rel4 = T.if_then_else(extra4, k4 - nb0, k4)
                        idx_slot = T.alloc_var(
                            "int32", init=(job4 * groups + T.if_then_else(extra4, groups0, 0) + rel4 // (ib // 64)) % index_versions
                        )
                        idx_off = T.alloc_var("int32", init=(rel4 % (ib // 64)) * 32)
                        T.ascend_cross_core_wait_flag(4, "PIPE_V", 0)
                        T.ascend_get_buf("PIPE_V", 2 + job4 % 2)
                        if k4 == 0:
                            softmax_vf(True, job4, k4, idx_slot, idx_off, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta)
                        else:
                            softmax_vf(False, job4, k4, idx_slot, idx_off, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta)
                        T.ascend_rls_buf("PIPE_V", 2 + job4 % 2)
                        T.ascend_set_flag("V_MTE3", 0)
                        if k4 > 0:
                            T.ascend_set_flag("V_S", 0)
                        T.ascend_wait_flag("V_MTE3", 0)
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 1)
                        T.call_extern("void", "fm_s_push", T.access_ptr(l1[129024 + sid * 512], "w"), T.access_ptr(s[0], "r"))
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 1)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 0)
                        if k4 == nb - 1:
                            if sink and job4 == 0:
                                T.ascend_wait_flag("MTE2_V", 2)
                            T.ascend_get_buf("PIPE_V", 2 + job4 % 2)
                            with T.SimdVF():
                                rmax, unused = T.simd.vdintlv(T.simd.vld(maxima[job4 % 2, 0]), T.simd.vld(maxima[job4 % 2, 0]))
                                smax, unused1 = T.simd.vdintlv(T.simd.vld(scaling_max[job4 % 2, 0]), T.simd.vld(scaling_max[job4 % 2, 0]))
                                rsum, unused2 = T.simd.vdintlv(T.simd.vld(sums[job4 % 2, 0]), T.simd.vld(sums[job4 % 2, 0]))
                                invalid = T.simd.vcmps(rmax, -3.4028234663852886e38, op="eq")
                                maximum = T.simd.vmuls(rmax, scale)
                                scaling = T.simd.vmuls(smax, scale)
                                logsum = T.simd.vadd(T.simd.vln(rsum), scaling)
                                denom = T.alloc_var("float32x64", init=rsum)
                                if sink:
                                    sinkprob = T.simd.vexpdif(T.simd.vld(attn_sink[0]), scaling)
                                    denom = T.simd.vadd(denom, sinkprob)
                                reciprocal = T.simd.vdiv(T.simd.vdup(1.0, "float32"), denom)
                                T.simd.vsts(denorm[job4 % 2, 0], T.simd.vsel(T.simd.vdup(0.0, "float32"), reciprocal, invalid))
                                T.simd.vsts(maxima[job4 % 2, 0], T.simd.vsel(T.simd.vdup(float("-inf"), "float32"), maximum, invalid))
                                T.simd.vsts(sums[job4 % 2, 0], T.simd.vsel(T.simd.vdup(float("inf"), "float32"), logsum, invalid))
                            T.ascend_rls_buf("PIPE_V", 2 + job4 % 2)
                            T.ascend_get_buf("PIPE_MTE3", 2 + job4 % 2)
                            if not decode:
                                T.copy(maxima[job4 % 2, :32], Max[bx + job4 * 32, sid * 32 : sid * 32 + 32])
                            T.copy(sums[job4 % 2, :32], LSE[bx + job4 * 32, sid * 32 : sid * 32 + 32])
                            T.ascend_rls_buf("PIPE_MTE3", 2 + job4 % 2)
                    if decode:  # noqa: SIM102 - compile-time format dispatch
                        # Stage 0: fetch the next index group.
                        if step < total:
                            job0 = step // nb
                            k0 = step % nb
                            q0 = bx + job0 * 32
                            extra0 = k0 >= nb0
                            rel0 = T.if_then_else(extra0, k0 - nb0, k0)
                            if rel0 % (ib // 64) == 0:
                                T.ascend_get_buf("PIPE_MTE2", 0)
                                if extra_topk > 0 and extra0:
                                    length0 = T.min(ib, extra_topk - rel0 * 64)
                                    T.copy(ExtraIndices[q0, rel0 * 64 : rel0 * 64 + length0], indices[:length0])
                                else:
                                    length0 = T.min(ib, topk - rel0 * 64)
                                    T.copy(Indices[q0, rel0 * 64 : rel0 * 64 + length0], indices[:length0])
                                T.ascend_rls_buf("PIPE_MTE2", 0)
                    # Prepare gather2 parameters for the selected cache segment.
                    job0 = step // nb
                    k0 = step % nb
                    extra0 = k0 >= nb0
                    rel0 = T.if_then_else(extra0, k0 - nb0, k0)
                    if step < total and rel0 % (ib // 64) == 0:
                        idx0 = T.alloc_var(
                            "int32", init=(job0 * groups + T.if_then_else(extra0, groups0, 0) + rel0 // (ib // 64)) % index_versions
                        )
                        main_length = T.min(topk, Lengths[bx + job0 * 32]) if variable_lengths else topk
                        extra_length = T.min(extra_topk, ExtraLengths[bx + job0 * 32]) if variable_extra_lengths else extra_topk
                        valid_len = T.alloc_var("int32", init=T.max(T.if_then_else(extra0, extra_length, main_length) - rel0 * 64, 0))
                        T.ascend_get_buf("PIPE_V", 0)
                        T.ascend_get_buf("PIPE_V", 1)
                        with T.SimdVF():
                            for i in rolled(ib // 128):
                                x0 = T.reinterpret("uint32x64", T.simd.vld(indices[i * 128]))
                                x1 = T.reinterpret("uint32x64", T.simd.vld(indices[i * 128 + 64]))
                                pos0 = T.simd.vci(i * 128, "int32")
                                pos1 = T.simd.vci(i * 128 + 64, "int32")
                                sentinel = T.simd.vdup(valid_limit, "uint32")
                                x0 = T.simd.vsel(x0, sentinel, T.simd.vcmps(pos0, valid_len, op="lt"))
                                x1 = T.simd.vsel(x1, sentinel, T.simd.vcmps(pos1, valid_len, op="lt"))
                                even, odd = T.simd.vdintlv(x0, x1)
                                lo = T.simd.vmin(even, odd)
                                hi = T.simd.vmax(even, odd)
                                lo = T.simd.vmins(lo, valid_limit)
                                difference = T.simd.vsub(hi, lo, T.simd.vcmps(hi, valid_limit, op="lt"))
                                T.simd.vsts(smaller[idx0, i * 64], lo)
                                T.simd.vsts(larger[idx0, i * 64], hi)
                                if decode:
                                    # Quantized records are contiguous within each page.
                                    # The wrapper currently requires compact page storage.
                                    bytes0 = T.if_then_else(extra0, extra_record_bytes, record_bytes)
                                    byte_offset = T.simd.vmuls(lo, bytes0)
                                    valid_small = T.simd.vcmps(lo, valid_limit, op="lt")
                                    byte_offset = T.simd.vsel(byte_offset, T.simd.vdup(T.uint32(0xFFFFFFFF), "uint32"), valid_small)
                                    T.simd.vsts(offsets[i * 64], byte_offset)
                                    T.simd.vsts(strides[i * 64], T.simd.vmuls(difference, bytes0))
                                else:
                                    T.simd.vsts(offsets[i * 64], T.simd.vmuls(lo, 512))
                                    T.simd.vsts(strides[i * 64], T.simd.vmuls(difference, 1024))
                        T.ascend_rls_buf("PIPE_V", 1)
                        T.ascend_rls_buf("PIPE_V", 0)
                    # Stage 5: rescale the UB accumulator only when requested.
                    if step >= 5 and step - 5 < total:
                        t5 = step - 5
                        job5 = T.alloc_var("int32", init=t5 // nb)
                        k5 = T.alloc_var("int32", init=t5 % nb)
                        if k5 == 0:
                            emitted = 0
                        rescale5 = T.alloc_var("int32", init=0)
                        if k5 > 0:
                            T.ascend_cross_core_wait_flag(4, "PIPE_S", 2)
                            rescale5 = T.call_extern("int32", "fm_ss_get", (job5 % 8) * 8 + k5 % 8)
                        if rescale5:
                            for d in rolled(4):
                                T.ascend_get_buf("PIPE_V", 4 + d)
                                T.ascend_cross_core_wait_flag(4, "PIPE_V", 4 + d % 2)
                                if emitted > 0:
                                    scale_output_vf(True, d, job5, k5, factors, out, accum)
                                else:
                                    scale_output_vf(False, d, job5, k5, factors, out, accum)
                                T.ascend_cross_core_set_flag(4, "PIPE_V", 4 + d % 2)
                                T.ascend_rls_buf("PIPE_V", 4 + d)
                            emitted = emitted + 1
                        if k5 == nb - 1:
                            for d in T.unroll(4):
                                T.ascend_get_buf("PIPE_V", 4 + d)
                                T.ascend_cross_core_wait_flag(4, "PIPE_V", 4 + d % 2)
                                if emitted > 0:
                                    final_output_vf(True, d, job5, denorm, out, accum, final)
                                else:
                                    final_output_vf(False, d, job5, denorm, out, accum, final)
                                T.ascend_cross_core_set_flag(4, "PIPE_V", 4 + d % 2)
                                T.ascend_rls_buf("PIPE_V", 4 + d)
                                T.ascend_get_buf("PIPE_MTE3", 4 + d)
                                T.call_extern(
                                    "void",
                                    "fm_write_o",
                                    T.access_ptr(O[bx + job5 * 32, sid * 32, d * 128], "w"),
                                    T.access_ptr(final[d * 256], "r"),
                                )
                                T.ascend_rls_buf("PIPE_MTE3", 4 + d)
                    gather_k = step % nb
                    gather_rel = T.if_then_else(gather_k >= nb0, gather_k - nb0, gather_k)
                    gather_later = step < total and gather_rel % (ib // 64) == 0
                    if gather_later and step >= 4 and step - 4 < total and (step - 4) % nb > 0:
                        t4 = step - 4
                        T.ascend_wait_flag("V_S", 0)
                        T.call_extern(
                            "void", "fm_ss_set", (t4 // nb % 8) * 8 + t4 % nb % 8, sid, delta[t4 // nb % 2, t4 % nb % 2, 0] > 6.0 / scale
                        )
                        T.ascend_cross_core_set_flag(4, "PIPE_S", 2)
                    # Stage 0: 16 two-row DMA instructions per AIV.
                    if step < total:
                        k0 = step % nb
                        extra0 = k0 >= nb0
                        rel0 = T.if_then_else(extra0, k0 - nb0, k0)
                        start = ((rel0 % (ib // 64)) * 64 + sid * 32) // 2
                        T.ascend_get_buf("PIPE_MTE2", 8 + step % 3)
                        T.ascend_get_buf("PIPE_S", 1)
                        for i in T.unroll(16):
                            off = offsets[start + i]
                            if decode:
                                count0 = T.if_then_else(off != T.uint32(0xFFFFFFFF), 2, 0)
                                stride0 = T.Cast("int64", strides[start + i])
                                # Clamp the unused pointer so bounds legalization retains
                                # a zero-burst request, while no invalid memory is read.
                                safe_off = T.if_then_else(count0 == 0, T.uint32(0), off)
                                if extra_topk > 0 and extra0:
                                    if extra_fp4:
                                        T.call_extern(
                                            "void",
                                            "fm_pair_bytes",
                                            T.access_ptr(raw4[step % 3, i * 2, 0], "w"),
                                            T.access_ptr(ExtraKV[safe_off], "r"),
                                            count0,
                                            stride0,
                                            288,
                                            288,
                                            cache_hint,
                                        )
                                    else:
                                        T.call_extern(
                                            "void",
                                            "fm_pair_bytes",
                                            T.access_ptr(raw8[step % 3, i * 2, 0], "w"),
                                            T.access_ptr(ExtraKV[safe_off], "r"),
                                            count0,
                                            stride0,
                                            528,
                                            544,
                                            cache_hint,
                                        )
                                else:
                                    if fp4:
                                        T.call_extern(
                                            "void",
                                            "fm_pair_bytes",
                                            T.access_ptr(raw4[step % 3, i * 2, 0], "w"),
                                            T.access_ptr(KV[safe_off], "r"),
                                            count0,
                                            stride0,
                                            288,
                                            288,
                                            cache_hint,
                                        )
                                    else:
                                        T.call_extern(
                                            "void",
                                            "fm_pair_bytes",
                                            T.access_ptr(raw8[step % 3, i * 2, 0], "w"),
                                            T.access_ptr(KV[safe_off], "r"),
                                            count0,
                                            stride0,
                                            528,
                                            544,
                                            cache_hint,
                                        )
                            else:
                                T.call_extern(
                                    "void",
                                    "fm_pair",
                                    T.access_ptr(kv[step % 3, i * 2, 0], "w"),
                                    T.access_ptr(KV[off // 512, 0], "r"),
                                    T.if_then_else(off != nk * 512, 2, 0),
                                    T.Cast("int64", strides[start + i]),
                                )
                        T.ascend_rls_buf("PIPE_S", 1)
                        T.ascend_rls_buf("PIPE_MTE2", 8 + step % 3)
                    if not gather_later and step >= 4 and step - 4 < total and (step - 4) % nb > 0:
                        t4 = step - 4
                        T.ascend_wait_flag("V_S", 0)
                        T.call_extern(
                            "void", "fm_ss_set", (t4 // nb % 8) * 8 + t4 % nb % 8, sid, delta[t4 // nb % 2, t4 % nb % 2, 0] > 6.0 / scale
                        )
                        T.ascend_cross_core_set_flag(4, "PIPE_S", 2)
                T.ascend_pipe_barrier("PIPE_ALL")

    return main
