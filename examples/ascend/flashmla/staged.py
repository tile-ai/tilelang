"""FlashMLA with logical tiles and compiler-managed buffer versions.

A persistent KV stream spans query jobs. Q and the online state live until
the query's last KV tile; output chunks own their Fixpipe/Vector handoff.
Q uses two explicit query slots when its periodic ownership fits the event
budget; all per-tile buffer versions are compiler-managed.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_l0c_layout, make_ascend_major_k_layout, make_ascend_major_mn_layout
from examples.ascend.flashmla.quantization import dequantize

PASS_CONFIG = {
    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}


def rolled(n):
    return T.unroll(n, annotations={"pragma_unroll_factor": 1})


def sparse_attention(
    nq,
    nk,
    topk,
    *,
    sink=True,
    variable_lengths=False,
    scale=512 ** (-0.5),
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
    pipeline_stages=True,
):
    if nq <= 0 or nk <= 0 or topk <= 0 or topk % 64:
        raise ValueError("nq/nk must be positive and topk a positive multiple of 64")
    if kv_format not in ("bf16", "fp8") or extra_format not in ("fp8", "fp4"):
        raise ValueError("Supported formats are BF16 prefill and FP8 main / FP8 or FP4 extra decode")
    if extra_topk < 0 or extra_topk % 64:
        raise ValueError("extra_topk must be a nonnegative multiple of 64")
    if max(kv_storage_bytes, extra_storage_bytes) >= 2**32:
        raise ValueError("KV byte offsets must fit uint32")
    decode = kv_format != "bf16"
    extra_fp4 = extra_format == "fp4"
    valid_limit = T.uint32(2147483648) if decode else nk
    q_strides = q_strides or (64 * 512, 512, 1)
    index_stride = index_stride or topk
    extra_index_stride = extra_index_stride or max(extra_topk, 1)
    nb0 = topk // 64
    nb = nb0 + extra_topk // 64
    # Explicit query slots repeat every 2*nb lexical KV iterations.
    # The two AIC Q ownership directions need 2*(nb+1) mutex IDs, capped at 32.
    q_prefetch = pipeline_stages and 4 <= nb <= 15
    q_slots = 2 if q_prefetch else 1
    q_lead_shift = nb - 4 if q_prefetch else 0
    extra_metadata_lead = pipeline_stages and q_lead_shift > 0
    index_group = min(640, max(topk, extra_topk))
    gather_cache_policy = "normal_lv" if not decode or cache_hint else "notalloc_clean"
    direct_query_masks = not decode and pipeline_stages and extra_topk == 0 and 5 <= nb0 <= 10
    query_mask_stride = index_group // 2 + 32

    @T.macro
    def pv_gemm(k, unit_flag, clear, d, aq, bv, cpv):
        # Keep the zero-flag path explicit so intermediate rescale drains
        # retain their ordinary M/FIX synchronization.
        with T.Task():
            if (unit_flag != 0) if nb == 2 else (k == 0 or k == nb - 1):
                T.gemm(
                    aq[:, :64],
                    bv,
                    cpv[d, :, :],
                    clear_accum=clear,
                    unit_flag_ctrl=unit_flag if nb == 2 else T.if_then_else(k == nb - 1, 3, 2),
                )
            else:
                T.gemm(aq[:, :64], bv, cpv[d, :, :], clear_accum=clear, unit_flag_ctrl=0)

    @T.macro
    def softmax_vf(first, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta, mask_base=0):
        with T.SimdVF():
            full = T.simd.pset(32)
            eight = T.simd.pset(32, "PAT_VL8")
            one = T.simd.pset(32, "PAT_VL1")
            masks = T.alloc_local((4,), "boolx256")
            for col in T.unroll(4):
                lo = T.simd.vld(T.access_ptr(smaller[mask_base + col * 8], "r", extent=8), "BLK")
                hi = T.simd.vld(T.access_ptr(larger[mask_base + col * 8], "r", extent=8), "BLK")
                paired, ignored = T.simd.vintlv(lo, hi)
                masks[col] = T.simd.vcmps(paired, valid_limit, op="lt")
            old_sum = T.simd.vld(sums[0])
            old_scale_max = T.simd.vld(scaling_max[0])
            for r in rolled(8):
                rm = T.alloc_var("float32x64")
                rm = T.simd.vdup(-3.4028234663852886e38, "float32")
                if not first:
                    rm = T.simd.vld(maxima[r * 8], "E2B_B32")
                for col in T.unroll(4):
                    x = T.simd.vld(p[r * 4, col * 16])
                    rm = T.simd.vmax(rm, x, masks[col], mode="MODE_MERGING")
                reduced = T.simd.vcgmax(rm)
                even, odd = T.simd.vdintlv(reduced, reduced)
                rmax = T.simd.vmax(even, odd)
                repeated, ignored2 = T.simd.vintlv(rmax, rmax)
                T.simd.vsts(maxima[r * 8], repeated, eight)
            T.simd.mem_bar("VST_VLD")
            new_max = T.simd.vld(maxima[0])
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
                T.simd.vsts(delta[0], greatest, one)
            T.simd.vsts(factors[0], factor)
            T.simd.vsts(scaling_max[0], chosen_max)
            T.simd.mem_bar("VST_VLD")
            for r in rolled(4):
                rmax2 = T.alloc_local((2,), "float32x64")
                rsum2 = T.alloc_local((2,), "float32x64")
                packed2 = T.alloc_local((2, 2), "bfloat16x128")
                for rr in T.unroll(2):
                    rmax2[rr] = T.simd.vmuls(T.simd.vld(scaling_max[r * 16 + rr * 8], "E2B_B32"), scale)
                for pair in T.unroll(2):
                    for cc in T.unroll(2):
                        for rr in T.unroll(2):
                            x = T.simd.vld(p[r * 8 + rr * 4, (pair * 2 + cc) * 16])
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
                    T.simd.vsts(s[r * 8, pair * 32], col0)
                    T.simd.vsts(s[r * 8, pair * 32 + 16], col1)
                for rr in T.unroll(2):
                    reduced_sum = T.simd.vcpadd(T.simd.vcgadd(rsum2[rr]))
                    repeated_sum, ignored3 = T.simd.vintlv(reduced_sum, reduced_sum)
                    T.simd.vsts(sums[r * 16 + rr * 8], repeated_sum, eight)
            T.simd.mem_bar("VST_VLD")
            if not first:
                current_sum = T.simd.vld(sums[0])
                updated_sum = T.simd.vadd(T.simd.vmul(old_sum, factor), current_sum)
                T.simd.vsts(sums[0], updated_sum)

    @T.macro
    def scale_output_vf(add, d, factors, out, accum):
        with T.SimdVF():
            for r in rolled(32):
                factor5 = T.simd.vld(factors[r * 2], "BRC_B32")
                for j in T.unroll(2):
                    cur = T.alloc_var("float32x64", init=T.simd.vld(out[r, j * 64]))
                    if add:
                        old = T.simd.vld(accum[r, d * 128 + j * 64])
                        cur = T.simd.vadd(old, cur)
                    T.simd.vsts(accum[r, d * 128 + j * 64], T.simd.vmul(cur, factor5))

    @T.macro
    def final_output_vf(add, d, denorm, out, accum, final):
        with T.SimdVF():
            mask16 = T.simd.pset(16)
            mask32 = T.simd.pset(32)
            for r in rolled(16):
                packed = T.alloc_local((2, 2), "bfloat16x128")
                for rr in T.unroll(2):
                    norm = T.simd.vld(denorm[r * 2 + rr], "BRC_B32")
                    for cc in T.unroll(2):
                        x = T.simd.vld(out[r * 2 + rr, cc * 64])
                        if add:
                            old = T.simd.vld(accum[r * 2 + rr, d * 128 + cc * 64])
                            x = T.simd.vadd(x, old, mask32)
                        scaled = T.simd.vmul(x, norm, mask32)
                        if rr == 0:
                            packed[rr, cc] = T.simd.vcvt(scaled, "bfloat16", mask32, part=0)
                        else:
                            packed[rr, cc] = T.simd.vcvt(scaled, "bfloat16", mask32, part=1)
                merged0 = T.simd.vadd(packed[0, 0], packed[1, 0], mask16)
                merged1 = T.simd.vadd(packed[0, 1], packed[1, 1], mask16)
                row0, row1 = T.simd.vdintlv(merged0, merged1)
                T.simd.vsts(final[r * 2, d * 128], row0, mask16)
                T.simd.vsts(final[r * 2 + 1, d * 128], row1, mask16)

    @T.macro
    def load_metadata(
        k, job, qidx, extra, Indices, ExtraIndices, Lengths, ExtraLengths, indices, group_smaller, group_larger, smaller, larger
    ):
        rel = T.alloc_var("int32")
        rel = T.if_then_else(extra, k - nb0, k)
        group_tile = T.alloc_var("int32")
        group_tile = rel % (index_group // 64)
        if k == 0 if direct_query_masks else group_tile == 0:
            if extra_topk > 0 and extra:
                group_length = T.min(index_group, extra_topk - rel * 64)
                T.copy(ExtraIndices[qidx, rel * 64 : rel * 64 + group_length], indices[:group_length])
            else:
                group_length = T.min(index_group, topk - rel * 64)
                T.copy(Indices[qidx, rel * 64 : rel * 64 + group_length], indices[:group_length])
            main_length = T.min(topk, Lengths[qidx]) if variable_lengths else topk
            extra_length = T.min(extra_topk, ExtraLengths[qidx]) if variable_extra_lengths else extra_topk
            valid_len = T.max(T.if_then_else(extra, extra_length, main_length) - rel * 64, 0)
            with T.SimdVF():
                pairs32 = T.simd.pset(32, "PAT_VL32")
                for group_k in rolled(index_group // 64):
                    x = T.reinterpret("uint32x64", T.simd.vld(indices[group_k * 64]))
                    pos = T.simd.vci(group_k * 64, "int32")
                    x = T.simd.vsel(x, T.simd.vdup(valid_limit, "uint32"), T.simd.vcmps(pos, valid_len, op="lt"))
                    even, odd = T.simd.vdintlv(x, x)
                    lo = T.simd.vmins(T.simd.vmin(even, odd), valid_limit)
                    hi = T.simd.vmax(even, odd)
                    if direct_query_masks:
                        T.simd.vsts(group_smaller[(job % 2) * query_mask_stride + group_k * 32], lo, pairs32, extent=32)
                        T.simd.vsts(group_larger[(job % 2) * query_mask_stride + group_k * 32], hi, pairs32, extent=32)
                    else:
                        T.simd.vsts(group_smaller[group_k * 32], lo, pairs32, extent=32)
                        T.simd.vsts(group_larger[group_k * 32], hi, pairs32, extent=32)
        if not direct_query_masks:
            # Snapshot only the active tile's masks; the index group may
            # be replaced while a previous tile is still in softmax.
            with T.SimdVF():
                pairs32 = T.simd.pset(32, "PAT_VL32")
                T.simd.vsts(smaller[0], T.simd.vld(group_smaller[group_tile * 32]), pairs32, extent=32)
                T.simd.vsts(larger[0], T.simd.vld(group_larger[group_tile * 32]), pairs32, extent=32)

    @T.macro
    def gather_kv(k, job, sid, extra, KV, ExtraKV, group_smaller, group_larger, smaller, larger, kv, raw8, raw4):
        for i in T.unroll(16):
            with T.Stage(0):
                if direct_query_masks:
                    lo = group_smaller[(job % 2) * query_mask_stride + k * 32 + sid * 16 + i]
                    hi = group_larger[(job % 2) * query_mask_stride + k * 32 + sid * 16 + i]
                else:
                    lo = smaller[sid * 16 + i]
                    hi = larger[sid * 16 + i]
                second = T.if_then_else(hi < valid_limit, hi, lo)
                n_burst = T.alloc_var("int32")
                n_burst = T.if_then_else(lo < valid_limit, 2, 0)
                safe_lo = T.if_then_else(n_burst == 0, T.uint32(0), lo)
                stride = T.alloc_var("int64")
                stride = T.Cast("int64", second - lo)
                if decode and extra_topk > 0 and extra_fp4:
                    # Both formats retain separate bytes, sharing one
                    # MTE2 task and one V consumer ownership boundary.
                    with T.Task():
                        if extra:
                            extra_pairs = T.StridedTensor(
                                (extra_storage_bytes // 288, 2, 288),
                                (288, stride * 288, 1),
                                "uint8",
                                data=ExtraKV.data,
                            )
                            for _ in T.Serial(1):
                                T.assume(n_burst > 0)
                                T.assume(n_burst == 0 or safe_lo < extra_pairs.shape[0])
                                T.copy(
                                    extra_pairs[safe_lo, :n_burst, :],
                                    raw4[i * 2 : i * 2 + n_burst, :],
                                    l2_cache_ctrl=gather_cache_policy,
                                )
                        else:
                            main_pairs = T.StridedTensor(
                                (kv_storage_bytes // 528, 2, 528),
                                (528, stride * 528, 1),
                                "uint8",
                                data=KV.data,
                            )
                            for _ in T.Serial(1):
                                T.assume(n_burst > 0)
                                T.assume(n_burst == 0 or safe_lo < main_pairs.shape[0])
                                T.copy(
                                    main_pairs[safe_lo, :n_burst, :],
                                    raw8[i * 2 : i * 2 + n_burst, :528],
                                    l2_cache_ctrl=gather_cache_policy,
                                )
                elif decode:
                    if extra_topk > 0 and extra:
                        extra_pairs = T.StridedTensor(
                            (extra_storage_bytes // 528, 2, 528),
                            (528, stride * 528, 1),
                            "uint8",
                            data=ExtraKV.data,
                        )
                        # Limit DMA assumptions to this copy.
                        for _ in T.Serial(1):
                            with T.Stage(0):
                                T.assume(n_burst > 0)
                                T.assume(n_burst == 0 or safe_lo < extra_pairs.shape[0])
                                T.copy(
                                    extra_pairs[safe_lo, :n_burst, :],
                                    raw8[i * 2 : i * 2 + n_burst, :528],
                                    l2_cache_ctrl=gather_cache_policy,
                                )
                    else:
                        main_pairs = T.StridedTensor(
                            (kv_storage_bytes // 528, 2, 528),
                            (528, stride * 528, 1),
                            "uint8",
                            data=KV.data,
                        )
                        # Limit DMA assumptions to this copy.
                        for _ in T.Serial(1):
                            with T.Stage(0):
                                T.assume(n_burst > 0)
                                T.assume(n_burst == 0 or safe_lo < main_pairs.shape[0])
                                T.copy(
                                    main_pairs[safe_lo, :n_burst, :],
                                    raw8[i * 2 : i * 2 + n_burst, :528],
                                    l2_cache_ctrl=gather_cache_policy,
                                )
                else:
                    main_pairs = T.StridedTensor(
                        (nk, 2, 512),
                        (512, stride * 512, 1),
                        "bfloat16",
                        data=KV.data,
                    )
                    # Limit DMA assumptions to this copy.
                    for _ in T.Serial(1):
                        with T.Stage(0):
                            T.assume(n_burst > 0)
                            T.assume(n_burst == 0 or safe_lo < main_pairs.shape[0])
                            T.copy(
                                main_pairs[safe_lo, :n_burst, :],
                                kv[i * 2 : i * 2 + n_burst, :],
                                l2_cache_ctrl=gather_cache_policy,
                            )

    @T.macro
    def attention_body(Q, KV, Indices, Sink, Lengths, ExtraKV, ExtraIndices, ExtraLengths, O, Max, LSE):
        with T.MixedKernel(min(nq, 32)) as (bx, sid):
            q_l1 = T.alloc_l1((q_slots, 64, 512), "bfloat16")
            kv_l1_0 = T.alloc_l1((32, 512), "bfloat16")
            kv_l1_1 = T.alloc_l1((32, 512), "bfloat16")
            s_l1 = T.alloc_l1((64, 64), "bfloat16")
            aq = T.alloc_l0a((64, 128), "bfloat16")
            bq = T.alloc_l0b((64, 128), "bfloat16")
            # QK and PV keep distinct layouts on one physical versioned ring.
            bv = T.view(bq, shape=(64, 128))
            cq = T.alloc_l0c((64, 64), "float32")
            # Four disjoint output-column tiles, each accumulated across all KV tiles.
            cpv = T.alloc_l0c((4, 64, 128), "float32")
            kv = T.alloc_shared((32, 1 if decode else 512), "bfloat16")
            raw8 = T.alloc_shared((32, 544 if decode else 1), "uint8")
            raw4 = T.alloc_shared((32, 288 if decode and extra_fp4 and extra_topk else 1), "uint8")
            scales = T.alloc_shared((1024 if decode else 1,), "bfloat16")
            nz = T.alloc_shared((33, 512), "bfloat16")
            p = T.alloc_shared((32, 64), "float32")
            s = T.alloc_shared((32, 64), "bfloat16")
            accum = T.alloc_shared((32, 512), "float32")
            out = T.alloc_shared((32, 128), "float32")
            # Pack consumed FP32 chunks into the row's first 512 BF16 elements.
            # Drain the complete row only after every d chunk is normalized.
            final = T.view(accum, shape=(32, 1024), dtype="bfloat16")
            maxima = T.alloc_shared((64,), "float32")
            scaling_max = T.alloc_shared((64,), "float32")
            sums = T.alloc_shared((64,), "float32")
            denorm = T.alloc_shared((64,), "float32")
            factors = T.alloc_shared((64,), "float32")
            delta = T.alloc_shared((64,), "float32")
            attn_sink = T.alloc_shared((64,), "float32")
            indices = T.alloc_shared((index_group,), "int32")
            group_smaller = T.alloc_shared(((2 if direct_query_masks else 1) * query_mask_stride,), "uint32")
            group_larger = T.alloc_shared(((2 if direct_query_masks else 1) * query_mask_stride,), "uint32")
            if not direct_query_masks:
                smaller = T.alloc_shared((32,), "uint32")
                larger = T.alloc_shared((32,), "uint32")
            ss = T.alloc_ss((2,), "uint32")
            ss_pairs = T.view(ss, shape=(1,), dtype="uint64")
            ss_result = T.alloc_ss((1,), "uint32")
            T.annotate_l1_offsets(
                {
                    kv_l1_0: 0,
                    q_l1: 131072,
                    kv_l1_1: 262144,
                    s_l1: 393216,
                }
            )
            T.annotate_layout(
                {
                    aq: make_ascend_major_k_layout(aq),
                    bq: make_ascend_major_k_layout(bq),
                    bv: make_ascend_major_mn_layout(bv),
                    q_l1: make_ascend_major_k_layout(q_l1),
                    kv_l1_0: make_ascend_major_k_layout(kv_l1_0),
                    kv_l1_1: make_ascend_major_k_layout(kv_l1_1),
                    s_l1: make_ascend_major_k_layout(s_l1),
                    nz: make_ascend_compact_nz_layout(nz),
                    s: make_ascend_compact_nz_layout(s),
                    p: make_ascend_l0c_layout(p),
                }
            )
            if q_prefetch:
                T.annotate_manual_multi_buffer({q_l1: 2 * nb})
            else:
                T.annotate_buffer_versions({q_l1: 1})
            T.annotate_buffer_versions(
                {
                    kv_l1_0: 4,
                    kv_l1_1: 4,
                    s_l1: 2,
                    aq: 4,
                    bq: 4,
                    cq: 1,
                    cpv: 1,
                    kv: 1 if decode else 3,
                    raw8: 3 if decode else 1,
                    raw4: 3 if decode and extra_fp4 and extra_topk else 1,
                    nz: 1,
                    scales: 1,
                    p: 1,
                    s: 1,
                    accum: 1,
                    out: 2,
                    final: 1,
                    maxima: 1,
                    scaling_max: 1,
                    sums: 1,
                    denorm: 2,
                    factors: 2,
                    delta: 1,
                    indices: 1,
                    group_smaller: 1,
                    group_larger: 1,
                    **(
                        {}
                        if direct_query_masks
                        else {
                            smaller: 6 if extra_metadata_lead else 5,
                            larger: 6 if extra_metadata_lead else 5,
                        }
                    ),
                    attn_sink: 1,
                    ss: 2,
                    ss_result: 1,
                }
            )
            if direct_query_masks:
                T.annotate_manual_multi_buffer({group_smaller: 2 * nb, group_larger: 2 * nb})
            if sink:
                T.copy(Sink[sid * 32 : sid * 32 + 32], attn_sink[:32])
            # Initialize every automatic version once; zero-burst DMA preserves it.
            with T.SimdVF():
                if decode:
                    T.fill(raw8, 0)
                    if extra_topk > 0 and extra_fp4:
                        T.fill(raw4, 0)
                else:
                    T.fill(kv, 0)
            emitted = T.alloc_var("int32", init=0)
            # Each nested scheduler loop is manual independently. VF loops
            # belong to their enclosing VF task and cannot contain T.Stage.
            for step in T.Pipelined(T.ceildiv(nq - bx, 32) * nb, num_stages=6 + q_lead_shift, annotations={"enable_offset": True}):
                job = step // nb
                k = step % nb
                qidx = bx + job * 32
                with T.Stage(0):
                    if q_prefetch and k == 0:
                        T.copy(Q[qidx, :, :], q_l1[job % q_slots, :, :])
                # Flat execution needs the producer tiles before their consumers.
                if not pipeline_stages:
                    with T.Stage(q_lead_shift - (1 if extra_metadata_lead else 0)):
                        extra = k >= nb0
                        load_metadata(
                            k,
                            job,
                            qidx,
                            extra,
                            Indices,
                            ExtraIndices,
                            Lengths,
                            ExtraLengths,
                            indices,
                            group_smaller,
                            group_larger,
                            None if direct_query_masks else smaller,
                            None if direct_query_masks else larger,
                        )
                    with T.Stage(q_lead_shift + (0 if extra_metadata_lead else 1) if pipeline_stages else 0):
                        gather_kv(
                            k,
                            job,
                            sid,
                            extra,
                            KV,
                            ExtraKV,
                            group_smaller,
                            group_larger,
                            None if direct_query_masks else smaller,
                            None if direct_query_masks else larger,
                            kv,
                            raw8,
                            raw4,
                        )
                # In the pipeline, source order follows issue order; Stage still
                # identifies the logical tile used by dependency analysis.
                with T.Stage(2 + q_lead_shift if pipeline_stages else 0):
                    if decode:
                        if extra_topk > 0 and extra_fp4:
                            with T.Task():
                                if k >= nb0:
                                    dequantize(raw4, scales, nz, None, True)
                                else:
                                    dequantize(raw8, scales, nz, None, False)
                        else:
                            dequantize(raw8, scales, nz, None, False)
                    else:
                        with T.SimdVF():
                            nd_mask = T.simd.pset(16)
                            src_ptr = T.simd.make_ubuf_ptr(T.access_ptr(kv[0, 0], "r", extent=16384), "bfloat16")
                            dst0 = T.simd.make_ubuf_ptr(T.access_ptr(nz[0, 0], "w", extent=16896), "bfloat16")
                            dst1 = T.simd.make_ubuf_ptr(T.access_ptr(nz[0, 128], "w", extent=12672), "bfloat16")
                            for _col in rolled(2):
                                for row in T.unroll(32):
                                    x0, src_ptr = T.simd.vld(src_ptr, post_inc=128)
                                    x1, src_ptr = T.simd.vld(src_ptr, post_inc=T.if_then_else(row == 31, -15744, 384))
                                    config = (33 << 16) | T.if_then_else(row == 31, 497, 1)
                                    dst0 = T.simd.vsstb(x0, dst0, config, nd_mask, update=True)
                                    dst1 = T.simd.vsstb(x1, dst1, config, nd_mask, update=True)
                    if sid == 0:
                        T.copy(nz[:32, :], kv_l1_0)
                    else:
                        T.copy(nz[:32, :], kv_l1_1)
                with T.Stage(4 + q_lead_shift if pipeline_stages else 0):
                    # Adjacent Cube configuration regions follow physical phase.
                    if not q_prefetch and k == 0:
                        T.copy(Q[qidx, :, :], q_l1[0, :, :])
                    T.set_mmad_direction("m")
                    for d in T.Pipelined(4, num_stages=4):
                        with T.Stage(0):
                            T.copy(q_l1[job % q_slots, :, d * 128 : (d + 1) * 128], aq)
                            with T.Task():
                                T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bq[:32, :], l0_pitch=4)
                                T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bq[32:, :], l0_pitch=4)
                            T.gemm(aq, bq, cq, transpose_B=True, clear_accum=d == 0, unit_flag_ctrl=T.if_then_else(d == 3, 3, 2))
                    T.dual_copy(cq, p, unit_flag_ctrl=3)
                with T.Stage(4 + q_lead_shift if pipeline_stages else 0):
                    if k == 0:
                        with T.SimdVF():
                            T.fill(maxima, -3.4028234663852886e38)
                            T.fill(scaling_max, 0.0)
                            T.fill(sums, 0.0)
                    delta[0] = T.float32(0)
                    if k == 0:
                        if direct_query_masks:
                            softmax_vf(
                                True,
                                group_smaller,
                                group_larger,
                                p,
                                s,
                                maxima,
                                sums,
                                scaling_max,
                                factors,
                                delta,
                                (job % 2) * query_mask_stride + k * 32,
                            )
                        else:
                            softmax_vf(True, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta)
                    else:
                        if direct_query_masks:
                            softmax_vf(
                                False,
                                group_smaller,
                                group_larger,
                                p,
                                s,
                                maxima,
                                sums,
                                scaling_max,
                                factors,
                                delta,
                                (job % 2) * query_mask_stride + k * 32,
                            )
                        else:
                            softmax_vf(False, smaller, larger, p, s, maxima, sums, scaling_max, factors, delta)
                    T.copy(s[:32, :], s_l1[sid * 32 : sid * 32 + 32, :])
                    if not pipeline_stages:
                        ss[sid] = T.Cast("uint32", delta[0] > 6.0 / scale)
                with T.Stage(4 + q_lead_shift if pipeline_stages else 0):
                    if k == nb - 1:
                        with T.SimdVF():
                            rmax, unused = T.simd.vdintlv(T.simd.vld(maxima[0]), T.simd.vld(maxima[0]))
                            smax, unused1 = T.simd.vdintlv(T.simd.vld(scaling_max[0]), T.simd.vld(scaling_max[0]))
                            rsum, unused2 = T.simd.vdintlv(T.simd.vld(sums[0]), T.simd.vld(sums[0]))
                            invalid = T.simd.vcmps(rmax, -3.4028234663852886e38, op="eq")
                            maximum = T.simd.vmuls(rmax, scale)
                            scaling = T.simd.vmuls(smax, scale)
                            logsum = T.simd.vadd(T.simd.vln(rsum), scaling)
                            denom = T.alloc_var("float32x64", init=rsum)
                            if sink:
                                sinkprob = T.simd.vexpdif(T.simd.vld(attn_sink[0]), scaling)
                                denom = T.simd.vadd(denom, sinkprob)
                            reciprocal = T.simd.vdiv(T.simd.vdup(1.0, "float32"), denom)
                            T.simd.vsts(denorm[0], T.simd.vsel(T.simd.vdup(0.0, "float32"), reciprocal, invalid))
                            T.simd.vsts(maxima[0], T.simd.vsel(T.simd.vdup(float("-inf"), "float32"), maximum, invalid))
                            T.simd.vsts(sums[0], T.simd.vsel(T.simd.vdup(float("inf"), "float32"), logsum, invalid))
                        if not decode:
                            T.copy(maxima[:32], Max[qidx, sid * 32 : sid * 32 + 32])
                        T.copy(sums[:32], LSE[qidx, sid * 32 : sid * 32 + 32])
                if pipeline_stages:
                    with T.Stage(q_lead_shift - (1 if extra_metadata_lead else 0)):
                        extra = k >= nb0
                        load_metadata(
                            k,
                            job,
                            qidx,
                            extra,
                            Indices,
                            ExtraIndices,
                            Lengths,
                            ExtraLengths,
                            indices,
                            group_smaller,
                            group_larger,
                            None if direct_query_masks else smaller,
                            None if direct_query_masks else larger,
                        )
                with T.Stage(5 + q_lead_shift if pipeline_stages else 0):
                    if k == 0:
                        emitted = 0
                    T.set_mmad_direction("n")
                    # Bind this pure predicate before the PV owner also on PIPE_S.
                    last_output = k == nb - 1
                    if nb == 2:
                        # Keep the zero-flag alternative visible to sync
                        # analysis when both KV tiles are boundaries.
                        pv_flag = T.alloc_var("int32", init=T.if_then_else(k == nb - 1, 3, 2))
                    for _pv_region in T.Pipelined(1, num_stages=4):
                        with T.Stage(0):
                            T.copy(s_l1, aq[:, :64])
                            clear = T.alloc_var("int32", init=T.Cast("int32", ss_pairs[0] != T.uint64(0)))
                            ss_result[0] = T.Cast("uint32", clear)
                        for _pv_branches in T.serial(1):
                            if ss_result[0] != T.uint32(0):
                                for d in T.Pipelined(4, num_stages=2):
                                    with T.Stage(0):
                                        T.dual_copy(cpv[d, :, :], out)
                                        if emitted > 0:
                                            scale_output_vf(True, d, factors, out, accum)
                                        else:
                                            scale_output_vf(False, d, factors, out, accum)
                                emitted = emitted + 1
                                for d in T.Pipelined(4, num_stages=2):
                                    with T.Stage(0):
                                        with T.Task():
                                            T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bv[:32, :], l0_pitch=8)
                                            T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bv[32:, :], l0_pitch=8)
                                        pv_gemm(k, pv_flag if nb == 2 else 0, True, d, aq, bv, cpv)
                            else:
                                for d in T.Pipelined(4, num_stages=2):
                                    with T.Stage(0):
                                        with T.Task():
                                            T.copy(kv_l1_0[:, d * 128 : (d + 1) * 128], bv[:32, :], l0_pitch=8)
                                            T.copy(kv_l1_1[:, d * 128 : (d + 1) * 128], bv[32:, :], l0_pitch=8)
                                        pv_gemm(k, pv_flag if nb == 2 else 0, k == 0, d, aq, bv, cpv)
                with T.Stage(5 + q_lead_shift if pipeline_stages else 0):
                    if last_output:
                        for d in T.Pipelined(4, num_stages=2):
                            with T.Stage(0):
                                T.dual_copy(cpv[d, :, :], out, unit_flag_ctrl=3)
                                if emitted > 0:
                                    final_output_vf(True, d, denorm, out, accum, final)
                                else:
                                    final_output_vf(False, d, denorm, out, accum, final)
                        T.copy(final[:, :512], O[qidx, sid * 32 : sid * 32 + 32, :])
                if pipeline_stages:
                    with T.Stage(q_lead_shift + (0 if extra_metadata_lead else 1) if pipeline_stages else 0):
                        gather_kv(
                            k,
                            job,
                            sid,
                            extra,
                            KV,
                            ExtraKV,
                            group_smaller,
                            group_larger,
                            None if direct_query_masks else smaller,
                            None if direct_query_masks else larger,
                            kv,
                            raw8,
                            raw4,
                        )
                    with T.Stage(4 + q_lead_shift):
                        # Source order dispatches gather DMA before the scalar
                        # wait for softmax's delta, despite the higher stage.
                        ss[sid] = T.Cast("uint32", delta[0] > 6.0 / scale)

    if decode:

        @T.prim_func
        def main(
            Q: T.StridedTensor((nq, 64, 512), q_strides, "bfloat16"),
            KV: T.Tensor((kv_storage_bytes,), "uint8"),
            Indices: T.StridedTensor((nq, topk), (index_stride, 1), "int32"),
            Sink: T.Tensor((64,), "float32"),
            Lengths: T.Tensor((nq,), "int32"),
            ExtraKV: T.Tensor((extra_storage_bytes,), "uint8"),
            ExtraIndices: T.StridedTensor((nq, max(extra_topk, 1)), (extra_index_stride, 1), "int32"),
            ExtraLengths: T.Tensor((nq,), "int32"),
            O: T.Tensor((nq, 64, 512), "bfloat16"),
            LSE: T.Tensor((nq, 64), "float32"),
        ):
            attention_body(Q, KV, Indices, Sink, Lengths, ExtraKV, ExtraIndices, ExtraLengths, O, None, LSE)

    else:

        @T.prim_func
        def main(
            Q: T.StridedTensor((nq, 64, 512), q_strides, "bfloat16"),
            KV: T.Tensor((nk, 512), "bfloat16"),
            Indices: T.StridedTensor((nq, topk), (index_stride, 1), "int32"),
            Sink: T.Tensor((64,), "float32"),
            Lengths: T.Tensor((nq,), "int32"),
            O: T.Tensor((nq, 64, 512), "bfloat16"),
            Max: T.Tensor((nq, 64), "float32"),
            LSE: T.Tensor((nq, 64), "float32"),
        ):
            attention_body(Q, KV, Indices, Sink, Lengths, None, None, None, O, Max, LSE)

    return main


def prefill(nq, nk, topk, **kwargs):
    return sparse_attention(nq, nk, topk, **kwargs)


def compile_prefill(nq, nk, topk, **kwargs):
    # UB ND-to-NZ copy layout inference currently needs an active target context.
    with tvm.target.Target("ascend"):
        return tilelang.compile(
            prefill(nq, nk, topk, **kwargs),
            target="ascend",
            out_idx=[5, 6, 7],
            pass_configs=PASS_CONFIG,
            compile_flags=["-Ofast", "-mllvm", "-enable-hiipu-vf-loop-unroll"],
        )
