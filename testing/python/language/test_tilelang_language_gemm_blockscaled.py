"""Block-scaled GEMM surface.

`T.gemm_blockscaled` is a common, auto-dispatching entry. CUDA picks the
block-scaled instruction from the target and accumulator scope (TMEM on
SM100 => TCGEN5MMA, fragment on SM120 => warp-level mma.sync). Backends
without block-scaled support reject the op. `T.tcgen05_gemm_blockscaled`
and `T.mma_gemm_blockscaled` are explicit CUDA variants.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _gemm_calls(func) -> list[tirx.Call]:
    found = []

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")).startswith("tl.tileop.") and "gemm" in str(node.op.name):
            found.append(node)

    post_order_visit(func.body, visit)
    return found


def _annotations(call: tirx.Call) -> dict:
    return {str(k): v for k, v in call.annotations.items()}


def _make_blockscaled_op(gemm_api="gemm_blockscaled", *, a_scope="shared", c_scope="shared.tmem", use_2cta=False, **kwargs):
    ab_dtype = "float8_e4m3fn" if c_scope == "shared.tmem" else "float4_e2m1fn"
    sf_scope = "shared.tmem" if c_scope == "shared.tmem" else "shared"
    sf_granularity = 128 if c_scope == "shared.tmem" else 16
    a = tirx.decl_buffer((128, 128), ab_dtype, name="A", scope=a_scope)
    b = tirx.decl_buffer((128, 128), ab_dtype, name="B", scope="shared")
    c = tirx.decl_buffer((128, 256 if use_2cta else 128), "float32", name="C", scope=c_scope)
    sfa = tirx.decl_buffer((128, 4), "uint32", name="SFA", scope=sf_scope)
    sfb = tirx.decl_buffer((128, 8 if use_2cta else 4), "uint32", name="SFB", scope=sf_scope)
    if gemm_api != "mma_gemm_blockscaled":
        done = tirx.decl_buffer((1,), "uint64", name="done", scope="shared")
        kwargs.update(mbar=done[0], use_2cta=use_2cta)
    call = getattr(T, gemm_api)(
        a,
        b,
        c,
        sfa,
        sfb,
        transpose_B=True,
        k_start=0,
        sf_a_granularity_k=sf_granularity,
        sf_b_granularity_k=sf_granularity,
        **kwargs,
    )
    return call.op.get_attr("TLOpBuilder")(call.args, call.annotations)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "gemm_api, arch, c_scope, use_2cta, expected",
    [
        ("gemm_blockscaled", "sm_100", "shared.tmem", False, "cuda.tcgen05"),
        ("gemm_blockscaled", "sm_100", "shared.tmem", True, "cuda.tcgen05"),
        ("tcgen05_gemm_blockscaled", "sm_100", "shared.tmem", False, "cuda.tcgen05"),
        ("gemm_blockscaled", "sm_120", "local.fragment", False, "cuda.mma.blockscaled"),
        ("mma_gemm_blockscaled", "sm_120", "local.fragment", False, "cuda.mma.blockscaled"),
    ],
)
def test_blockscaled_instruction_selection(gemm_api, arch, c_scope, use_2cta, expected):
    op = _make_blockscaled_op(gemm_api, c_scope=c_scope, use_2cta=use_2cta)
    assert op._select_gemm_instruction(128, tvm.target.Target({"kind": "cuda", "arch": arch})) == expected


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("gemm_api", ["gemm_blockscaled", "tcgen05_gemm_blockscaled"])
@pytest.mark.parametrize("use_2cta", [False, True])
def test_blockscaled_tcgen05_rejects_tmem_a(gemm_api, use_2cta):
    op = _make_blockscaled_op(gemm_api, a_scope="shared.tmem", use_2cta=use_2cta)
    with pytest.raises(Exception, match="Block-scaled GEMM requires.*A/B in shared memory"):
        op._select_gemm_instruction(128, tvm.target.Target({"kind": "cuda", "arch": "sm_100"}))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("gemm_api, use_2cta", [("tcgen05_gemm_blockscaled", False), ("gemm_blockscaled", True)])
def test_blockscaled_tcgen05_request_cannot_select_sm120_mma(gemm_api, use_2cta):
    op = _make_blockscaled_op(gemm_api, c_scope="local.fragment", use_2cta=use_2cta)
    with pytest.raises(Exception, match="Block-scaled GEMM requires Blackwell SM100 TCGEN5MMA"):
        op._select_gemm_instruction(128, tvm.target.Target({"kind": "cuda", "arch": "sm_120"}))


@tilelang.testing.requires_cuda
def test_blockscaled_gemm_rejects_wgmma_annotation():
    op = _make_blockscaled_op(annotations={"is_wgmma": 1})
    with pytest.raises(Exception, match="Block-scaled GEMM does not support WGMMA"):
        op._select_gemm_instruction(128, tvm.target.Target({"kind": "cuda", "arch": "sm_100"}))


# ---------------------------------------------------------------------------
# Call protocol (hardware-free)
# ---------------------------------------------------------------------------


def test_gemm_blockscaled_emits_dispatching_call():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float8_e4m3fn), B: T.Tensor((128, 128), T.float8_e4m3fn)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c = T.alloc_tmem([128, 128], T.float32)
            sfa = T.alloc_tmem([128, 4], T.uint32)
            sfb = T.alloc_tmem([128, 4], T.uint32)
            done = T.alloc_barrier([1])
            T.gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                mbar=done[0],
                k_start=256,
                sf_a_granularity_k=128,
                sf_b_granularity_k=32,
            )

    (call,) = _gemm_calls(main)
    # The unified op has its own op key but does not pin an ISA.
    assert str(call.op.name) == "tl.tileop.gemm_blockscaled"
    # 13 dense slots + SFA + SFB + k_start.
    assert len(call.args) == 16
    assert int(call.args[15]) == 256
    ann = _annotations(call)
    assert int(ann["sf_a_granularity_k"]) == 128
    assert int(ann["sf_b_granularity_k"]) == 32
    assert "is_tcgen05" not in ann
    assert "use_2cta" not in ann


def test_gemm_blockscaled_parses_to_its_own_tile_op():
    """tl.tileop.gemm_blockscaled builds a GemmBlockScaled node.

    The node is a Gemm subclass: passes that only care about "a GEMM" keep
    matching it, while the scale-factor operands are first-class fields
    instead of trailing optional slots on the dense op.
    """
    from tilelang.tileop import Gemm, GemmBlockScaled

    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float8_e4m3fn), B: T.Tensor((128, 128), T.float8_e4m3fn)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c = T.alloc_tmem([128, 128], T.float32)
            sfa = T.alloc_tmem([128, 4], T.uint32)
            sfb = T.alloc_tmem([128, 4], T.uint32)
            done = T.alloc_barrier([1])
            T.gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                mbar=done[0],
                k_start=256,
                sf_a_granularity_k=128,
                sf_b_granularity_k=128,
            )

    (call,) = _gemm_calls(main)
    op = call.op.get_attr("TLOpBuilder")(call.args, call.annotations)
    assert isinstance(op, GemmBlockScaled)
    assert isinstance(op, Gemm)
    assert op.is_blockscaled
    # Inherited dense fields and the block-scaled fields both resolve.
    assert (int(op.m), int(op.n), int(op.k)) == (128, 128, 128)
    assert op.sfaRegion.buffer.name == "sfa"
    assert op.sfbRegion.buffer.name == "sfb"
    assert int(op.sfKStart) == 256
    assert op.transB and not op.transA

    # The dense op refuses the 16-slot protocol rather than ignoring SFA/SFB.
    dense_builder = tvm.ir.Op.get("tl.tileop.gemm").get_attr("TLOpBuilder")
    with pytest.raises(Exception, match="at most 13 positional slots"):
        dense_builder(call.args, call.annotations)


def test_gemm_blockscaled_records_use_2cta_and_sf_layout():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float8_e4m3fn), B: T.Tensor((128, 128), T.float8_e4m3fn)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c = T.alloc_tmem([128, 256], T.float32)
            sfa = T.alloc_tmem([128, 4], T.uint32)
            sfb = T.alloc_tmem([128, 8], T.uint32)
            done = T.alloc_barrier([1])
            T.gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                mbar=done[0],
                use_2cta=True,
                sf_layout="rowmajor",
                k_start=0,
                sf_a_granularity_k=128,
                sf_b_granularity_k=128,
            )

    (call,) = _gemm_calls(main)
    ann = _annotations(call)
    assert int(ann["use_2cta"]) == 1
    assert isinstance(ann["sf_layout"], tirx.StringImm)
    assert ann["sf_layout"].value == "rowmajor"


def test_gemm_blockscaled_checks_2cta_n_extent():
    with pytest.raises(AssertionError, match="2CTA"):

        @T.prim_func
        def main(A: T.Tensor((128, 128), T.float8_e4m3fn), B: T.Tensor((128, 128), T.float8_e4m3fn)):
            with T.Kernel(1, threads=128):
                a = T.alloc_shared((128, 128), T.float8_e4m3fn)
                b = T.alloc_shared((128, 128), T.float8_e4m3fn)
                # 2CTA supplies half of N per CTA, so a full-N B tile is a shape error.
                c = T.alloc_tmem([128, 128], T.float32)
                sfa = T.alloc_tmem([128, 4], T.uint32)
                sfb = T.alloc_tmem([128, 4], T.uint32)
                done = T.alloc_barrier([1])
                T.gemm_blockscaled(
                    a,
                    b,
                    c,
                    sfa,
                    sfb,
                    transpose_B=True,
                    mbar=done[0],
                    use_2cta=True,
                    k_start=0,
                    sf_a_granularity_k=128,
                    sf_b_granularity_k=128,
                )


def test_tcgen05_gemm_blockscaled_pins_tcgen05_path():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float8_e4m3fn), B: T.Tensor((128, 128), T.float8_e4m3fn)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c = T.alloc_tmem([128, 128], T.float32)
            sfa = T.alloc_tmem([128, 4], T.uint32)
            sfb = T.alloc_tmem([128, 4], T.uint32)
            done = T.alloc_barrier([1])
            T.tcgen05_gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                mbar=done[0],
                k_start=0,
                sf_a_granularity_k=128,
                sf_b_granularity_k=128,
            )

    (call,) = _gemm_calls(main)
    # Same op as T.gemm_blockscaled; the ISA is pinned through the annotation,
    # exactly like T.tcgen05_gemm relative to T.gemm.
    assert str(call.op.name) == "tl.tileop.gemm_blockscaled"
    assert len(call.args) == 16
    assert int(_annotations(call)["is_tcgen05"]) == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("gemm_api", ["gemm_blockscaled", "tcgen05_gemm_blockscaled"])
def test_blockscaled_gemm_without_mbar_requires_explicit_async(gemm_api):
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c = T.alloc_tmem([128, 128], T.float32)
            sfa = T.alloc_tmem([128, 4], T.uint32)
            sfb = T.alloc_tmem([128, 4], T.uint32)
            getattr(T, gemm_api)(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                k_start=0,
                sf_a_granularity_k=128,
                sf_b_granularity_k=128,
            )

    (call,) = _gemm_calls(main)
    op = call.op.get_attr("TLOpBuilder")(call.args, call.annotations)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})
    with target:
        layouts = op.infer_layout(target, 128)

        def lower():
            return op.lower(layouts, target, tvm.ir.Range(0, 128), tirx.Var("tx", "int32"), tirx.const(0, "int32"))

        if gemm_api == "gemm_blockscaled":
            with pytest.raises(ValueError, match="requires a valid mbarrier"):
                lower()
        else:
            lowered = str(lower())
            assert "ptx_tcgen05_mma_blockscaled_ss" in lowered
            assert "tcgen05_mma_arrive" not in lowered
            assert "mbarrier_wait_parity" not in lowered


def test_mma_gemm_blockscaled_records_sf_layout():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float4_e2m1fn), B: T.Tensor((128, 128), T.float4_e2m1fn)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float4_e2m1fn)
            b = T.alloc_shared((128, 128), T.float4_e2m1fn)
            sfa = T.alloc_shared((128, 2), T.uint32)
            sfb = T.alloc_shared((128, 2), T.uint32)
            c = T.alloc_fragment((128, 128), T.float32)
            T.mma_gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
                sf_layout="blockscaled_chunk_kmajor",
            )

    (call,) = _gemm_calls(main)
    assert str(call.op.name) == "tl.tileop.gemm_blockscaled"
    assert len(call.args) == 16
    # No mbarrier for the synchronous warp-level path: the slot holds the
    # integer placeholder the C++ side ignores.
    assert isinstance(call.args[10], tirx.IntImm)
    ann = _annotations(call)
    assert "is_tcgen05" not in ann
    assert ann["sf_layout"].value == "blockscaled_chunk_kmajor"


def test_blockscaled_gemm_rejected_by_backend_without_support():
    """The common API can express block-scaled GEMM for any backend.

    A backend without a dedicated block-scaled selector (CPU here) fails
    loudly instead of invoking its dense selector and dropping SFA/SFB.
    """
    from tilelang.cpu import language as T

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float8_e4m3fn),
        B: T.Tensor((128, 128), T.float8_e4m3fn),
        C: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b = T.alloc_shared((128, 128), T.float8_e4m3fn)
            sfa = T.alloc_shared((128, 4), T.uint32)
            sfb = T.alloc_shared((128, 4), T.uint32)
            c = T.alloc_fragment((128, 128), T.float32)
            T.copy(A, a)
            T.copy(B, b)
            T.gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                k_start=0,
                sf_a_granularity_k=32,
                sf_b_granularity_k=32,
            )
            T.copy(c, C)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target("c")
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    with target, pytest.raises(Exception, match="Block-scaled GEMM is not supported by the cpu.Gemm backend"):
        tilelang.transform.LayoutInference()(mod)


# ---------------------------------------------------------------------------
# SM100: 1-CTA MXFP8 block-scaled GEMM through both entry points
# ---------------------------------------------------------------------------


def _make_mxfp8_1cta_kernel(
    gemm_api: str, block_M: int, block_N: int, block_K: int, num_stages: int, sf_granularity_k: int, defer_arrival: bool = False
):
    """1-CTA MXFP8 block-scaled GEMM (A [M, K], B [N, K], group-major packed E8M0 SF).

    Mirrors examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py but
    parametrized on the GEMM entry point. The 1-CTA path is the one that has
    no `use_2cta` short-circuit in instruction selection, so it exercises the
    scope/target driven block-scaled dispatch.
    """
    gemm_fn = getattr(T, gemm_api)

    @tilelang.jit
    def kernel(A, B, SFA, SFB):
        M, N, K = T.const("M, N, K")
        k_iters = T.ceildiv(K, block_K)
        # One uint32 SF word packs 4 K-blocks, so SF is reloaded every 4 blocks.
        sf_load_period = sf_granularity_k * 4 // block_K
        sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)

        A: T.Tensor[[M, K], T.float8_e4m3fn]
        B: T.Tensor[[N, K], T.float8_e4m3fn]
        SFA: T.Tensor[[sf_k_groups * M], T.uint32]
        SFB: T.Tensor[[sf_k_groups * N], T.uint32]
        C = T.empty((M, N), T.float32)

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((num_stages, block_N, block_K), T.float8_e4m3fn)
            SFA_shared = T.alloc_shared((num_stages, block_M), T.uint32)
            SFB_shared = T.alloc_shared((num_stages, block_N), T.uint32)

            C_tmem = T.alloc_tmem([block_M, block_N], T.float32)
            SFA_tmem = T.alloc_tmem([block_M, block_M // 128 * 4], T.uint32)
            SFB_tmem = T.alloc_tmem([block_M, block_N // 128 * 4], T.uint32)

            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            loaded = T.alloc_barrier([32] * num_stages)
            with_sf_full = T.alloc_barrier([32] * num_stages)
            consumed = T.alloc_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()

            if tx < 32:
                for k in T.serial(k_iters):
                    stage = k % num_stages
                    T.mbarrier_wait_parity(consumed[stage], ((k // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    T.tma_copy(
                        B[by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K],
                        B_shared[stage, :, :],
                        barrier=loaded[stage],
                    )
                    if k % sf_load_period == 0:
                        sf_group_idx = k // sf_load_period
                        T.tma_copy(
                            SFA[sf_group_idx * M + bx * block_M : sf_group_idx * M + (bx + 1) * block_M],
                            SFA_shared[stage, :],
                            barrier=loaded[stage],
                        )
                        T.tma_copy(
                            SFB[sf_group_idx * N + by * block_N : sf_group_idx * N + (by + 1) * block_N],
                            SFB_shared[stage, :],
                            barrier=loaded[stage],
                        )
                    T.mbarrier_arrive(loaded[stage])

            elif tx < 64:
                for k in T.serial(k_iters):
                    stage = k % num_stages
                    phase = (k // num_stages) & 1
                    T.mbarrier_wait_parity(loaded[stage], phase)
                    T.mbarrier_wait_parity(with_sf_full[stage], phase)
                    if k % sf_load_period == 0:
                        T.tcgen05_cp_warpx4(SFA_shared[stage, :], SFA_tmem)
                        T.tcgen05_cp_warpx4(SFB_shared[stage, :], SFB_tmem)
                    gemm_fn(
                        A_shared[stage, :, :],
                        B_shared[stage, :, :],
                        C_tmem,
                        SFA_tmem,
                        SFB_tmem,
                        transpose_B=True,
                        mbar=None if defer_arrival else consumed[stage],
                        clear_accum=k == 0,
                        k_start=k * block_K,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    if defer_arrival:
                        T.tcgen05_mma_arrive(consumed[stage])
                T.tcgen05_mma_arrive(tmem_full)

            elif tx < 96:
                for k in T.serial(k_iters):
                    stage = k % num_stages
                    T.mbarrier_wait_parity(loaded[stage], (k // num_stages) & 1)
                    if k % sf_load_period == 0:
                        T.tcgen05_sf_warp_transpose(SFA_shared[stage, :])
                        T.tcgen05_sf_warp_transpose(SFB_shared[stage, :])
                        T.fence_proxy_async()
                    T.mbarrier_arrive(with_sf_full[stage])

            T.mbarrier_wait_parity(tmem_full, 0)
            T.sync_threads()
            T.copy(C_tmem, C_local)
            T.copy(C_local, C[bx * block_M, by * block_N])

        return C

    return kernel


def _pack_sf_words(sf_u8):
    """[MN, 4 * groups] uint8 E8M0 exponents -> group-major packed uint32 [groups * MN]."""
    import torch

    words = sf_u8.to(torch.int64)
    packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24)).to(torch.uint32)
    return packed.T.contiguous().reshape(-1)


def _mxfp8_reference(a, b, sfa_u8, sfb_u8, sf_granularity_k):
    """fp32 reference of (A * 2^(SFA-127)) @ (B * 2^(SFB-127))^T with per-K-block scales."""
    import torch

    M, K = a.shape
    N = b.shape[0]
    kb = K // sf_granularity_k
    sa = torch.pow(2.0, sfa_u8[:, :kb].float() - 127.0)
    sb = torch.pow(2.0, sfb_u8[:, :kb].float() - 127.0)
    a32 = (a.float().view(M, kb, sf_granularity_k) * sa[:, :, None]).view(M, K)
    b32 = (b.float().view(N, kb, sf_granularity_k) * sb[:, :, None]).view(N, K)
    return a32 @ b32.T


def _make_mxfp8_inputs(M, N, K, sf_granularity_k):
    import torch

    a = (torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.5).to(torch.float8_e4m3fn)
    kb_padded = ((K + sf_granularity_k - 1) // sf_granularity_k + 3) // 4 * 4
    sfa_u8 = torch.randint(127 - 3, 127 + 3, (M, kb_padded), device="cuda", dtype=torch.uint8)
    sfb_u8 = torch.randint(127 - 3, 127 + 3, (N, kb_padded), device="cuda", dtype=torch.uint8)
    return a, b, sfa_u8, sfb_u8


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
@pytest.mark.parametrize(
    "gemm_api, defer_arrival",
    [("gemm_blockscaled", False), ("tcgen05_gemm_blockscaled", False), ("tcgen05_gemm_blockscaled", True)],
)
def test_sm100_1cta_blockscaled_gemm_lowers_to_tcgen05(gemm_api, defer_arrival):
    import torch

    torch.manual_seed(0)
    M, N, K = 256, 256, 512
    sf_granularity_k = 128
    kernel = _make_mxfp8_1cta_kernel(gemm_api, 128, 128, 128, 2, sf_granularity_k, defer_arrival)

    a, b, sfa_u8, sfb_u8 = _make_mxfp8_inputs(M, N, K, sf_granularity_k)
    sfa, sfb = _pack_sf_words(sfa_u8), _pack_sf_words(sfb_u8)

    source = kernel.get_kernel_source(a, b, sfa, sfb)
    assert "tcgen05mma_blockscaled_ss" in source
    assert "sm120_mma_sync_blockscaled" not in source

    c = kernel(a, b, sfa, sfb)
    ref = _mxfp8_reference(a, b, sfa_u8, sfb_u8, sf_granularity_k)
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_sm100_rejects_fragment_accumulator_blockscaled_gemm():
    """A block-scaled GEMM must never fall back to a dense instruction: with
    C in a fragment there is no SM100 block-scaled instruction, so compilation
    fails instead of silently dropping the scale factors."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float4_e2m1fn),
        B: T.Tensor((128, 128), T.float4_e2m1fn),
        C: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 128), T.float4_e2m1fn)
            b = T.alloc_shared((128, 128), T.float4_e2m1fn)
            sfa = T.alloc_shared((128, 2), T.uint32)
            sfb = T.alloc_shared((128, 2), T.uint32)
            c = T.alloc_fragment((128, 128), T.float32)
            T.copy(A, a)
            T.copy(B, b)
            T.gemm_blockscaled(
                a,
                b,
                c,
                sfa,
                sfb,
                transpose_B=True,
                k_start=0,
                sf_a_granularity_k=16,
                sf_b_granularity_k=16,
            )
            T.copy(c, C)

    with pytest.raises(Exception, match="Block-scaled GEMM requires"):
        tilelang.compile(main, target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
