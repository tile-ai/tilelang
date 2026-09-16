"""Language-level tests for the building blocks of two-warp-group (ping-pong) kernels:

* ``Fragment.bind_thread_range``: a user-annotated fragment that lives in a thread-predicated
  region must be bound to that region's thread range, otherwise the loop partition maps the
  wrong threads and folds the statements away.
* explicit ``software_pipeline_async_producers`` / ``_groups`` annotations on a manually
  scheduled ``T.Pipelined`` loop merge several copies into one commit group (one wait per
  iteration instead of one per copy).
* a register-carried operand: a thread-predicated statement at a lower pipeline stage writes a
  fragment (its own warps' rows) that a later-stage shared statement consumes one iteration later.
* a rotated accumulator: a statement at a *higher* stage (an older tile) is emitted first and
  accumulates into a buffer that a lower-stage statement then updates in the same iteration.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Fragment


def _group_fragment_layout(rows: int = 64, cols: int = 64) -> Fragment:
    # 128 threads: thread = (i // 16) * 32 + (i % 16) * 2 + j // 32, index = j % 32 (bijective)
    return Fragment(
        (rows, cols),
        forward_thread_fn=lambda i, j: (i // 16) * 32 + (i % 16) * 2 + j // 32,
        forward_index_fn=lambda i, j: j % 32,
    )


def _build_bind_thread_range(bind: bool):
    rows, cols, threads = 128, 64, 256
    lay = _group_fragment_layout()
    layA = lay.bind_thread_range(0, 128) if bind else lay
    layB = lay.bind_thread_range(128, 128) if bind else lay

    @T.prim_func
    def main(A: T.Tensor((rows, cols), T.float32), B: T.Tensor((rows, cols), T.float32)):
        with T.Kernel(1, threads=threads) as _:
            tx = T.get_thread_binding()
            FA = T.alloc_fragment((rows // 2, cols), T.float32)
            FB = T.alloc_fragment((rows // 2, cols), T.float32)
            T.annotate_layout({FA: layA, FB: layB})
            if tx // 128 == 0:
                T.copy(A[0:64, :], FA)
                for i, j in T.Parallel(rows // 2, cols):
                    FA[i, j] = FA[i, j] * 2.0
                T.copy(FA, B[0:64, :])
            else:
                T.copy(A[64:128, :], FB)
                for i, j in T.Parallel(rows // 2, cols):
                    FB[i, j] = FB[i, j] * 2.0
                T.copy(FB, B[64:128, :])

    return main


@tilelang.testing.requires_cuda
def test_fragment_bind_thread_range_in_predicated_region():
    kernel = tilelang.jit(out_idx=[1])(_build_bind_thread_range)(True)
    a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    b = kernel(a)
    torch.testing.assert_close(b, a * 2.0)


def _build_manual_pipeline(explicit_groups: bool):
    """Copies A and C are last used by different GEMMs, so the implicit grouping puts them in
    different commit groups; the explicit annotation merges all three copies into one."""
    M = N = K = 128
    BK = 32
    n_stages = 3
    ann = {}
    if explicit_groups:
        ann = {
            "software_pipeline_async_producers": [1, 1, 1, 0, 0],
            "software_pipeline_async_producer_groups": [0, 0, 0, -1, -1],
        }

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, K), T.float16),
        O1: T.Tensor((M, N), T.float32),
        O2: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128) as _:
            A_sh = T.alloc_shared((M, BK), T.float16)
            B_sh = T.alloc_shared((BK, N), T.float16)
            C_sh = T.alloc_shared((M, BK), T.float16)
            acc1 = T.alloc_fragment((M, N), T.float32)
            acc2 = T.alloc_fragment((M, N), T.float32)
            T.clear(acc1)
            T.clear(acc2)
            for ko in T.Pipelined(K // BK, order=[2, 3, 4, 0, 1], stage=[0, 0, 0, n_stages - 1, n_stages - 1], annotations=ann):
                T.copy(A[0, ko * BK], A_sh)
                T.copy(B[ko * BK, 0], B_sh)
                T.copy(C[0, ko * BK], C_sh)
                T.gemm(A_sh, B_sh, acc1)
                T.gemm(C_sh, B_sh, acc2)
            T.copy(acc1, O1)
            T.copy(acc2, O2)

    return main


@tilelang.testing.requires_cuda
def test_manual_pipeline_explicit_async_producer_groups():
    k_impl = tilelang.jit(out_idx=[3, 4])(_build_manual_pipeline)(False)
    k_expl = tilelang.jit(out_idx=[3, 4])(_build_manual_pipeline)(True)
    src_impl = k_impl.get_kernel_source()
    src_expl = k_expl.get_kernel_source()
    # the three copies form one commit group -> fewer commits than the implicit two-group form
    assert src_expl.count("cp_async_commit") < src_impl.count("cp_async_commit")
    a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    c = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    ref1 = a.float() @ b.float()
    ref2 = c.float() @ b.float()
    for k in (k_expl, k_impl):
        o1, o2 = k(a, b, c)
        torch.testing.assert_close(o1, ref1, rtol=1e-2, atol=1e-1)
        torch.testing.assert_close(o2, ref2, rtol=1e-2, atol=1e-1)


def _build_register_carry():
    """Two warp groups accumulate C += A_k @ B_k over k tiles; group B's GEMM runs one pipeline
    stage ahead (its rows of ``acc`` are consumed by the shared accumulate one iteration later)."""
    M, N, K, BK = 128, 64, 256, 32
    threads = 256
    SA = 2
    # statements: copy A, copy B, A-group gemm, shared accumulate, B-group gemm. The copies are
    # issued first: with a consumer at prefetch distance 1 (stage SA-1), a computes-first order
    # makes the pipeline wait one commit group short (the newest tile is still in flight).
    order = [0, 1, 2, 3, 4]
    stage = [0, 0, SA, SA, SA - 1]

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=threads) as _:
            tx = T.get_thread_binding()
            A_sh = T.alloc_shared((M, BK), T.float16)
            B_sh = T.alloc_shared((BK, N), T.float16)
            acc = T.alloc_fragment((M, N), T.float32)
            csum = T.alloc_fragment((M, N), T.float32)
            T.clear(acc)
            T.clear(csum)
            for ko in T.Pipelined(K // BK, order=order, stage=stage):
                T.copy(A[0, ko * BK], A_sh)
                T.copy(B[ko * BK, 0], B_sh)
                if tx // 128 == 0:
                    T.gemm(A_sh, B_sh, acc, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(M, N):
                    csum[i, j] = csum[i, j] + acc[i, j]
                if tx // 128 == 1:
                    T.gemm(A_sh, B_sh, acc, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
            T.copy(csum, C)

    return main


@tilelang.testing.requires_cuda
def test_manual_pipeline_register_carry_between_warp_groups():
    kernel = tilelang.jit(out_idx=[2])(_build_register_carry)()
    a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 64, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float()
    torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)


def _build_rotated_accumulator(n_tiles: int, decay: float):
    """acc += A_t @ B (one tile behind) while acc is scaled by `decay` every iteration.

    The GEMM sits one pipeline stage later than the scaling, which would normally be rejected as a
    backwards dependency; it is legal because the GEMM is emitted first, so within one iteration
    the accumulation for tile t-1 lands before the scaling for tile t.
    """
    M, N, BK = 64, 64, 32
    K = BK * n_tiles
    S = 2
    # statements: copy A (stage 0), gemm of the previous tile (stage S+1), scale (stage S)
    order = [0, 1, 2]
    stage = [0, S + 1, S]

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((BK, N), T.float16), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=128) as _:
            A_sh = T.alloc_shared((M, BK), T.float16)
            B_sh = T.alloc_shared((BK, N), T.float16)
            acc = T.alloc_fragment((M, N), T.float32)
            T.clear(acc)
            T.copy(B, B_sh)
            for ko in T.Pipelined(n_tiles, order=order, stage=stage):
                T.copy(A[0, ko * BK], A_sh)
                T.gemm(A_sh, B_sh, acc)
                for i, j in T.Parallel(M, N):
                    acc[i, j] = acc[i, j] * decay
            T.copy(acc, C)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("n_tiles", [2, 4, 7])  # 2 is shorter than the pipeline depth
def test_manual_pipeline_rotated_accumulator(n_tiles):
    decay, M, N, BK = 0.5, 64, 64, 32
    kernel = tilelang.jit(out_idx=[2])(_build_rotated_accumulator)(n_tiles, decay)
    src = kernel.get_kernel_source()
    assert "float acc[" in src  # the accumulator stays single-versioned (one array, not two)
    a = torch.randn(M, BK * n_tiles, device="cuda", dtype=torch.float16)
    b = torch.randn(BK, N, device="cuda", dtype=torch.float16)
    ref = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    for t in range(n_tiles):  # gemm(t) is followed by the scalings of tiles t+1 .. n-1
        ref += (a[:, t * BK : (t + 1) * BK].float() @ b.float()) * (decay ** (n_tiles - 1 - t))
    torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
    tilelang.testing.main()
