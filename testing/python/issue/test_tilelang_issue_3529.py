"""Regression test for issue #3529.

Under automatic warp specialization, a consumer that reads a TMA-filled stage
through the generic proxy (e.g. ``T.Parallel`` loads, ``ldmatrix``) must issue
``fence.proxy.async`` before its back-pressure ``mbarrier.arrive``; otherwise
the producer's next TMA load (async proxy) can overwrite the stage while those
reads are still in flight. Consumers that only read the stage through wgmma
(async proxy) need no fence.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing

N, C, BK = 16384, 32, 64
ROW_SCALE = 11.313708498984761


@T.prim_func
def exp_sum(
    X: T.Tensor((N, C), T.float32),
    R: T.Tensor((N, C), T.bfloat16),
    M: T.Tensor((C,), T.float32),
    P: T.Tensor((N // 256, C), T.float32),
):
    with T.Kernel(1, N // 256, threads=128) as (by, bz):
        X_s = T.alloc_shared((BK, C), T.float32)
        R_s = T.alloc_shared((BK, C), T.bfloat16)
        M_s = T.alloc_shared((C,), T.float32)
        e = T.alloc_fragment((C, BK), T.float32)
        acc = T.alloc_fragment((C,), T.float32)
        part = T.alloc_fragment((C,), T.float32)
        T.copy(M[0:C], M_s)
        T.clear(acc)
        for kk in T.Pipelined(4, num_stages=3):
            T.copy(X[kk * BK + 256 * bz : kk * BK + 256 * bz + BK, 0:C], X_s)
            T.copy(R[kk * BK + 256 * bz : kk * BK + 256 * bz + BK, 0:C], R_s)
            # Elementwise (generic-proxy) reads of the TMA-filled stage.
            for i, j in T.Parallel(C, BK):
                e[i, j] = T.exp(X_s[j, i] * T.Cast(T.float32, R_s[j, i]) / ROW_SCALE - M_s[i])
            T.reduce_sum(e, part, dim=1)
            for i in T.Parallel(C):
                acc[i] = acc[i] + part[i]
        T.copy(acc, P[bz, 0:C])


def matmul(M, N, K, block_M, block_N, block_K, num_stages, threads=128, clear_from_stage=False):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                if clear_from_stage:
                    # A scalar generic-proxy read of the stage nested in an
                    # otherwise wgmma-only GEMM call.
                    T.gemm(A_shared, B_shared, C_local, clear_accum=A_shared[0, 0] > 1e4)
                else:
                    T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _compile(func, pass_configs=None, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(func, target="cuda", pass_configs=pass_configs or {}, **kwargs)
    finally:
        tilelang.enable_cache()


def _consumer_releases_fenced(src):
    """For every back-pressure ``mbarrier[...].arrive();`` in the consumer
    branch of `src`, whether a ``tl::fence_proxy_async();`` precedes it with
    only other arrives in between (one fence covers consecutive releases)."""
    lines = [line.strip() for line in src.splitlines() if line.strip()]
    split = next(i for i, line in enumerate(lines) if line.startswith("if (((int)threadIdx.x) < "))
    depth, consumer_start = 0, None
    for i in range(split, len(lines)):
        if depth == 1 and lines[i].startswith("} else {"):
            consumer_start = i
            break
        depth += lines[i].count("{") - lines[i].count("}")
    assert consumer_start is not None, src
    arrive = re.compile(r"mbarrier\[.*\]\.arrive\(\);")
    releases = [i for i in range(consumer_start, len(lines)) if arrive.fullmatch(lines[i])]
    assert releases, src
    fenced = []
    for i in releases:
        j = i - 1
        while arrive.fullmatch(lines[j]):
            j -= 1
        fenced.append(lines[j] == "tl::fence_proxy_async();")
    return fenced


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_generic_stage_reads_fence_before_release():
    kernel = _compile(exp_sum, {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    src = kernel.get_kernel_source()
    assert "if (((int)threadIdx.x) < 128)" in src, "expected warp specialization"
    assert all(_consumer_releases_fenced(src)), src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_mma_sync_stage_reads_fence_before_release():
    # Without wgmma the GEMM reads the stage with ldmatrix (generic proxy).
    kernel = _compile(matmul(512, 512, 512, 128, 128, 64, 3), {tilelang.PassConfigKey.TL_DISABLE_WGMMA: True}, out_idx=[2])
    src = kernel.get_kernel_source()
    assert "wgmma" not in src
    assert all(_consumer_releases_fenced(src)), src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_wgmma_only_consumer_has_no_release_fence():
    kernel = _compile(matmul(512, 512, 512, 128, 128, 64, 3), out_idx=[2])
    src = kernel.get_kernel_source()
    assert "wgmma" in src
    assert not any(_consumer_releases_fenced(src)), src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_wgmma_with_nested_stage_read_fences_release():
    # The wgmma operands are async-proxy reads, but `clear_accum` reads the
    # stage through the generic proxy.
    kernel = _compile(matmul(512, 512, 512, 128, 128, 64, 3, clear_from_stage=True), out_idx=[2])
    src = kernel.get_kernel_source()
    assert "wgmma" in src
    assert all(_consumer_releases_fenced(src)), src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ws_generic_stage_reads_fast_math_correctness():
    import torch

    kernel = _compile(exp_sum, {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    g = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(N, C, device="cuda", generator=g) * 20
    R = (torch.rand(N, C, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    arg = X.double() * R.double() / ROW_SCALE
    M = arg.max(0).values.float().contiguous()
    ref = torch.exp(arg - M.double()[None, :]).reshape(N // 256, 256, C).sum(1)
    for _ in range(5):
        P = torch.full((N // 256, C), float("nan"), device="cuda")
        kernel(X, R, M, P)
        torch.testing.assert_close(P.double(), ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
