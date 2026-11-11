"""T.reducer_scale: a running sum rescaled per step, reduced once at the end."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch
from tilelang.tileop.base import GemmWarpPolicy


@tilelang.jit(out_idx=[-1], target="hip")
def _scaled_sum(M, N, steps, threads, one_thread_per_row=False):
    layout = T.Fragment((M,), forward_fn=lambda i: (i, 0)) if one_thread_per_row else None

    @T.prim_func
    def main(
        A: T.Tensor((steps, M, 32), T.float16),
        B: T.Tensor((N, 32), T.float16),
        out: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, 32), T.float16)
            B_s = T.alloc_shared((N, 32), T.float16)
            s = T.alloc_fragment((M, N), T.float32)
            m = T.alloc_fragment((M,), T.float32)
            f = T.alloc_fragment((M,), T.float32)
            r = T.alloc_fragment((M,), T.float32)
            acc = T.alloc_reducer((M,), T.float32, op="sum")
            T.reducer_init(acc)
            T.copy(B[0, 0], B_s)
            for k in T.serial(steps):
                T.copy(A[k, 0, 0], A_s)
                T.clear(s)
                T.gemm(A_s, B_s, s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                # The row maximum is replicated over the row's threads, as a factor must be.
                T.reduce_max(s, m, dim=1)
                for i in T.Parallel(M):
                    f[i] = T.exp(-T.abs(m[i]))
                for i in T.Parallel(M, loop_layout=layout):
                    T.reducer_scale(acc[i], f[i])
                for i, j in T.Parallel(M, N):
                    T.reducer_update(acc[i], s[i, j])
            T.finalize_reducer(acc, r)
            for i, j in T.Parallel(M, N):
                out[i, j] = r[i]

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("threads", "M", "N"), [(64, 64, 64), (256, 128, 64), (256, 64, 64)])
def test_scaled_sum_matches_reference(threads, M, N):
    steps = 3
    kernel = _scaled_sum(M, N, steps, threads)
    A = torch.randn(steps, M, 32, dtype=torch.float16, device="cuda")
    B = torch.randn(N, 32, dtype=torch.float16, device="cuda")
    out = kernel(A, B)
    ref = torch.zeros(M, device="cuda")
    for k in range(steps):
        scores = A[k].float() @ B.float().T
        ref = ref * torch.exp(-scores.max(dim=1).values.abs()) + scores.sum(dim=1)
    assert (out - ref[:, None]).abs().max() <= 1e-2 * ref.abs().max()


@tilelang.jit(out_idx=[-1], target="hip")
def _scale_max():
    @T.prim_func
    def main(A: T.Tensor((64, 64), T.float32), out: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=64):
            s = T.alloc_fragment((64, 64), T.float32)
            r = T.alloc_fragment((64,), T.float32)
            acc = T.alloc_reducer((64,), T.float32, op="max")
            T.reducer_init(acc)
            for i in T.Parallel(64):
                T.reducer_scale(acc[i], 2.0)
            for i, j in T.Parallel(64, 64):
                s[i, j] = A[i, j]
            for i, j in T.Parallel(64, 64):
                T.reducer_update(acc[i], s[i, j])
            T.finalize_reducer(acc, r)
            for i in T.Parallel(64):
                out[i] = r[i]

    return main


@tilelang.testing.requires_rocm
def test_only_sum_can_be_scaled():
    with pytest.raises(Exception, match="only a sum reducer can be scaled"):
        _scale_max().get_kernel_source()


@tilelang.testing.requires_rocm
def test_scale_must_reach_every_partial():
    with pytest.raises(Exception, match="does not run on every thread"):
        _scaled_sum(64, 64, 2, 64, one_thread_per_row=True).get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
