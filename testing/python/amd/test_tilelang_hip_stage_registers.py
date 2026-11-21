"""A tile of fp16 loads held across a loop in a fragment stays in registers."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch
from tilelang.tileop.base import GemmWarpPolicy


def _make(M, N, D, steps, threads):
    per_row = D // 8

    @T.prim_func
    def main(
        A: T.Tensor((M, D), T.float16),
        B: T.Tensor((steps * N, D), T.float16),
        out: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, D), T.float16)
            B_s = T.alloc_shared((N, D), T.float16)
            B_r = T.alloc_fragment((N, D), T.float16)
            s = T.alloc_fragment((M, N), T.float32)
            T.annotate_layout({B_r: T.Fragment((N, D), forward_fn=lambda i, j: ((i // 2) * per_row + j // 8, (i % 2) * 8 + j % 8))})
            T.copy(A[0, 0], A_s)
            T.clear(s)
            T.copy(B[0:N, :], B_r)
            for k in T.serial(steps):
                T.copy(B_r, B_s)
                T.gemm(A_s, B_s, s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                if k + 1 < steps:
                    T.copy(B[(k + 1) * N : (k + 2) * N, :], B_r)
            T.copy(s, out[0, 0])

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("M", "N", "D", "threads"), [(64, 32, 128, 256), (64, 32, 64, 128)])
def test_staged_tile_uses_no_scratch(M, N, D, threads):
    steps = 3
    kernel = tilelang.compile(_make(M, N, D, steps, threads), out_idx=[-1], target="hip")
    assert kernel.get_kernel_source().count("tl::RegVec<uint4>") > 0
    usage = next(iter(dict(kernel.resource_usage).values()))
    assert usage.scratch_bytes == 0
    A = torch.randn(M, D, dtype=torch.float16, device="cuda")
    B = torch.randn(steps * N, D, dtype=torch.float16, device="cuda")
    out = kernel(A, B)
    ref = sum(A.float() @ B[k * N : (k + 1) * N].float().T for k in range(steps))
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


@tilelang.testing.requires_rocm
def test_gemm_operand_keeps_plain_access():
    """A fragment that GEMM reads with the K loop index stays a plain array: on clang 22 a promoted one keeps the K loop and reads it through selects."""
    M, N, D = 16, 32, 128

    @T.prim_func
    def main(A: T.Tensor((M, D), T.float16), B: T.Tensor((N, D), T.float16), out: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, threads=64):
            A_f = T.alloc_fragment((M, D), T.float16)
            B_s = T.alloc_shared((N, D), T.float16)
            s = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_f)
            T.copy(B, B_s)
            T.gemm(A_f, B_s, s, transpose_B=True, clear_accum=True, policy=GemmWarpPolicy.FullRow)
            T.copy(s, out)

    src = tilelang.compile(main, out_idx=[2], target="hip").get_kernel_source()
    assert "float16x" in src
    assert "RegVec<uint2>*)(A_f" not in src
    assert "RegVec<uint4>*)(A_f" not in src


if __name__ == "__main__":
    tilelang.testing.main()
