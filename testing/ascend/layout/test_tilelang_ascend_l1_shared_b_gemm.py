"""L1 GEMM accuracy: one shared B [K, N] buffer serves both trans_B flavors.

    C1 = A1 [M, K] @ B_s       transpose_B=False   (NN)
    C2 = A2 [M, N] @ B_s^T     transpose_B=True    (NT)

The (K, N) geometry is swapped between the two GEMMs, so the shared buffer is
exercised in both axis roles. The NN half is the one that matters: its B operand
reaches the MAD through a transposed L1 -> L0B staging.

16-bit inputs only for now; b8/b32 are covered once their walks are finalized.
"""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

# (M, K, N) of the shared B [K, N] buffer.
SHAPES = [
    (16, 64, 64),  # square
    (16, 128, 128),  # square, larger
    (32, 128, 128),  # square, M=32
    (16, 64, 128),  # K < N
    (16, 128, 64),  # K > N
    (32, 128, 64),  # K > N, M=32
    (16, 256, 128),
]
DTYPES = ["bfloat16", "float16"]


def make_shared_b_gemm_kernel(M: int, K: int, N: int, dtype: str):
    @T.prim_func
    def main(
        A1: T.Buffer((M, K), dtype),
        A2: T.Buffer((M, N), dtype),
        B: T.Buffer((K, N), dtype),
        C1: T.Buffer((M, N), "float32"),
        C2: T.Buffer((M, K), "float32"),
    ):
        with T.Kernel(1):
            A1_s = T.alloc_l1((M, K), dtype)
            A2_s = T.alloc_l1((M, N), dtype)
            B_s = T.alloc_l1((K, N), dtype)  # the single shared buffer
            acc1 = T.alloc_l0c((M, N), "float32")
            acc2 = T.alloc_l0c((M, K), "float32")

            T.copy(A1, A1_s)
            T.copy(A2, A2_s)
            T.copy(B, B_s)

            # NN: B_s as [K, N], transposed during L1 -> L0B
            T.gemm(A1_s, B_s, acc1, transpose_B=False, clear_accum=True)
            # NT: B_s as [N, K]
            T.gemm(A2_s, B_s, acc2, transpose_B=True, clear_accum=True)

            T.copy(acc1, C1)
            T.copy(acc2, C2)

    return main


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
def test_l1_shared_b_gemm(shape, dtype):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    M, K, N = shape

    torch_dtype = getattr(torch, dtype)
    A1 = torch.randn(M, K, dtype=torch_dtype, device="npu")
    A2 = torch.randn(M, N, dtype=torch_dtype, device="npu")
    B = torch.randn(K, N, dtype=torch_dtype, device="npu")

    kernel = tilelang.compile(make_shared_b_gemm_kernel(M, K, N, dtype), target="ascend", out_idx=[3, 4])
    C1, C2 = kernel(A1, A2, B)
    torch.npu.synchronize()

    torch.testing.assert_close(C1, A1.float() @ B.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(C2, A2.float() @ B.T.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
