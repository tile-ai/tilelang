import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _nonzero_min_vectorized(
    A: T.Tensor((8,), T.int32),
    B: T.Tensor((8,), T.int32),
):
    with T.Kernel(1, threads=1):
        for value in T.vectorized(2, 6):
            B[value] = A[value] + 1


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min():
    a = torch.arange(8, device="cuda", dtype=torch.int32)
    b = torch.full((8,), -1, device="cuda", dtype=torch.int32)

    _nonzero_min_vectorized(a, b)

    expected = torch.tensor(
        [-1, -1, 3, 4, 5, 6, -1, -1],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
