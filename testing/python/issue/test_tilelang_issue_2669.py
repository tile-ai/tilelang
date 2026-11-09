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


@tilelang.testing.requires_cuda
def test_nonzero_min_crosses_broadcast_and_condition_boundary():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32), B: T.Tensor((4,), T.int32)):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(2, 6):
                if i // 4 == 0:
                    B[i - 2] = A[i // 4] + 1
                else:
                    B[i - 2] = A[i // 4] - 1

    kernel = tilelang.compile(main, target="cuda")
    a = torch.tensor([10, 20], device="cuda", dtype=torch.int32)
    b = torch.full((4,), -1, device="cuda", dtype=torch.int32)
    kernel(a, b)
    torch.testing.assert_close(b, torch.tensor([11, 11, 19, 19], device="cuda", dtype=torch.int32), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_nested_vectorized_inner_min_runtime():
    @T.prim_func
    def main(A: T.Tensor((16,), T.int32), B: T.Tensor((16,), T.int32)):
        with T.Kernel(1, threads=1):
            for j in T.vectorized(2):
                for i in T.serial(2, 6):
                    B[j * 8 + i] = A[j * 8 + i] + 1

    kernel = tilelang.compile(main, target="cuda")
    a = torch.arange(16, device="cuda", dtype=torch.int32)
    b = torch.full_like(a, -1)
    kernel(a, b)
    expected = torch.full_like(a, -1)
    expected.view(2, 8)[:, 2:6] = a.view(2, 8)[:, 2:6] + 1
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def _check_mixed_cast_nonzero_min(target, device):
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float16),
        B: T.Tensor((8,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            a_local = T.alloc_local((8,), T.float32)
            for value in T.vectorized(2, 6):
                if value < 5:
                    a_local[value] = A[value]
            for value in T.vectorized(2, 6):
                if value < 5:
                    B[value] = a_local[value] + 1

    kernel = tilelang.compile(main, target=target, execution_backend="cython")
    # Check that both cast directions actually went through staging.
    source = kernel.get_kernel_source()
    assert "A_local_cast" in source
    assert "B_local_cast" in source

    a = torch.arange(8, device=device, dtype=torch.float16)
    b = torch.full((8,), -1, device=device, dtype=torch.float16)
    kernel(a, b)
    expected = a + 1
    expected[:2] = -1
    expected[5:] = -1
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def test_vectorized_nonzero_min_mixed_cast_cpu():
    _check_mixed_cast_nonzero_min("c", "cpu")


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min_mixed_cast_cuda():
    _check_mixed_cast_nonzero_min("cuda", "cuda")


if __name__ == "__main__":
    tilelang.testing.main()
