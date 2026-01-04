import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def test_tensor_annot_mul():
    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n * 4,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n * 4):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device="cuda")
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32, device="cuda")
    assert torch.equal(A, expected)


def test_tensor_annot_add():
    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n + 1,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n + 1):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device="cuda")
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32, device="cuda")
    assert torch.equal(A, expected)


def test_tensor_annot_mul_add():
    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n * 3 + 1,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n * 3 + 1):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device="cuda")
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32, device="cuda")
    assert torch.equal(A, expected)


if __name__ == "__main__":
    tilelang.testing.main()
