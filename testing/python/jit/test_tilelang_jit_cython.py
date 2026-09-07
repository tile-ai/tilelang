from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import tilelang
import torch
import pytest


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_cython_pdl():
    """Test pdl."""

    N = 64

    @tilelang.jit(execution_backend="cython")
    def multi_kernels_with_pdl(N, block_size=256, dtype=T.float32):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        B[idx] = A[idx] + 1.0
                T.pdl_trigger()

            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
                T.pdl_sync()
                for i in T.Parallel(block_size):
                    idx = bx2 * block_size + i
                    if idx < N:
                        C[idx] = B[idx] * 2.0

        return main

    # Compile the kernel
    kernel = multi_kernels_with_pdl(N)

    # Create test tensors
    a = torch.randn(N, dtype=torch.float32).cuda()
    b = torch.randn(N, dtype=torch.float32).cuda()
    c = torch.randn(N, dtype=torch.float32).cuda()

    ref_b = a + 1.0
    ref_c = ref_b * 2.0

    kernel(a, b, c)

    # Verify correctness

    tilelang.testing.torch_assert_close(b, ref_b, atol=1e-5, rtol=1e-5)
    tilelang.testing.torch_assert_close(c, ref_c, atol=1e-5, rtol=1e-5)

    print("pdl test passed!")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.uint8, T.uint16, T.uint32, T.uint64])
def test_cython_unsigned_scalar_param(dtype):
    """An unsigned scalar parameter must marshal like its signed counterpart."""

    @T.prim_func
    def main(A: T.Tensor((128,), T.int32), s: dtype, B: T.Tensor((128,), T.int32)):
        with T.Kernel(1, threads=128):
            i = T.get_thread_binding()
            B[i] = A[i] + s

    kernel = tilelang.compile(main, execution_backend="cython")
    a = torch.arange(128, dtype=torch.int32, device="cuda")
    b = torch.empty(128, dtype=torch.int32, device="cuda")
    kernel(a, 5, b)
    tilelang.testing.torch_assert_close(b, a + 5, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
