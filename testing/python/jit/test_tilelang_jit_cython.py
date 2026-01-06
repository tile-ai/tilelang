from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import tilelang
import torch


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


if __name__ == "__main__":
    tilelang.testing.main()
