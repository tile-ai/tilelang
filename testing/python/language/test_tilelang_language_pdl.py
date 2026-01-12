import tilelang.testing
import tilelang.language as T


def kernels_with_pdl_trigger(N, block_size=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    B[idx] = A[idx] + 1.0
            T.pdl_trigger()

    return main


def kernels_with_pdl_sync(N, block_size=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            T.pdl_sync()
            for i in T.Parallel(block_size):
                idx = bx2 * block_size + i
                if idx < N:
                    B[idx] = A[idx] * 2.0

    return main


@tilelang.testing.requires_cuda
def test_pdl_trigger():
    N = 64
    program = kernels_with_pdl_trigger(N)

    pdl_kernel = tilelang.compile(program, target="cuda -arch=sm_90")
    code = pdl_kernel.get_kernel_source()
    assert "cudaTriggerProgrammaticLaunchCompletion" in code


@tilelang.testing.requires_cuda
def test_pdl_sync():
    N = 64
    program = kernels_with_pdl_sync(N)
    pdl_kernel = tilelang.compile(program, target="cuda -arch=sm_90")
    code = pdl_kernel.get_kernel_source()
    assert "cudaGridDependencySynchronize" in code
    assert "__restrict__" not in code


if __name__ == "__main__":
    tilelang.testing.main()
