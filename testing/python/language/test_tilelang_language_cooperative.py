import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit
def grid_sync(N=1024):
    block = 64

    @T.prim_func
    def kernel(A: T.Tensor((N), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            A_local = T.alloc_fragment((block), dtype=T.float32)
            n_idx = bx * block
            for i in T.Parallel(block):
                A[n_idx + i] = n_idx + i
            T.sync_grid()
            for i in T.Parallel(block):
                A_local[i] = A[N - n_idx - i - 1]
                T.sync_grid()
                A[n_idx + i] = A[n_idx + i] + A_local[i]

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(6, 0)
def test_grid_sync():
    N = 1024
    kernel = grid_sync(N)
    assert "cooperative_groups::this_grid().sync()" in kernel.get_kernel_source()
    tensor = torch.rand((N), dtype=torch.float32, device="cuda")
    kernel(tensor)
    target = torch.full_like(tensor, tensor[0])
    torch.testing.assert_close(tensor, target)


if __name__ == "__main__":
    tilelang.testing.main()
