import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_predicated_pipeline_copy_uses_cp_async_and_zero_fills():
    length = 514
    tile_size = 512

    @T.prim_func
    def main(
        A: T.Tensor((length,), T.float16),
        B: T.Tensor((2 * tile_size,), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((tile_size,), T.float16)
            for tile in T.Pipelined(2, num_stages=1):
                for i in T.Parallel(tile_size):
                    if tile * tile_size + i < length:
                        A_shared[i] = A[tile * tile_size + i]
                    else:
                        A_shared[i] = T.float16(0)
                for i in T.Parallel(tile_size):
                    B[tile * tile_size + i] = A_shared[i]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    assert "cp_async_gs" in kernel.get_kernel_source()

    a = torch.arange(length, device="cuda", dtype=torch.float16)
    b = kernel(a)
    expected = torch.zeros(2 * tile_size, device="cuda", dtype=torch.float16)
    expected[:length] = a
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
