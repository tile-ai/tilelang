import tilelang
import tilelang.language as T
import torch
import pytest
import tilelang.testing


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42):
    num_per_thread = 128
    threads = 1
    blk_M = num_per_thread * threads

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), "uint32")):
        with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx * blk_M + tx * num_per_thread)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    A[idx] = T.rng_rand()

    return rand_kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("M, seed", [(1024, 42), (512, 123), (128, 0)])
def test_rand_1d(M, seed):
    kernel = tilelang_rand_1d(M, seed)
    tilelang_result = torch.empty(M, dtype=torch.uint32, device="cuda")
    kernel(tilelang_result)


if __name__ == "__main__":
    tilelang.testing.main()
