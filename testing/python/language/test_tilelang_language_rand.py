import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
import pytest
import tilelang.testing

tilelang.disable_cache()


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42):
    blk_M = M
    num_threads = 1

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), "uint32")):
        with T.Kernel(M // blk_M, threads=num_threads) as bx:
            T.rng_init(seed)
            for i in T.Parallel(blk_M):
                A[bx * blk_M + i] = T.rng_rand()
                # match a particular RNG sequence of triton
                T.rng_rand()
                T.rng_rand()
                T.rng_rand()

    return rand_kernel


@triton.jit
def triton_rand_1d(X, M, blk_M, seed):
    pid = tl.program_id(0)
    offset = pid * blk_M + tl.arange(0, blk_M)
    rand = tl.randint(seed, offset)
    tl.store(X + offset, rand, mask=offset < M)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("M, seed", [(1024, 42), (512, 123), (128, 0)])
def test_rand_1d(M, seed):
    kernel = tilelang_rand_1d(M, seed)
    tilelang_result = torch.empty(M, dtype=torch.uint32, device="cuda")
    kernel(tilelang_result)

    triton_result = torch.empty(M, dtype=torch.uint32, device="cuda")
    grid = (M // 128,)
    triton_rand_1d[grid](triton_result, tl.constexpr(M), tl.constexpr(128), seed)

    torch.testing.assert_close(tilelang_result, triton_result)


if __name__ == "__main__":
    tilelang.testing.main()
