import tilelang
import tilelang.language as T  # noqa: N812
import torch
import triton
import triton.language as tl


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42):
    blk_M = 128
    num_threads = 128

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), "uint32")):
        with T.Kernel(M // blk_M, threads=num_threads) as bx:
            T.rng_init(seed)
            for i in T.Parallel(blk_M):
                A[bx * blk_M + i] = T.rng_rand()

    return rand_kernel


@triton.jit
def triton_rand_1d(X, M, seed):
    pid = tl.program_id(0)
    offset = pid * M + tl.arange(0, M)
    rand = tl.randint(seed, offset)
    tl.store(X + offset, rand, mask=offset < M)


if __name__ == "__main__":
    M = 1024
    kernel = tilelang_rand_1d()
    x = torch.empty(M, dtype=torch.uint32, device="cuda")
    kernel(x)
