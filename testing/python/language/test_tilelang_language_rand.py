import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import triton
import triton.language as tl


@tilelang.jit(out_idx=[0])
def tilelang_rand_1d(M=1024, seed=42, dtype="float32"):

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=1024) as bx:
            rand_buffer = T.alloc_fragment((M,), dtype)
            T.rand(rand_buffer, seed)

            for i in T.Parallel(M):
                A[bx * M + i] = rand_buffer[i]

    return rand_kernel


@triton.jit
def triton_rand_1d(X, M, seed, dtype: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * M + tl.arange(0, M)
    rand = tl.rand(seed, offset)
    tl.store(X + offset, rand, mask=offset < M)


def run_rand_1d(M=1024, seed=42, dtype="float32", device="cuda"):
    tilelang_kernel = tilelang_rand_1d(M=M, seed=seed, dtype=dtype)
    tilelang_result = torch.empty(M, dtype=torch.float32, device=device)
    tilelang_result = tilelang_kernel()

    triton_result = torch.empty(M, dtype=getattr(torch, dtype), device=device)
    grid = (1,)
    BLOCK = tl.constexpr(M)
    triton_rand_1d[grid](triton_result, BLOCK, seed=seed, dtype=getattr(tl, dtype))

    torch.testing.assert_close(tilelang_result, triton_result, atol=1e-3, rtol=1e-3)


@tilelang.testing.requires_cuda
def test_rand_1d():
    run_rand_1d()


if __name__ == "__main__":
    tilelang.testing.main()
