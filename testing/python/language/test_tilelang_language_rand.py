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


# copy from triton codebase
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    })
def tilelang_seeded_dropout(M, p, seed, dtype="float32"):
    block_m = 64

    @T.prim_func
    def seeded_dropout_kernel(A: T.Tensor((M,)), B: T.Tensor((M,))):
        with T.Kernel(T.ceildiv(M, block_m), threads=128) as (bx):
            block_start = bx * block_m
            A_shared = T.alloc_shared((block_m,), dtype)
            rand_buffer = T.alloc_fragment((block_m,), dtype)
            output_buffer = T.alloc_fragment((block_m,), dtype)

            T.copy(A[block_start], A_shared)
            T.rand(rand_buffer, seed)

            for i in T.Parallel(block_m):
                output_buffer[i] = T.if_then_else(
                    rand_buffer[i] > p,
                    A_shared[i] / (1.0 - p),
                    0.0,
                )

            T.copy(output_buffer, B[block_start])

    return seeded_dropout_kernel


def run_seeded_dropout(M=1024, p=0.5, seed=42, dtype="float32", device="cuda"):
    x = torch.rand(M, dtype=torch.float32, device=device)
    triton_output = torch.empty(M, dtype=getattr(torch, dtype), device=device)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _seeded_dropout[grid](x, triton_output, n_elements, p, seed, BLOCK_SIZE=1024)

    tilelang_kernel = tilelang_seeded_dropout(M=M, p=p, seed=seed, dtype=dtype)
    tilelang_result = tilelang_kernel(x)

    torch.testing.assert_close(tilelang_result, triton_output, atol=1e-3, rtol=1e-3)


@triton.jit
def triton_rand_2d(X, M, N, seed, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    offsets_m = block_start_m + tl.arange(0, BLOCK_M)[:, None]
    offsets_n = block_start_n + tl.arange(0, BLOCK_N)[None, :]

    offsets = offsets_m * N + offsets_n

    mask = (offsets_m < M) & (offsets_n < N)

    rand = tl.rand(seed, offsets)

    tl.store(X + offsets, rand, mask=mask)


@tilelang.jit(out_idx=[0])
def tilelang_rand_2d(M, N, seed, dtype, block_m, block_n):

    @T.prim_func
    def rand_kernel(A: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(N, block_n), threads=64) as (bx, by):
            rand_buffer = T.alloc_fragment((block_m, block_n), dtype)
            T.rand(rand_buffer, seed)
            T.copy(rand_buffer, A[bx * block_m, by * block_n])

    return rand_kernel


def run_rand_2d(M=128, N=128, seed=42, dtype="float32", device="cuda", block_m=16, block_n=16):
    tilelang_kernel = tilelang_rand_2d(
        M=M, N=N, seed=seed, dtype=dtype, block_m=block_m, block_n=block_n)
    tilelang_result = torch.empty(M, N, dtype=getattr(torch, dtype), device=device)
    tilelang_result = tilelang_kernel()

    triton_result = torch.empty(M, N, dtype=getattr(torch, dtype), device=device)
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    BLOCK_M = tl.constexpr(block_m)
    BLOCK_N = tl.constexpr(block_n)
    triton_rand_2d[[triton.cdiv(M, block_m), triton.cdiv(N, block_n)
                   ]](triton_result,
                      M,
                      N,
                      seed=seed,
                      BLOCK_M=BLOCK_M,
                      BLOCK_N=BLOCK_N,
                      dtype=getattr(tl, dtype))

    torch.testing.assert_close(tilelang_result, triton_result, atol=1e-3, rtol=1e-3)


def run_triton_rand_2d(M=8, N=8, seed=42, dtype="float32", block_m=4, block_n=4, device="cuda"):
    triton_result = torch.empty(M, N, dtype=getattr(torch, dtype), device=device)
    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    BLOCK_M = tl.constexpr(block_m)
    BLOCK_N = tl.constexpr(block_n)
    triton_rand_2d[[triton.cdiv(M, block_m), triton.cdiv(N, block_n)
                   ]](triton_result,
                      M,
                      N,
                      seed=seed,
                      BLOCK_M=BLOCK_M,
                      BLOCK_N=BLOCK_N,
                      dtype=getattr(tl, dtype))
    return triton_result


@tilelang.testing.requires_cuda
def test_rand():
    run_seeded_dropout()
    run_rand_1d()
    run_rand_2d()


if __name__ == "__main__":
    tilelang.testing.main()
