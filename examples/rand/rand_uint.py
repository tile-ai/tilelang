import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl


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


@triton.jit
def triton_rand_1d(X, M, elements_per_thread, seed):
    pid = tl.program_id(0)
    offset = pid * elements_per_thread + tl.arange(0, elements_per_thread)

    r0, r1, r2, r3 = tl.randint4x(seed, offset)

    base_idx = offset * 4
    tl.store(X + base_idx, r0, mask=base_idx < M)
    tl.store(X + base_idx + 1, r1, mask=(base_idx + 1) < M)
    tl.store(X + base_idx + 2, r2, mask=(base_idx + 2) < M)
    tl.store(X + base_idx + 3, r3, mask=(base_idx + 3) < M)


def test_rand_1d(M, seed):
    kernel = tilelang_rand_1d(M, seed)
    tilelang_result = torch.empty(M, dtype=torch.uint32, device="cuda")
    kernel(tilelang_result)

    triton_result = torch.empty(M, dtype=torch.uint32, device="cuda")
    grid = (triton.cdiv(M, 128),)
    triton_rand_1d[grid](triton_result, tl.constexpr(M), tl.constexpr(128 // 4), seed)

    torch.testing.assert_close(tilelang_result, triton_result)


if __name__ == "__main__":
    test_rand_1d(1024, 42)
    test_rand_1d(512, 123)
    test_rand_1d(128, 0)
