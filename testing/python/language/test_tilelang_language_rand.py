import tilelang
import tilelang.language as T
import torch
import pytest
import tilelang.testing


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42, generator="curandStatePhilox4_32_10_t"):
    num_per_thread = 128
    threads = 1
    blk_M = num_per_thread * threads

    @T.prim_func
    def rand_kernel(
        A: T.Tensor((M,), "uint32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M,), "float64"),
        D: T.Tensor((M,), "float32"),
        E: T.Tensor((M,), "float64"),
    ):
        with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx * blk_M + tx * num_per_thread, generator=generator)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    A[idx] = T.rng_rand()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    B[idx] = T.rng_rand_float()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    C[idx] = T.rng_rand_float(bit=64)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    D[idx] = T.rng_rand_float(dist="normal")
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    E[idx] = T.rng_rand_float(bit=64, dist="normal")

    return rand_kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, seed, generator", [(1024, 42, "curandStateMRG32k3a_t"), (512, 123, "curandStatePhilox4_32_10_t"), (128, 0, "curandStateXORWOW_t")]
)
def test_rand_1d(M, seed, generator):
    kernel = tilelang_rand_1d(M, seed, generator)
    A = torch.empty(M, dtype=torch.uint32, device="cuda")
    B = torch.empty(M, dtype=torch.float32, device="cuda")
    C = torch.empty(M, dtype=torch.float64, device="cuda")
    D = torch.empty(M, dtype=torch.float32, device="cuda")
    E = torch.empty(M, dtype=torch.float64, device="cuda")
    kernel(A, B, C, D, E)


if __name__ == "__main__":
    tilelang.testing.main()
