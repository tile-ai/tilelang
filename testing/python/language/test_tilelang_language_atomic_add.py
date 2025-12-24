import tilelang.testing
import tilelang.language as T


@tilelang.jit
def atomic_add_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_add


def run_atomic_add(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] += A[k, i, j]

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_add_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_add(B[bx * block_M, by * block_N], A_shared)

    return atomic_add


def run_tile_atomic_add(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    print(kernel.get_kernel_source())
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] += A[k, i, j]

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    print(B)
    print(ref_B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_max_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_max(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_max(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_max


def run_atomic_max(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_max_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = max(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_min_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_min(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_min(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_min


def run_atomic_min(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_min_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = min(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.full((M, N), float("inf"), dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_load_store_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_load_store(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    val = T.atomic_load(A[idx_i, idx_j])
                    T.atomic_store(B[idx_i, idx_j], val)

    return atomic_load_store


def run_atomic_load_store(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_load_store_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    kernel(A, B)
    torch.testing.assert_close(B, A, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_memory_order_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_with_memory_order(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(B[bx * block_M + i, by * block_N + j], A_shared[i, j], memory_order="relaxed")

    return atomic_with_memory_order


def run_atomic_memory_order(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_memory_order_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] += A[k, i, j]

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_addx2_program(M, N, block_M, block_N):
    @T.prim_func
    def atomic_addx2(A: T.Tensor((M, N), T.float16), B: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 2):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 2
                T.atomic_addx2(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx2


def run_atomic_addx2(M, N, block_M, block_N):
    kernel = atomic_addx2_program(M, N, block_M, block_N)
    import torch

    A = torch.randn(M, N, dtype=torch.float16).cuda()
    B = torch.zeros(M, N, dtype=torch.float16).cuda()
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 1, 2):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


def test_atomic_add():
    run_atomic_add(8, 128, 128, 32, 32)


def test_atomic_max():
    run_atomic_max(4, 64, 64, 16, 16)


def test_atomic_min():
    run_atomic_min(4, 64, 64, 16, 16)


def test_atomic_load_store():
    run_atomic_load_store(64, 64, 16, 16)


def test_atomic_memory_order():
    run_atomic_memory_order(4, 64, 64, 16, 16)


def test_atomic_addx2():
    run_atomic_addx2(32, 64, 8, 16)


@tilelang.jit
def atomic_different_memory_orders_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_different_orders(
        A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype), D: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    val = A[idx_i, idx_j]
                    T.atomic_add(B[idx_i, idx_j], val, memory_order="release")
                    T.atomic_max(C[idx_i, idx_j], val, memory_order="relaxed")
                    T.atomic_min(D[idx_i, idx_j], val, memory_order="relaxed")

    return atomic_different_orders


def run_atomic_different_memory_orders(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_different_memory_orders_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    C = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    D = torch.full((M, N), float("inf"), dtype=getattr(torch, dtype)).cuda()

    kernel(A, B, C, D)

    torch.testing.assert_close(B, A, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(C, torch.maximum(torch.zeros_like(A), A))
    torch.testing.assert_close(D, torch.minimum(torch.full_like(A, float("inf")), A))


@tilelang.jit
def atomic_addx4_program(M, N, block_M, block_N):
    @T.prim_func
    def atomic_addx4(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 4):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 4
                T.atomic_addx4(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx4


def run_atomic_addx4(M, N, block_M, block_N):
    kernel = atomic_addx4_program(M, N, block_M, block_N)
    import torch

    A = torch.randn(M, N, dtype=torch.float32).cuda()
    B = torch.zeros(M, N, dtype=torch.float32).cuda()
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 3, 4):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
            ref_B[i, j + 2] += A[i, j + 2]
            ref_B[i, j + 3] += A[i, j + 3]

    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_return_prev_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_with_return_prev(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), old_vals: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    old_vals[idx_i, idx_j] = T.atomic_add(B[idx_i, idx_j], A[idx_i, idx_j], return_prev=True)

    return atomic_with_return_prev


def run_atomic_return_prev(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_return_prev_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.ones(M, N, dtype=getattr(torch, dtype)).cuda() * 5.0
    B = torch.ones(M, N, dtype=getattr(torch, dtype)).cuda() * 2.0
    old_vals = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()

    initial_B = B.clone()
    kernel(A, B, old_vals)

    torch.testing.assert_close(old_vals, initial_B, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(B, initial_B + A, atol=1e-3, rtol=1e-3)


def test_atomic_different_memory_orders():
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float32)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float16)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.bfloat16)


def test_atomic_addx4():
    run_atomic_addx4(16, 64, 4, 4)


def test_atomic_return_prev():
    run_atomic_return_prev(32, 32, 8, 8)


def test_tile_atomic_add():
    run_tile_atomic_add(8, 128, 128, 32, 32)


if __name__ == "__main__":
    tilelang.testing.main()
