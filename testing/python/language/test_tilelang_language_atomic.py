import tilelang.testing
import tilelang.layout
import tilelang.language as T
import torch


# ======================= Thread-level atomic add =======================


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
def atomic_addx2_program(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def atomic_addx2(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 2):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 2
                T.atomic_addx2(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx2


def run_atomic_addx2(M, N, block_M, block_N, dtype=T.float16):
    kernel = atomic_addx2_program(M, N, block_M, block_N, dtype=dtype)
    print(kernel.get_kernel_source())
    import torch

    A = torch.randn(M, N, dtype=torch.float32).cuda().to(getattr(torch, dtype))
    B = torch.zeros(M, N, dtype=torch.float32).cuda().to(getattr(torch, dtype))
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 1, 2):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


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


@tilelang.jit
def tma_atomic_add_program(out, explicit_swizzle=False):
    out: T.Tensor[(16, 16), T.float32]

    with T.Kernel(
        1,
    ):
        out_shared = T.alloc_shared((16, 16), dtype=T.float32)
        if explicit_swizzle:
            T.annotate_layout({out_shared: tilelang.layout.make_swizzled_layout(out_shared)})
        T.fill(out_shared, 1)
        for _ in range(16):
            T.atomic_add(out, out_shared, use_tma=True)


@tilelang.testing.requires_cuda
def test_tma_atomic_add():
    out = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
    tma_atomic_add_program(out)
    torch.testing.assert_close(out, torch.ones((16, 16), dtype=torch.float32, device="cuda") * 16)

    kernel = tma_atomic_add_program.compile(out=T.Tensor[(16, 16), T.float32])
    assert "tma_store_add" in kernel.get_kernel_source()
    assert "desc" in kernel.get_kernel_source()  # Ensure using cp.reduce.async.bulk.tensor

    kernel_with_explicit_swizzle = tma_atomic_add_program.compile(out=T.Tensor[(16, 16), T.float32], explicit_swizzle=True)
    # Ensure auto swizzled layout is applied
    assert kernel.get_kernel_source() == kernel_with_explicit_swizzle.get_kernel_source()


def run_atomic_add_auto_vectorized(K, M, N, block_M, block_N, dtype=T.float32):
    tilelang.disable_cache()
    kernel = atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    print(kernel.get_kernel_source())
    assert "AtomicAddx4" in kernel.get_kernel_source()


@tilelang.jit
def atomic_add_complicated_parallel_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                value = A_shared[i, j]
                T.atomic_add(B[bx * block_M + i, by * block_N + j], value)

    return atomic_add


def run_atomic_add_complicated_parallel(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_add_complicated_parallel_program(K, M, N, block_M, block_N, dtype=dtype)
    assert "float4 value" in kernel.get_kernel_source()
    assert "AtomicAddx4" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_atomic_memory_order():
    run_atomic_memory_order(4, 64, 64, 16, 16)


@tilelang.testing.requires_cuda
def test_atomic_addx2_half():
    run_atomic_addx2(32, 64, 8, 16, dtype=T.float16)


def test_atomic_addx2_float():
    run_atomic_addx2(32, 64, 8, 16, dtype=T.float32)


@tilelang.testing.requires_cuda
def test_atomic_different_memory_orders():
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float32)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float16)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.bfloat16)


def test_atomic_addx4():
    run_atomic_addx4(16, 64, 4, 4)


def test_atomic_return_prev():
    run_atomic_return_prev(32, 32, 8, 8)


def test_atomic_add():
    run_atomic_add(8, 128, 128, 32, 32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_atomic_add_auto_vectorized():
    run_atomic_add_auto_vectorized(8, 128, 128, 32, 32)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_atomic_add_complicated_parallel():
    run_atomic_add_complicated_parallel(8, 128, 128, 32, 32)


# ======================= Tile-level atomic add =======================


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
def tile_atomic_add_expr_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            T.atomic_add(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], 1.0)

    return atomic_add


def run_tile_atomic_add_expr(M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_add_expr_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A):
        for i in range(M):
            for j in range(N):
                A[i, j] += 1

    A = torch.zeros(M, N, dtype=torch.float32).cuda()
    ref_A = A.clone()
    ref_program(ref_A)
    kernel(A)
    torch.testing.assert_close(A, ref_A, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_add_scalar_program(dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((1), dtype), B: T.Tensor((1), dtype)):
        with T.Kernel(
            1,
        ) as _:
            A_local = T.alloc_local([1], dtype)
            T.copy(A, A_local)
            T.clear(B)
            T.atomic_add(B, A_local)
            T.atomic_add(B, 1)

    return atomic_add


def run_tile_atomic_add_scalar(dtype=T.float32):
    kernel = tile_atomic_add_scalar_program(dtype=dtype)
    import torch

    def ref_program(A, B):
        B[0] = A[0] + 1

    A = torch.randn(1, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(1, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


def test_tile_atomic_add():
    run_tile_atomic_add(8, 128, 128, 32, 32)


def test_tile_atomic_add_expr():
    run_tile_atomic_add_expr(128, 128, 32, 32)


def test_tile_atomic_add_scalar():
    run_tile_atomic_add_scalar()


# ======================= Thread-level atomic max/min/load store =======================


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


def test_atomic_max():
    run_atomic_max(4, 64, 64, 16, 16)


def test_atomic_min():
    run_atomic_min(4, 64, 64, 16, 16)


@tilelang.testing.requires_cuda
def test_atomic_load_store():
    run_atomic_load_store(64, 64, 16, 16)


# ======================= Tile-level atomic max/min =======================


@tilelang.jit
def tile_atomic_max_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def tile_atomic_max(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_max(B[bx * block_M, by * block_N], A_shared)

    return tile_atomic_max


def run_tile_atomic_max(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_max_program(K, M, N, block_M, block_N, dtype=dtype)

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = max(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.full((M, N), float("-inf"), dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_min_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def tile_atomic_min(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_min(B[bx * block_M, by * block_N], A_shared)

    return tile_atomic_min


def run_tile_atomic_min(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_min_program(K, M, N, block_M, block_N, dtype=dtype)

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
def tile_atomic_max_expr_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_max(A: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            T.atomic_max(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], 0.5)

    return atomic_max


def run_tile_atomic_max_expr(M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_max_expr_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A):
        for i in range(M):
            for j in range(N):
                A[i, j] = max(A[i, j], 0.5)

    A = torch.randn(M, N, dtype=torch.float32).cuda()
    ref_A = A.clone()
    ref_program(ref_A)
    kernel(A)
    torch.testing.assert_close(A, ref_A, atol=1e-3, rtol=1e-3)


def test_tile_atomic_max():
    run_tile_atomic_max(8, 128, 128, 32, 32)


def test_tile_atomic_min():
    run_tile_atomic_min(8, 128, 128, 32, 32)


def test_tile_atomic_max_expr():
    run_tile_atomic_max_expr(128, 128, 32, 32)


if __name__ == "__main__":
    tilelang.testing.main()
