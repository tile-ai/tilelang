import argparse
import logging
import time

import torch

import tilelang
import tilelang.language as T

logging.getLogger("tilelang").setLevel(logging.WARNING)

BLOCK_CONFIGS = [
    ("simdgroup", 16, 16, 16, 128, 0, "row"),
    ("simdgroup", 32, 32, 16, 128, 0, "row"),
    ("simdgroup", 32, 32, 32, 128, 0, "row"),
    ("simdgroup", 64, 64, 32, 128, 0, "row"),
    ("ct_shared", 32, 64, 32, 128, 0, "row"),
    ("ct_shared", 64, 64, 32, 128, 0, "row"),
    # Direct global cooperative tensor path.  The K tile is the full problem K,
    # so C is accumulated in cooperative-tensor registers and written once.
    ("ct_global", 64, 64, 0, 64, 0, "row"),
    ("ct_global", 64, 128, 0, 128, 0, "row"),
    ("ct_global", 64, 128, 0, 128, 4, "mlx"),
    ("ct_global", 64, 128, 0, 256, 4, "mlx"),
]


@tilelang.jit
def matmul_simdgroup(M, N, K, block_M=64, block_N=64, block_K=32, dtype=T.float16, accum_dtype=T.float32):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


@tilelang.jit
def matmul_cooperative_tensor_shared_c(
    M,
    N,
    K,
    block_M=64,
    block_N=64,
    block_K=32,
    dtype=T.float16,
    accum_dtype=T.float32,
):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")
            T.clear(C_shared)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm_kernel


@tilelang.jit
def matmul_cooperative_tensor_global(
    M,
    N,
    K,
    block_M=64,
    block_N=64,
    threads=128,
    swizzle_panel=0,
    swizzle_order="row",
    dtype=T.float16,
    accum_dtype=T.float32,
):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        tiles_n = T.ceildiv(N, block_N)
        tiles_m = T.ceildiv(M, block_M)
        use_mlx_swizzle = swizzle_panel and swizzle_order == "mlx"
        grid_n = tiles_n * swizzle_panel if use_mlx_swizzle else tiles_n
        grid_m = T.ceildiv(tiles_m, swizzle_panel) if use_mlx_swizzle else tiles_m
        with T.Kernel(grid_n, grid_m, threads=threads) as (bx, by):
            logical_bx = bx // swizzle_panel if use_mlx_swizzle else bx
            logical_by = by * swizzle_panel + bx % swizzle_panel if use_mlx_swizzle else by

            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel, order=swizzle_order)

            if use_mlx_swizzle:
                if logical_by < tiles_m:
                    T.gemm(
                        A[logical_by * block_M : (logical_by + 1) * block_M, 0:K],
                        B[0:K, logical_bx * block_N : (logical_bx + 1) * block_N],
                        C[
                            logical_by * block_M : (logical_by + 1) * block_M,
                            logical_bx * block_N : (logical_bx + 1) * block_N,
                        ],
                        clear_accum=True,
                    )
            else:
                T.gemm(
                    A[logical_by * block_M : (logical_by + 1) * block_M, 0:K],
                    B[0:K, logical_bx * block_N : (logical_bx + 1) * block_N],
                    C[
                        logical_by * block_M : (logical_by + 1) * block_M,
                        logical_bx * block_N : (logical_bx + 1) * block_N,
                    ],
                    clear_accum=True,
                )

    return gemm_kernel


def _tflops(M, N, K, seconds):
    return 2.0 * M * N * K / seconds / 1e12


def _bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats


def bench_torch_mps(M, N, K, warmup, repeats):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    avg_s = _bench(lambda: torch.mm(a, b), warmup, repeats)
    return _tflops(M, N, K, avg_s)


def bench_mlx(M, N, K, warmup, repeats):
    try:
        import mlx.core as mx
    except ImportError:
        return None

    a = mx.random.normal((M, K)).astype(mx.float16)
    b = mx.random.normal((K, N)).astype(mx.float16)
    mx.eval(a, b)

    for _ in range(warmup):
        c = a @ b
        mx.eval(c)

    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
        mx.eval(c)
    return _tflops(M, N, K, (time.perf_counter() - t0) / repeats)


def bench_tilelang(mode, M, N, K, block_M, block_N, block_K, threads, swizzle_panel, swizzle_order, warmup, repeats):
    output_dtype = T.float16
    if mode == "ct_shared":
        kernel = matmul_cooperative_tensor_shared_c(M, N, K, block_M, block_N, block_K, accum_dtype=output_dtype)
    elif mode == "ct_global":
        kernel = matmul_cooperative_tensor_global(
            M, N, K, block_M, block_N, threads, swizzle_panel, swizzle_order, accum_dtype=output_dtype
        )
    else:
        kernel = matmul_simdgroup(M, N, K, block_M, block_N, block_K, accum_dtype=output_dtype)
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=output_dtype.as_torch(), device="mps")
    avg_s = _bench(lambda: kernel(a, b, c), warmup, repeats)
    return _tflops(M, N, K, avg_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metal GEMM Benchmark (simdgroup)")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Sweep all block configs instead of using default CT config")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k

    print(f"torch:    {torch.__version__}")
    print(f"tilelang: {tilelang.__version__}")
    print(f"MPS:      {torch.backends.mps.is_available()}")
    print(f"M={M}, N={N}, K={K}, warmup={args.warmup}, repeats={args.repeats}")
    print()

    ref_tflops = bench_torch_mps(M, N, K, args.warmup, args.repeats)
    print(f"PyTorch MPS (torch.mm fp16): {ref_tflops:.1f} TFLOPS")
    mlx_tflops = bench_mlx(M, N, K, args.warmup, args.repeats)
    if mlx_tflops is not None:
        print(f"MLX matmul fp16:           {mlx_tflops:.1f} TFLOPS")
    print()

    configs = BLOCK_CONFIGS if args.sweep else [("ct_global", 64, 128, 0, 128, 0, "row")]

    print(f"{'path':>10s} | {'block (M,N,K)':>16s} | {'thr':>4s} | {'swizzle':>8s} | {'TileLang':>14s} | {'vs Torch':>8s} | {'vs MLX':>8s}")
    print("-" * 88)

    best_tflops = 0.0
    best_config = configs[0]
    for mode, bM, bN, bK, threads, swizzle_panel, swizzle_order in configs:
        block_text = f"({bM},{bN},{bK if bK else 'all'})"
        swizzle_text = f"{swizzle_panel}:{swizzle_order}" if swizzle_panel else "-"
        try:
            tl = bench_tilelang(mode, M, N, K, bM, bN, bK, threads, swizzle_panel, swizzle_order, args.warmup, args.repeats)
            torch_ratio = tl / ref_tflops * 100
            mlx_ratio = tl / mlx_tflops * 100 if mlx_tflops else None
            if tl > best_tflops:
                best_tflops = tl
                best_config = (mode, bM, bN, bK, threads, swizzle_panel, swizzle_order)
            mlx_text = f"{mlx_ratio:>7.0f}%" if mlx_ratio is not None else "     N/A"
            print(
                f"{mode:>10s} | {block_text:>16s} | {threads:>4d} | {swizzle_text:>8s} | "
                f"{tl:>10.1f} TFLOPS | {torch_ratio:>7.0f}% | {mlx_text}"
            )
        except Exception as e:
            print(f"{mode:>10s} | {block_text:>16s} | {threads:>4d} | {swizzle_text:>8s} | {'FAILED':>14s} | {e}")

    if args.sweep:
        print()
        print(f"Best config: {best_config}")
        print(f"Best TFlops: {best_tflops:.1f}")
        print(f"Reference TFlops (PyTorch MPS): {ref_tflops:.1f}")
