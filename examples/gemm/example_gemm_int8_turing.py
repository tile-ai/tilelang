import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit
def matmul_int8_turing(
    M,
    N,
    K,
    block_M=128,
    block_N=256,
    block_K=128,
    threads=256,
    num_stages=1,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bm, bn):
            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bm * block_M, ko * block_K], A_shared)
                T.copy(B[bn * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[bm * block_M, bn * block_N])

    return main


def tops_from_ms(M, N, K, latency_ms):
    return 2.0 * M * N * K / (latency_ms * 1e-3 * 1e12)


def ref_program(A, B):
    return torch._int_mm(A, B.t())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup-ms", type=float, default=30)
    parser.add_argument("--rep-ms", type=float, default=300)
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    print(f"INT8 GEMM for Turing, M={M}, N={N}, K={K}")
    print(f"Device: {torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")

    kernel = matmul_int8_turing(M, N, K)
    A = torch.randint(-50, 50, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-50, 50, (N, K), dtype=torch.int8, device="cuda")

    C = torch.empty((M, N), dtype=torch.int32, device="cuda")
    kernel(A, B, C)
    ref = ref_program(A, B)
    max_diff = torch.max(torch.abs(C - ref)).item()
    assert max_diff == 0, f"max_diff={max_diff}"

    torch_latency = do_bench(
        lambda: ref_program(A, B),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        backend=args.backend,
        return_mode="mean",
    )
    tilelang_latency = do_bench(
        lambda: kernel(A, B, C),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        backend=args.backend,
        return_mode="mean",
    )

    print(f"TileLang latency: {tilelang_latency:.3f} ms")
    print(f"TileLang TOPS: {tops_from_ms(M, N, K, tilelang_latency):.2f}")
    print(f"PyTorch latency: {torch_latency:.3f} ms")
    print(f"PyTorch TOPS: {tops_from_ms(M, N, K, torch_latency):.2f}")
    print(f"TileLang/PyTorch: {torch_latency / tilelang_latency:.3f}x")


if __name__ == "__main__":
    main()
