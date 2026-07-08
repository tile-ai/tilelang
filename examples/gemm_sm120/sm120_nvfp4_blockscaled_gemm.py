"""Simple SM120 NVFP4 block-scaled GEMM example.

This example demonstrates the intended user flow:

* quantize BF16 operands to packed NVFP4 plus block-scaled UE4M3 scales;
* run a non-persistent TileLang GEMM with TMA loads and auto warp-specialization;
* print a small benchmark result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.quantize import (
    decode_packed_fp4_e2m1,
    decode_ue4m3_scale_bytes,
    quantize_bf16_to_nvfp4_blockscaled,
    unswizzle_blockscaled_chunk_kmajor_scale_words,
)


def _tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return 2.0 * m * n * k / (latency_ms * 1.0e-3) / 1.0e12


@tilelang.jit(out_idx=None)
def nvfp4_blockscaled_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    out_dtype,
):
    """Non-persistent SM120 NVFP4 block-scaled GEMM kernel."""

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_K % 64 == 0
    assert num_stages >= 2

    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    sf_granularity_k = 16
    sf_words_per_block_k = block_K // 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
            SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            data_ready = T.alloc_barrier(arrive_count=128)
            compute_done = T.alloc_barrier(arrive_count=128)

            with T.ws(0):
                T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                with T.ws(1):
                    T.barrier_wait(compute_done, (ko + 1) % 2)
                    T.tma_copy(
                        A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                        A_shared,
                        barrier=data_ready,
                    )
                    T.tma_copy(
                        B[bx * block_N : (bx + 1) * block_N, ko * block_K : (ko + 1) * block_K],
                        B_shared,
                        barrier=data_ready,
                    )
                    T.tma_copy(
                        SFA[
                            by * block_M : (by + 1) * block_M,
                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                        ],
                        SFA_shared,
                        barrier=data_ready,
                    )
                    T.tma_copy(
                        SFB[
                            bx * block_N : (bx + 1) * block_N,
                            ko * sf_words_per_block_k : (ko + 1) * sf_words_per_block_k,
                        ],
                        SFB_shared,
                        barrier=data_ready,
                    )
                    T.barrier_arrive(data_ready)

                with T.ws(0):
                    T.barrier_wait(data_ready, ko % 2)
                    T.mma_gemm_blockscaled(
                        A_shared,
                        B_shared,
                        C_local,
                        SFA_shared,
                        SFB_shared,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.Square,
                        clear_accum=False,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.barrier_arrive(compute_done)

            with T.ws(0):
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def _reference_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA_bytes: torch.Tensor,
    SFB_bytes: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    a = decode_packed_fp4_e2m1(A)
    b = decode_packed_fp4_e2m1(B)
    sfa = decode_ue4m3_scale_bytes(SFA_bytes).repeat_interleave(16, dim=1)
    sfb = decode_ue4m3_scale_bytes(SFB_bytes).repeat_interleave(16, dim=1)
    return ((a * sfa) @ (b * sfb).T).to(out_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--warmup-ms", type=float, default=5)
    parser.add_argument("--rep-ms", type=float, default=20)
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (12, 0):
        raise RuntimeError(f"SM120 or newer is required, got compute capability {capability}")

    torch.manual_seed(args.seed)
    out_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32

    A_bf16 = torch.randn((args.m, args.k), device="cuda", dtype=torch.bfloat16)
    B_bf16 = torch.randn((args.n, args.k), device="cuda", dtype=torch.bfloat16)
    A, SFA_source, SFA_bytes = quantize_bf16_to_nvfp4_blockscaled(A_bf16, return_scale_bytes=True)
    B, SFB_source, SFB_bytes = quantize_bf16_to_nvfp4_blockscaled(B_bf16, return_scale_bytes=True)
    SFA = unswizzle_blockscaled_chunk_kmajor_scale_words(SFA_source)
    SFB = unswizzle_blockscaled_chunk_kmajor_scale_words(SFB_source)
    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)

    kernel = nvfp4_blockscaled_gemm(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        args.threads,
        out_dtype,
    )
    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.verify:
        ref = _reference_gemm(A, B, SFA_bytes, SFB_bytes, out_torch_dtype)
        torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)
        print("TileLang correctness: passed")

    latency_ms = do_bench(
        lambda: kernel(A, B, SFA, SFB, C),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        backend=args.backend,
        return_mode=args.return_mode,
    )
    print(f"Shape: M={args.m}, N={args.n}, K={args.k}")
    print(f"Tile: {args.block_m}x{args.block_n}x{args.block_k}, stages={args.num_stages}, threads={args.threads}")
    print(f"TileLang latency: {latency_ms:.4f} ms")
    print(f"TileLang FLOPS: {_tflops(args.m, args.n, args.k, latency_ms):.2f} TFLOPS")


if __name__ == "__main__":
    main()
