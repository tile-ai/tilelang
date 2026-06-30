"""Experimental SM120 NVFP4 GEMM using TileLang warp-specialization blocks.

This file is intentionally separate from the main benchmark. It explores the
existing ``T.ws`` abstraction for a compact producer/consumer shape:

* ``T.ws(1)`` issues TMA copies for A, B, SFA, and SFB.
* ``T.ws(0)`` waits on the stage barrier and issues ``T.mma_gemm_blockscaled``.

The scale tensors use the semantic row-major ``[M or N, K / 64]`` uint32
contract. This example is for API/lowering study first; it is not the persistent
pingpong performance path.
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
from tilelang.profiler import do_bench

from sm120_nvfp4_blockscaled_gemm import (
    _make_binary_scale_words,
    _make_constant_scale_words,
    _make_ones_packed_fp4,
    _make_packed_fp4,
    _tflops,
    _verify_tilelang_output,
)


def _jit_pass_configs(ptxas_verbose: bool = False) -> dict:
    pass_configs = {}
    if ptxas_verbose:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT] = True
    return pass_configs


@tilelang.jit(out_idx=None, pass_configs=_jit_pass_configs())
def tilelang_nvfp4_blockscaled_ws(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    warp_policy,
    out_dtype,
):
    """Compact T.ws producer/consumer NVFP4 block-scaled GEMM."""

    assert M % block_M == 0
    assert N % block_N == 0
    assert K % block_K == 0
    assert block_K % 64 == 0
    assert num_stages >= 2

    in_dtype = T.float4_e2m1fn
    accum_dtype = T.float32
    sf_words_per_block_k = block_K // 64
    sf_granularity_k = 16

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
                        policy=warp_policy,
                        clear_accum=False,
                        k_start=0,
                        sf_a_granularity_k=sf_granularity_k,
                        sf_b_granularity_k=sf_granularity_k,
                    )
                    T.barrier_arrive(compute_done)

            with T.ws(0):
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


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
    parser.add_argument("--warp-policy", choices=["Square"], default="Square")
    parser.add_argument("--out-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["random", "ones"], default="random")
    parser.add_argument(
        "--scale-mode",
        choices=["constant", "random_binary", "random_sfa", "random_sfb"],
        default="random_binary",
    )
    parser.add_argument("--warmup-ms", type=float, default=1)
    parser.add_argument("--rep-ms", type=float, default=1)
    parser.add_argument("--backend", choices=["event", "cupti", "cudagraph"], default="event")
    parser.add_argument("--return-mode", choices=["min", "max", "mean", "median"], default="min")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dump-source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability < (12, 0):
        raise RuntimeError(f"SM120 or newer is required, got compute capability {capability}")

    out_dtype = T.bfloat16 if args.out_dtype == "bfloat16" else T.float32
    out_torch_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float32
    warp_policy = getattr(T.GemmWarpPolicy, args.warp_policy)

    kernel = tilelang_nvfp4_blockscaled_ws(
        args.m,
        args.n,
        args.k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        args.threads,
        warp_policy,
        out_dtype,
    )
    source = kernel.get_kernel_source()
    if args.dump_source:
        dump_path = Path(args.dump_source)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(source)
        print(f"TileLang CUDA source: {dump_path}")
    if "warp_specialize" not in source and "kWarpSpecializationScope" not in source:
        print("warning: generated source does not show a textual warp-specialization marker")

    if args.input_mode == "ones":
        A = _make_ones_packed_fp4(args.m, args.k)
        B = _make_ones_packed_fp4(args.n, args.k)
    else:
        A = _make_packed_fp4(args.m, args.k, seed=args.seed)
        B = _make_packed_fp4(args.n, args.k, seed=args.seed + 1)

    if args.scale_mode == "constant":
        SFA = _make_constant_scale_words(args.m, args.k)
        SFB = _make_constant_scale_words(args.n, args.k)
    elif args.scale_mode == "random_binary":
        SFA = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
    elif args.scale_mode == "random_sfa":
        SFA = _make_binary_scale_words(args.m, args.k, seed=args.seed + 100)
        SFB = _make_constant_scale_words(args.n, args.k)
    elif args.scale_mode == "random_sfb":
        SFA = _make_constant_scale_words(args.m, args.k)
        SFB = _make_binary_scale_words(args.n, args.k, seed=args.seed + 200)
    else:
        raise ValueError(f"Unsupported scale_mode={args.scale_mode!r}")

    C = torch.empty((args.m, args.n), device="cuda", dtype=out_torch_dtype)
    kernel(A, B, SFA, SFB, C)
    torch.cuda.synchronize()

    if args.verify:
        _verify_tilelang_output(A, B, SFA, SFB, C, out_torch_dtype, args.scale_mode, args.block_m, args.block_n, driver.get_num_sms())
        print("TileLang WS correctness: passed")

    latency_ms = do_bench(
        lambda: kernel(A, B, SFA, SFB, C),
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        backend=args.backend,
        return_mode=args.return_mode,
    )
    print(f"TileLang WS latency: {latency_ms:.4f} ms")
    print(f"TileLang WS FLOPS: {_tflops(args.m, args.n, args.k, latency_ms):.2f} TFLOPS")


if __name__ == "__main__":
    main()
