"""Benchmark script for the SM100 bf16×fp8→bf16 grouped GEMM.

Exercises `example_grouped_gemm_fwd_sm100_fp8_to_bf16.grouped_gemm_sm100_fp8_to_bf16`
over user-supplied (G, M, N, K) shapes and reports per-shape latency / TFLOPS.
Supports either uniform per-group M (--G/--M) or fully heterogeneous group
sizes (--batch_sizes), and can run a list of shapes in one invocation via
--shapes.

Examples
--------
# Uniform: 8 groups, M=1024 rows each, N=4096, K=4096
python benchmark_grouped_gemm_fp8_to_bf16.py --G 8 --M 1024 --N 4096 --K 4096

# Heterogeneous group sizes
python benchmark_grouped_gemm_fp8_to_bf16.py --batch_sizes 63,77,111,280 --N 2048 --K 2048

# Sweep multiple shapes (G,M,N,K), one per ';'
python benchmark_grouped_gemm_fp8_to_bf16.py --shapes "4,512,4096,4096;8,1024,4096,4096"
"""

import argparse
import torch

import tilelang
from example_grouped_gemm_fwd_sm100_fp8_to_bf16 import (
    grouped_gemm_sm100_fp8_to_bf16,
    construct_inputs_bf16_fp8,
    torch_gmm_bf16_fp8_ref,
)


# Defaults align with the kernel's defaults; SMEM budget assumes block_N=256.
DEFAULTS = {
    "block_M": 128,
    "block_N": 256,
    "block_K": 64,
    "num_stages": 3,
}


def parse_batch_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_shapes(s: str) -> list[tuple[int, int, int, int]]:
    shapes = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(x) for x in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Each shape must be 'G,M,N,K' — got '{chunk}'")
        shapes.append(tuple(parts))
    return shapes


def bench_one(
    batch_sizes_list: list[int],
    K: int,
    N: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    warmup_ms: int,
    rep_ms: int,
    return_mode: str,
    check: bool,
) -> dict:
    G = len(batch_sizes_list)
    total_M = sum(batch_sizes_list)
    device = torch.device("cuda")

    kernel = grouped_gemm_sm100_fp8_to_bf16(
        tuple(batch_sizes_list), K, N,
        block_M=block_M, block_N=block_N, block_K=block_K, num_stages=num_stages,
    )

    A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs_bf16_fp8(
        batch_sizes_list, K, N, padding_M=block_M, device=device,
    )

    correct = None
    if check:
        out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
        ref = torch_gmm_bf16_fp8_ref(A, B, batch_sizes_list, batch_offsets.tolist())
        correct = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    latency_ms = profiler.do_bench(
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode=return_mode,
        input_tensors=[A, B, batch_sizes, batch_offsets, batch_padded_offsets],
    )

    flops = 2 * total_M * N * K
    tflops = flops / latency_ms * 1e-9

    return {
        "G": G,
        "total_M": total_M,
        "N": N,
        "K": K,
        "batch_sizes": batch_sizes_list,
        "latency_ms": latency_ms,
        "tflops": tflops,
        "correct": correct,
    }


def fmt_row(r: dict) -> str:
    bs = ",".join(str(x) for x in r["batch_sizes"])
    if len(bs) > 28:
        bs = bs[:25] + "..."
    correct = "-" if r["correct"] is None else ("ok" if r["correct"] else "FAIL")
    return (
        f"{r['G']:>4}  {r['total_M']:>8}  {r['N']:>6}  {r['K']:>6}  "
        f"{r['latency_ms']:>10.4f}  {r['tflops']:>9.2f}  {correct:>5}  {bs}"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Shape inputs (precedence: --shapes > --batch_sizes > --G/--M)
    p.add_argument("--shapes", type=str, default=None,
                   help="Sweep multiple shapes: 'G,M,N,K;G,M,N,K;...'. Overrides single-shape args.")
    p.add_argument("--batch_sizes", type=str, default=None,
                   help="Heterogeneous group sizes (comma-separated). Overrides --G/--M.")
    p.add_argument("--G", type=int, default=8, help="number of groups (uniform sizes)")
    p.add_argument("--M", type=int, default=1024, help="rows per group (uniform sizes)")
    p.add_argument("--N", type=int, default=4096, help="output dim")
    p.add_argument("--K", type=int, default=4096, help="reduce dim")

    # Tiling / launch knobs
    p.add_argument("--block_M", type=int, default=DEFAULTS["block_M"])
    p.add_argument("--block_N", type=int, default=DEFAULTS["block_N"],
                   help="must be a multiple of 256 (32 lanes × vec-of-8)")
    p.add_argument("--block_K", type=int, default=DEFAULTS["block_K"])
    p.add_argument("--num_stages", type=int, default=DEFAULTS["num_stages"])

    # Bench knobs
    p.add_argument("--warmup_ms", type=int, default=100, help="do_bench warmup time (ms)")
    p.add_argument("--rep_ms", type=int, default=200, help="do_bench timing window (ms)")
    p.add_argument("--return_mode", type=str, default="median",
                   choices=["min", "max", "mean", "median"])
    p.add_argument("--no_check", action="store_true",
                   help="skip correctness check vs torch fp16 reference")
    args = p.parse_args()

    if args.shapes is not None:
        configs = []
        for G, M, N, K in parse_shapes(args.shapes):
            configs.append(([M] * G, N, K))
    elif args.batch_sizes is not None:
        configs = [(parse_batch_sizes(args.batch_sizes), args.N, args.K)]
    else:
        configs = [([args.M] * args.G, args.N, args.K)]

    print(f"Running {len(configs)} shape(s) on {torch.cuda.get_device_name()} "
          f"[bf16 × fp8→bf16, "
          f"block_M={args.block_M} block_N={args.block_N} block_K={args.block_K} "
          f"stages={args.num_stages}]")
    print(f"   G  total_M       N       K   latency_ms     TFLOPS  check  batch_sizes")
    print("-" * 90)

    results = []
    for batch_sizes_list, N, K in configs:
        try:
            r = bench_one(
                batch_sizes_list=batch_sizes_list,
                K=K,
                N=N,
                block_M=args.block_M,
                block_N=args.block_N,
                block_K=args.block_K,
                num_stages=args.num_stages,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                return_mode=args.return_mode,
                check=not args.no_check,
            )
        except Exception as e:
            print(f"[ERROR] G={len(batch_sizes_list)} N={N} K={K}: {e}")
            continue
        results.append(r)
        print(fmt_row(r))

    if not results:
        return
    if any(r["correct"] is False for r in results):
        print("\nNote: at least one shape failed the correctness check.")


if __name__ == "__main__":
    main()
