"""Benchmark script for grouped GEMM.

Sweeps tilelang's grouped GEMM kernel over user-supplied (G, M, N, K) shapes
and reports per-shape latency / TFLOPS. Supports either a uniform per-group
M (via --G/--M) or fully heterogeneous group sizes (via --batch_sizes), and
can run a list of shapes in one invocation via --shapes.

Examples
--------
# Uniform: 8 groups, M=1024 rows each, N=4096, K=4096
python benchmark_grouped_gemm.py --G 8 --M 1024 --N 4096 --K 4096

# Heterogeneous group sizes
python benchmark_grouped_gemm.py --batch_sizes 63,77,111,280 --N 8192 --K 8192

# Sweep multiple shapes (G,M,N,K), one per ';'
python benchmark_grouped_gemm.py --shapes "4,512,4096,4096;8,1024,4096,4096;16,2048,8192,8192"
"""

import argparse
import torch

import tilelang
from example_grouped_gemm_fwd import grouped_gemm, construct_inputs, torch_gmm
from example_grouped_gemm_fwd_sm100 import grouped_gemm_sm100
from example_grouped_gemm_fwd_sm100_fp8 import (
    grouped_gemm_sm100_fp8,
    construct_inputs_fp8,
    torch_gmm_fp8_ref,
)


# Per-backend defaults for tiling/dtype/stages.
BACKEND_DEFAULTS = {
    "sm80":      {"block_M":  64, "block_N": 128, "block_K":  64, "num_stages": 2,
                  "threads": 256, "dtype": torch.float16},
    "sm100":     {"block_M": 128, "block_N": 256, "block_K":  64, "num_stages": 4,
                  "threads": 128, "dtype": torch.bfloat16},
    "sm100_fp8": {"block_M": 128, "block_N": 256, "block_K": 128, "num_stages": 4,
                  "threads": 128, "dtype": torch.float8_e4m3fn},  # output is bf16
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
    backend: str,
    batch_sizes_list: list[int],
    K: int,
    N: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    warmup_ms: int,
    rep_ms: int,
    return_mode: str,
    check: bool,
) -> dict:
    """Build, optionally validate, and benchmark one shape. Returns a result dict."""
    G = len(batch_sizes_list)
    total_M = sum(batch_sizes_list)

    device = torch.device("cuda")

    if backend == "sm80":
        kernel = grouped_gemm(
            tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages, threads,
        )
        dtype = BACKEND_DEFAULTS[backend]["dtype"]
        A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(
            batch_sizes_list, K, N, trans_b=False, padding_M=block_M, device=device, dtype=dtype,
        )
    elif backend == "sm100":
        # SM100 kernel is hard-coded to 128 threads (warp 0 = TMA, warp 1 = tcgen05).
        kernel = grouped_gemm_sm100(
            tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages,
        )
        dtype = BACKEND_DEFAULTS[backend]["dtype"]
        A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(
            batch_sizes_list, K, N, trans_b=False, padding_M=block_M, device=device, dtype=dtype,
        )
    elif backend == "sm100_fp8":
        kernel = grouped_gemm_sm100_fp8(
            tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages,
        )
        # FP8 in / BF16 out, B has (G, N, K) layout (not (G, K, N)).
        A, B, _C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs_fp8(
            batch_sizes_list, K, N, padding_M=block_M, device=device,
            in_torch_dtype=BACKEND_DEFAULTS[backend]["dtype"],
            out_torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"unknown backend '{backend}'")

    correct = None
    if check:
        out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
        if backend == "sm100_fp8":
            ref = torch_gmm_fp8_ref(A, B, batch_sizes_list, batch_offsets.tolist(),
                                    out_torch_dtype=torch.bfloat16)
            correct = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)
        else:
            ref = torch_gmm(A, B, batch_sizes, batch_offsets, trans_b=False)
            correct = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)

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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Backend selection.
    p.add_argument("--backend", type=str, default="sm100",
                   choices=["sm80", "sm100", "sm100_fp8"],
                   help="sm80: T.gemm (mma) fp16; sm100: TMA + tcgen05.mma bf16 (WS); "
                        "sm100_fp8: TMA + tcgen05.mma fp8_e4m3 -> bf16 (WS, B is (G,N,K))")

    # Shape inputs (mutually-supplementary; precedence: --shapes > --batch_sizes > --G/--M)
    p.add_argument("--shapes", type=str, default=None,
                   help="Sweep multiple shapes: 'G,M,N,K;G,M,N,K;...'. Overrides single-shape args.")
    p.add_argument("--batch_sizes", type=str, default=None,
                   help="Heterogeneous group sizes (comma-separated). Overrides --G/--M.")
    p.add_argument("--G", type=int, default=8, help="number of groups (uniform sizes)")
    p.add_argument("--M", type=int, default=1024, help="rows per group (uniform sizes)")
    p.add_argument("--N", type=int, default=4096, help="output dim")
    p.add_argument("--K", type=int, default=4096, help="reduce dim")

    # Tiling / launch knobs (None => take per-backend defaults from BACKEND_DEFAULTS).
    p.add_argument("--block_M", type=int, default=None)
    p.add_argument("--block_N", type=int, default=None)
    p.add_argument("--block_K", type=int, default=None)
    p.add_argument("--num_stages", type=int, default=None)
    p.add_argument("--threads", type=int, default=None,
                   help="thread count (sm80 only; sm100 is fixed at 128)")

    # Bench knobs
    p.add_argument("--warmup_ms", type=int, default=100, help="do_bench warmup time (ms)")
    p.add_argument("--rep_ms", type=int, default=200, help="do_bench timing window (ms)")
    p.add_argument("--return_mode", type=str, default="median",
                   choices=["min", "max", "mean", "median"])
    p.add_argument("--no_check", action="store_true", help="skip correctness check vs torch")
    args = p.parse_args()

    # Fill in backend-specific defaults for any tiling knob the user didn't set.
    defaults = BACKEND_DEFAULTS[args.backend]
    block_M = args.block_M if args.block_M is not None else defaults["block_M"]
    block_N = args.block_N if args.block_N is not None else defaults["block_N"]
    block_K = args.block_K if args.block_K is not None else defaults["block_K"]
    num_stages = args.num_stages if args.num_stages is not None else defaults["num_stages"]
    threads = args.threads if args.threads is not None else defaults["threads"]

    # Build the list of shapes to run.
    if args.shapes is not None:
        configs = []
        for G, M, N, K in parse_shapes(args.shapes):
            configs.append(([M] * G, N, K))
    elif args.batch_sizes is not None:
        configs = [(parse_batch_sizes(args.batch_sizes), args.N, args.K)]
    else:
        configs = [([args.M] * args.G, args.N, args.K)]

    print(f"Running {len(configs)} shape(s) on {torch.cuda.get_device_name()} "
          f"[backend={args.backend}, dtype={defaults['dtype']}, "
          f"block_M={block_M} block_N={block_N} block_K={block_K} stages={num_stages}]")
    print(
        f"   G  total_M       N       K   latency_ms     TFLOPS  check  batch_sizes"
    )
    print("-" * 90)

    results = []
    for batch_sizes_list, N, K in configs:
        try:
            r = bench_one(
                backend=args.backend,
                batch_sizes_list=batch_sizes_list,
                K=K,
                N=N,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                num_stages=num_stages,
                threads=threads,
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
