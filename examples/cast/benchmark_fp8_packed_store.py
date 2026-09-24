"""Benchmark packed BF16-to-FP8 conversions followed by vector stores.

Run on either revision to compare the generated CUDA and runtime:
    python examples/cast/benchmark_fp8_packed_store.py --dtype float8_e4m3fn

The repeated stores intentionally stress conversion/packing rather than model
an end-to-end quantization kernel. Only the last round is returned.
"""

import argparse
import statistics
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


def cast_kernel(blocks, dtype):
    lanes, threads, repeats = 8, 256, 16
    size = blocks * threads * lanes

    @T.prim_func
    def main(x: T.Tensor((size,), T.bfloat16), y: T.Tensor((size,), dtype)):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            local = T.alloc_local((lanes,), T.bfloat16)
            shared = T.alloc_shared((repeats, threads * lanes), dtype)
            for i in T.vectorized(lanes):
                local[i] = x[(bx * threads + tx) * lanes + i]
            for r in T.unroll(repeats):
                for i in T.vectorized(lanes):
                    shared[r, tx * lanes + i] = local[i] * T.bfloat16(1.0 / (r + 1))
            for i in T.vectorized(lanes):
                y[(bx * threads + tx) * lanes + i] = shared[repeats - 1, tx * lanes + i]

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"], default="float8_e4m3fn")
    parser.add_argument("--blocks", type=int, help="defaults to 64 blocks per SM")
    parser.add_argument("--source", type=Path, help="write the generated CUDA source")
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    blocks = args.blocks if args.blocks is not None else props.multi_processor_count * 64
    kernel = tilelang.compile(cast_kernel(blocks, args.dtype), target="cuda")
    if args.source:
        args.source.write_text(kernel.get_kernel_source())
    torch.manual_seed(0)
    x = torch.randn(blocks * 256 * 8, device="cuda", dtype=torch.bfloat16)
    if args.dtype == "float8_e8m0fnu":
        # Positive normal values give an exact reference for E8M0's round-up
        # conversion: encode the exponent of the next power of two.
        x = x.abs() + 1
    y = torch.empty_like(x, dtype=getattr(torch, args.dtype))
    kernel(x, y)
    scaled = x * torch.tensor(1 / 16, dtype=torch.bfloat16, device="cuda")
    if args.dtype == "float8_e8m0fnu":
        expected = (scaled.float().log2().ceil() + 127).to(torch.uint8)
    else:
        expected = scaled.to(y.dtype).view(torch.uint8)
    assert torch.equal(y.view(torch.uint8), expected)
    timings = [tilelang.profiler.do_bench(lambda: kernel(x, y), warmup=100, rep=500) * 1000 for _ in range(3)]
    print(f"{props.name}, sm_{props.major}{props.minor}, {blocks} blocks, {args.dtype}")
    print(f"Median: {statistics.median(timings):.2f} us; samples: {timings}")


if __name__ == "__main__":
    main()
