"""Isolate the reciprocal-selection cost of a narrow launch divisor."""

import json
from pathlib import Path
import random
import statistics
import torch
import tilelang
import tilelang.language as T
from bench_invariant_arithmetic import capture, measure


def kernel(size, divisor_dtype, remainder):
    @T.prim_func
    def main(B: T.Tensor((size,), "int64"), d: T.dtype(divisor_dtype)):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                x = (T.int64(bx) * 128 + T.int64(lane)) * 1000003 + T.int64(2**48)
                if remainder:
                    B[bx * 128 + lane] = x % d
                else:
                    B[bx * 128 + lane] = x // d * 67 + x % d

    return main


def main():
    root = Path(".cache/launch_plan/barrett_width")
    root.mkdir(parents=True, exist_ok=True)
    size = 1048576
    x = torch.arange(size, device="cuda", dtype=torch.int64) * 1000003 + 2**48
    out = torch.empty_like(x)
    rows = []
    for remainder in [False, True]:
        kernels = {}
        for dtype in ["int32", "int64"]:
            k = tilelang.compile(
                kernel(size, dtype, remainder),
                target="cuda",
                target_host="c",
                execution_backend="tvm_ffi",
                pass_configs={"tl.enable_invariant_arithmetic": True},
            )
            kernels[dtype] = k
            (root / f"{remainder}_{dtype}.cu").write_text(k.get_kernel_source())
            (root / f"{remainder}_{dtype}.host.txt").write_text(k.get_host_source())
        for d in [1, 37, 2147483647]:
            expected = x % d if remainder else x // d * 67 + x % d
            variants = []
            for dtype, k in kernels.items():

                def call(k=k, d=d):
                    k(out, d)

                call()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = capture(call, out, expected, 256)
                variants.append((dtype, call, graph))
            samples = {dtype: [] for dtype in kernels}
            rng = random.Random(3261)
            for _ in range(11):
                order = list(variants)
                rng.shuffle(order)
                for dtype, call, graph in order:
                    samples[dtype].append(measure(graph, call, 256))
            for dtype, s in samples.items():
                row = dict(
                    remainder=remainder,
                    divisor=d,
                    dtype=dtype,
                    gpu_us=statistics.median(x[0] for x in s),
                    wall_us=statistics.median(x[1] for x in s),
                    samples=s,
                )
                rows.append(row)
                print(row, flush=True)
            (root / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
