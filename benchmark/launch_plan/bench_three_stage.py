"""Same-process three-stage invariant arithmetic benchmark."""

import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch
import tilelang
from bench_invariant_arithmetic import workload, capture, measure
from bench_materialize_paired import stress_kernel
from bench_barrett_width import kernel as narrow_divisor_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path(".cache/launch_plan/three_stage"))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    materialize = tilelang.transform.MaterializeInvariantArithmetic
    results = []
    cases = [
        "decode",
        "gather",
        "aligned_gather",
        "banked",
        "wide_layout",
        "wide_remainder",
        "wide_i32_layout",
        "wide_i32_remainder",
        "stress6",
        "gather_nonzero",
        "gather_positive",
        "gather_bounded",
        "gather_unsigned",
    ]
    for size in args.sizes:
        for case in cases:
            if case == "stress6":
                torch.manual_seed(3261)
                a = torch.randint(-1000000, 1000000, (size,), device="cuda", dtype=torch.int32)
                out = torch.empty_like(a)
                expected = a.clone()
                for _ in range(6):
                    expected = expected // 7 + expected % 13
                func = stress_kernel(size, 6)
                inputs = (a, out, 7, 13)
            elif case.startswith("wide_i32"):
                remainder = case.endswith("remainder")
                x = torch.arange(size, device="cuda", dtype=torch.int64) * 1000003 + 2**48
                out = torch.empty_like(x)
                expected = x % 37 if remainder else x // 37 * 67 + x % 37
                func = narrow_divisor_kernel(size, "int32", remainder)
                inputs = (out, 37)
            else:
                func, inputs, out, expected = workload(case, size)
            variants = []
            hosts = {}
            for mode in ["off", "lower", "late"]:
                tilelang.transform.MaterializeInvariantArithmetic = materialize if mode == "late" else lambda: lambda mod: mod
                begin = time.perf_counter()
                try:
                    k = tilelang.compile(
                        func,
                        target="cuda",
                        target_host="c",
                        execution_backend="tvm_ffi",
                        pass_configs={"tl.enable_invariant_arithmetic": mode != "off"},
                    )
                finally:
                    tilelang.transform.MaterializeInvariantArithmetic = materialize
                compile_s = time.perf_counter() - begin
                stem = f"{case}_{size}_{mode}"
                source = k.get_kernel_source()
                hosts[mode] = k.get_host_source()
                (root / (stem + ".cu")).write_text(source)
                (root / (stem + ".host.txt")).write_text(hosts[mode])

                def call(k=k, inputs=inputs):
                    k(*inputs)

                call()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = capture(call, out, expected, args.count)
                variants.append((mode, k, call, graph, compile_s, len(source.encode())))
            assert hosts["lower"] == hosts["late"]
            # Warm every variant before collecting the interleaved samples.
            for _, _, _, graph, _, _ in variants:
                for _ in range(10):
                    graph.replay()
            torch.cuda.synchronize()
            samples = {mode: [] for mode, *_ in variants}
            rng = random.Random(3261)
            for _ in range(args.rounds):
                order = list(variants)
                rng.shuffle(order)
                for mode, _, call, graph, _, _ in order:
                    samples[mode].append(measure(graph, call, args.count))
            for mode, _, _, _, compile_s, source_bytes in variants:
                rows = samples[mode]
                row = dict(
                    case=case,
                    size=size,
                    mode=mode,
                    gpu_us=statistics.median(x[0] for x in rows),
                    wall_us=statistics.median(x[1] for x in rows),
                    samples=rows,
                    compile_s=compile_s,
                    cuda_bytes=source_bytes,
                )
                results.append(row)
                print({k: v for k, v in row.items() if k != "samples"}, flush=True)
            (root / "results.json").write_text(
                json.dumps(
                    dict(device=torch.cuda.get_device_name(), rounds=args.rounds, count=args.count, sizes=args.sizes, results=results),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
