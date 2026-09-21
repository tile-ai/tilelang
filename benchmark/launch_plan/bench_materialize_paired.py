"""Pair production late materialization enabled/disabled, with unchanged host preparation."""

import json
from pathlib import Path
import random
import statistics

import torch
import tilelang
import tilelang.language as T
from bench_invariant_arithmetic import workload, capture, measure


def stress_kernel(size, depth):
    @T.prim_func
    def main(A: T.Tensor((size,), "int32"), B: T.Tensor((size,), "int32"), d: T.int32, e: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            for lane in T.Parallel(128):
                idx = bx * 128 + lane
                v = T.alloc_local((1,), "int32")
                v[0] = A[idx]
                for _step in T.unroll(depth):
                    v[0] = (v[0] // d) + (v[0] % e)
                B[idx] = v[0]

    return main


def main():
    root = Path(".cache/launch_plan/materialize_paired")
    root.mkdir(parents=True, exist_ok=True)
    materialize = tilelang.transform.MaterializeInvariantArithmetic
    results = []
    for case in ["gather", "wide_layout", "wide_remainder", "aligned_gather", "banked", "stress6"]:
        size = 1048576
        if case == "stress6":
            torch.manual_seed(3261)
            a = torch.randint(-1000000, 1000000, (size,), device="cuda", dtype=torch.int32)
            out = torch.empty_like(a)
            expected = a.clone()
            for _ in range(6):
                expected = expected // 7 + expected % 13
            func = stress_kernel(size, 6)
            inputs = (a, out, 7, 13)
        else:
            func, inputs, out, expected = workload(case, size)
        variants = []
        for enabled in [False, True]:
            tilelang.transform.MaterializeInvariantArithmetic = materialize if enabled else lambda: lambda mod: mod
            try:
                k = tilelang.compile(
                    func, target="cuda", target_host="c", execution_backend="tvm_ffi", pass_configs={"tl.enable_invariant_arithmetic": True}
                )
            finally:
                tilelang.transform.MaterializeInvariantArithmetic = materialize
            stem = case + "_" + str(int(enabled))
            (root / (stem + ".cu")).write_text(k.get_kernel_source())
            (root / (stem + ".host.txt")).write_text(k.get_host_source())

            def call(k=k, inputs=inputs):
                k(*inputs)

            call()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            graph = capture(call, out, expected, 256)
            variants.append((enabled, k, call, graph))
        assert (root / (case + "_0.host.txt")).read_text() == (root / (case + "_1.host.txt")).read_text()
        samples = {False: [], True: []}
        rng = random.Random(3261)
        for _ in range(11):
            order = list(variants)
            rng.shuffle(order)
            for enabled, _k, call, graph in order:
                samples[enabled].append(measure(graph, call, 256))
        for enabled, rows in samples.items():
            row = dict(
                case=case,
                enabled=enabled,
                gpu_us=statistics.median(x[0] for x in rows),
                wall_us=statistics.median(x[1] for x in rows),
                samples=rows,
            )
            results.append(row)
            print(row, flush=True)
        (root / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
