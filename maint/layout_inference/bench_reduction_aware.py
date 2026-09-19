"""SM90 reproducer: equal register scores, different reduction communication.

Each CTA reuses an 8x128 value tile for several weighted sums. Native width 4
and scalar width 1 both cost 18 fragment slots. The old two-attempt policy keeps
the native candidate on this tie; the reduction-aware policy avoids its
cross-warp finalizations. Pinning width 4 reproduces that old winner without
changing the compiler or adding a pass configuration.
"""

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import torch

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tilelang.profiler.bench import do_bench


def make_weighted_pooling(groups=256, queries=32, width=None):
    rows, columns = 8, 128
    width = None if width is None else T.int32(width)

    @T.prim_func
    def main(
        inputs: T.Tensor((groups, rows, columns), "float32"),
        weights: T.Tensor((groups, queries, rows), "float32"),
        output: T.Tensor((groups, queries, columns), "float32"),
    ):
        with T.Kernel(groups, threads=128) as group:
            values = T.alloc_fragment((rows, columns), "float32")
            row_weights = T.alloc_fragment((rows,), "float32")
            acc = T.alloc_reducer((columns,), "float32")
            result = T.alloc_fragment((columns,), "float32")
            T.copy(inputs[group, 0:rows, 0:columns], values)
            for query in T.serial(queries):
                T.copy(weights[group, query, 0:rows], row_weights)
                T.reducer_init(acc)
                for row, column in T.Parallel(rows, columns, coalesced_width=width):
                    T.reducer_update(acc[column], values[row, column] * row_weights[row])
                T.finalize_reducer(acc, result)
                T.copy(result, output[group, query, 0:columns])

    return main


def inspect_plan(function, target):
    with target, tl.transform.PassContext():
        module = tvm.IRModule({"main": function})
        module = tvm.tirx.transform.BindTarget(target)(module)
        module = tl.transform.MaterializeKernelLaunch()(module)
        module = tl.transform.LayoutInference()(module)
    buffer_slots = {}

    def collect(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buffer, layout in node.annotations.get("layout_map", {}).items():
                if isinstance(layout, tl.Fragment):
                    buffer_slots[buffer.name] = math.prod(int(extent) for extent in layout.get_output_shape())

    tvm.tirx.stmt_functor.post_order_visit(module["main"].body, collect)
    costs = tvm.get_global_func("tl.analysis.ReducerCost")(module["main"])
    cost = next(cost for buffer, cost in costs.items() if buffer.name == "acc")
    assert cost["known"]
    return {
        "width": int(cost["vector_widths"][0]),
        "register_slots": sum(buffer_slots.values()),
        "buffer_slots": buffer_slots,
        "local_issues": int(cost["local_issues"]),
        "barriers": int(cost["barriers"]),
        "shuffle_issues": int(cost["shuffle_issues"]),
    }


def benchmark(groups, queries, rounds, repeat, target):
    kernels, plans, sources = {}, {}, {}
    for name, width in (("native", 4), ("scalar", 1), ("auto", None)):
        function = make_weighted_pooling(groups, queries, width)
        plans[name] = inspect_plan(function, target)
        kernels[name] = tl.compile(function, target=target)
        sources[name] = kernels[name].get_kernel_source()
    assert all(plan["register_slots"] == 18 for plan in plans.values())
    assert plans["native"]["width"] == 4 and plans["native"]["barriers"] == 16 * queries
    assert plans["auto"]["width"] == 1 and plans["auto"]["barriers"] == 0
    assert sources["auto"] == sources["scalar"]

    output = torch.empty((groups, queries, 128), device="cuda")
    for integer_inputs in (True, False):
        if integer_inputs:
            inputs = torch.randint(-2, 3, (groups, 8, 128), device="cuda").float()
            weights = torch.randint(-2, 3, (groups, queries, 8), device="cuda").float()
        else:
            inputs = torch.randn((groups, 8, 128), device="cuda")
            weights = torch.randn((groups, queries, 8), device="cuda")
        expected = (weights.unsqueeze(-1) * inputs.unsqueeze(1)).sum(dim=2)
        tolerance = 0 if integer_inputs else 1e-5
        for kernel in kernels.values():
            kernel(inputs, weights, output)
            torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)

    samples = {"native": [], "auto": []}
    for round_index in range(rounds):
        order = list(samples)
        random.Random(round_index).shuffle(order)
        for name in order:
            elapsed_ms = do_bench(
                partial(kernels[name], inputs, weights, output),
                backend="cudagraph",
                return_mode="median",
                _n_repeat=repeat,
                _n_warmup=32,
            )
            samples[name].append(elapsed_ms * 1000)
    latencies = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "groups": groups,
        "queries": queries,
        "plans": plans,
        "source_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        "latency_us": latencies,
        "samples_us": samples,
        "speedup": latencies["native"] / latencies["auto"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=256)
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()
    if min(args.groups, args.rounds, args.repeat, *args.queries) < 1:
        parser.error("groups, queries, rounds, and repeat must be positive")
    if torch.version.hip is not None or not torch.cuda.is_available():
        parser.error("this benchmark requires a CUDA GPU")
    if torch.cuda.get_device_capability() != (9, 0):
        parser.error("this reproducer targets SM90 (H100/H200); its legacy native width is architecture-specific")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tl.disable_cache()
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    print(f"GPU: {torch.cuda.get_device_name()}; warm-cache CUDA Graph, interleaved variants, {args.rounds} rounds", flush=True)
    results = []
    for queries in args.queries:
        result = benchmark(args.groups, queries, args.rounds, args.repeat, target)
        results.append(result)
        native, automatic = result["latency_us"]["native"], result["latency_us"]["auto"]
        print(f"queries={queries:3d}: old native={native:8.3f} us; new auto={automatic:8.3f} us; {result['speedup']:.2f}x", flush=True)
    if args.json_path is not None:
        report = {
            "gpu": torch.cuda.get_device_name(),
            "target": str(target),
            "rounds": args.rounds,
            "repeat": args.repeat,
            "timing": "warm-cache CUDA Graph, interleaved variants",
            "results": results,
        }
        args.json_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
