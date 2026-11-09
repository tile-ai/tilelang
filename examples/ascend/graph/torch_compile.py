"""Run Torch compilation and separate retiling from graph fusion performance.

Example:
    python examples/ascend/graph/torch_compile.py --bench --output /tmp/ascend-graph.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

import torch

import tilelang
from tilelang.contrib import bisheng
from tilelang.graph import GraphConfig, make_backend
from tilelang.graph.frontend import fx_to_relax
from tilelang.graph.fusion import make_fusion_candidate, validate_runner
from tilelang.graph.runtime import build_runner
from tilelang.profiler import do_bench


def model(x, bias):
    return torch.relu(x + bias) * bias


def gemm_epilogue(x, weight, bias):
    return torch.relu(x @ weight + bias)


def kernel_metadata(kernel):
    """Record the compiler's resolved choice, including loaded cache entries."""
    context = kernel.backend_context
    return {
        "target": str(context.target),
        "target_backend": context.module.name,
        "execution_backend": context.execution_backend.name,
        "adapter": type(kernel.adapter).__name__,
    }


def summarize_measurements(result):
    """Keep noisy or missing evidence separate from a valid comparison."""
    for variant in result["variants"].values():
        variant["timing_validity"] = {}
        for method in ("event", "wall"):
            samples = variant[method + "_ms"]
            valid = len(samples) >= 3 and all(math.isfinite(value) and value > 0 for value in samples)
            median = statistics.median(samples) if valid else None
            spread = (max(samples) - min(samples)) / median if valid else None
            valid = valid and spread <= 0.02
            variant["median_" + method + "_ms"] = median
            variant["timing_validity"][method] = {
                "valid": valid,
                "relative_range": spread,
                "reason": "valid" if valid else "missing, non-finite, or more than 2% variability",
            }
    rows = result["variants"]
    isolated = result["performance_attribution"]["fusion_isolated"]
    comparison = "fusion_speedup_B_over_C" if isolated else "combined_epilogue_speedup_B_over_C"
    for name, left, right in (
        ("retiling_speedup_A_over_B", "A_default_unfused", "B_candidate_tile_unfused"),
        (comparison, "B_candidate_tile_unfused", "C_candidate_tile_fused"),
    ):
        valid = rows[left]["timing_validity"]["event"]["valid"] and rows[right]["timing_validity"]["event"]["valid"]
        result[name] = rows[left]["median_event_ms"] / rows[right]["median_event_ms"] if valid else None
    if not isolated:
        result["fusion_speedup_B_over_C"] = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--workload", choices=("elementwise", "gemm_epilogue"), default="elementwise")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--target",
        default=GraphConfig().target,
        help="Ascend target alias or JSON object; use pto for PTO code generation",
    )
    parser.add_argument(
        "--execution-backend",
        default="auto",
        choices=("auto", "tvm_ffi", "pto"),
        help="Execution adapter; auto resolves to tvm_ffi for AscendC and pto for PTO",
    )
    parser.add_argument("--default-tile", type=int, default=1024)
    parser.add_argument("--candidate-tile", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.k, args.repeats) <= 0:
        parser.error("rows, cols, k, and repeats must be positive")
    if args.bench and args.repeats < 3:
        parser.error("benchmarking requires at least three complete rounds")
    if args.workload == "gemm_epilogue" and args.dtype == "float32":
        parser.error("the initial GEMM schedule requires float16 or bfloat16 inputs")
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    selected_model = model if args.workload == "elementwise" else gemm_epilogue
    inputs = [torch.randn(args.rows, args.cols, device=device, dtype=dtype), torch.randn(args.cols, device=device, dtype=dtype)]
    if args.workload == "gemm_epilogue":
        inputs = [
            torch.randn(args.rows, args.k, device=device, dtype=dtype),
            torch.randn(args.k, args.cols, device=device, dtype=dtype),
            torch.randn(args.cols, device=device, dtype=dtype),
        ]
    mod = fx_to_relax(torch.fx.symbolic_trace(selected_model), inputs)
    execution_options = {"target": args.target, "execution_backend": args.execution_backend, "strict": True}
    default_config = GraphConfig(tile_size=args.default_tile, **execution_options)
    candidate_config = GraphConfig(tile_size=args.candidate_tile, **execution_options)
    variants = {
        "A_default_unfused": build_runner(mod, default_config),
        "B_candidate_tile_unfused": build_runner(mod, candidate_config),
        "C_candidate_tile_fused": build_runner(make_fusion_candidate(mod), candidate_config),
    }
    reports = []
    backend = make_backend(default_config)

    def recording_backend(gm, example_inputs):
        runner = backend(gm, example_inputs)
        reports.append(runner.graph_report)
        return runner

    compiled = torch.compile(selected_model, backend=recording_backend, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(*inputs), selected_model(*inputs))
        if not reports or not any(partition.compiled for report in reports for partition in report.partitions):
            raise RuntimeError("The Dynamo graph did not execute an Ascend kernel")
        for runner in variants.values():
            validate_runner(runner, selected_model, inputs)

    result = {
        "shape": [args.rows, args.cols],
        "workload": args.workload,
        "reduction_k": args.k if args.workload == "gemm_epilogue" else None,
        "dtype": args.dtype,
        "device": str(device),
        "target": default_config.target,
        "execution_backend": default_config.execution_backend,
        "device_name": torch.npu.get_device_name(device),
        "torch_version": torch.__version__,
        "tilelang_version": tilelang.__version__,
        "bisheng_version": bisheng.get_bisheng_version(),
        "seed": args.seed,
        "default_tile": args.default_tile,
        "candidate_tile": args.candidate_tile,
        "performance_attribution": {
            "fusion_isolated": args.workload == "elementwise",
            "reason": (
                "B and C use the same one-dimensional output tiles and core mapping"
                if args.workload == "elementwise"
                else "The fused GEMM epilogue uses Cube output tiles; B/C includes epilogue scheduling changes"
            ),
        },
        "torch_compile_correct": True,
        "dynamo_partitions": [asdict(partition) for report in reports for partition in report.partitions],
        "measurement": {
            "rounds": args.repeats,
            "warmup_ms": 10,
            "repeat_budget_ms": 30,
            "event_cache_flush_mb": 256,
            "wall_cache_flush": False,
            "allocations_and_dispatch_included": True,
        },
        "variants": {},
    }
    for name, runner in variants.items():
        kernels = {symbol: kernel_metadata(kernel) for symbol, kernel in runner.kernels.items()}
        result["variants"][name] = {
            "correct": True,
            "kernel_calls": runner.kernel_count,
            "kernels": kernels,
            "event_ms": [],
            "wall_ms": [],
        }
        if args.source_dir:
            directory = args.source_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "graph.py").write_text(runner.mod.script())
            for symbol, kernel in runner.kernels.items():
                extension = ".ptodsl.py" if kernels[symbol]["target_backend"] == "pto" else ".asc"
                (directory / f"{symbol}{extension}").write_text(kernel.get_kernel_source())
    if args.bench:
        measured = {**variants, "Torch_NPU_eager": selected_model}
        result["variants"]["Torch_NPU_eager"] = {"correct": True, "event_ms": [], "wall_ms": []}
        with torch.no_grad():
            for repeat in range(args.repeats):
                order = list(measured.items())
                if repeat % 2:
                    order.reverse()
                for name, runner in order:
                    for method in ("event", "wall"):
                        latency = do_bench(
                            lambda runner=runner: runner(*inputs),
                            device=device,
                            backend=method,
                            warmup=10,
                            rep=30,
                            return_mode="median",
                            early_stop_baseline=None,
                        )
                        result["variants"][name][method + "_ms"].append(latency)
        summarize_measurements(result)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
