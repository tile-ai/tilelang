import argparse
import hashlib
import json
import itertools
import platform
import statistics
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
from tilelang.env import TILELANG_TEMPLATE_PATH


def launch(kernel, backend, *args):
    if backend == "nvrtc":
        kernel(*args, stream=torch.cuda.current_stream().cuda_stream)
    else:
        kernel(*args)


def make_kernel(n, steps, nan_clamp):
    @T.prim_func
    def kernel(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            i = block * 256 + T.get_thread_binding()
            if i < n:
                v = T.alloc_var("float32", init=X[i])
                for _ in T.serial(steps):
                    # Dependent bounds, like clipping an interval against another.
                    # All intermediates remain finite for the benchmark inputs.
                    if nan_clamp:
                        lo = T.clamp(v - 0.25, 0.0, 1.0)
                        hi = T.clamp(v + 0.25, lo, 1.0)
                        v = T.clamp(v * 1.01 + 0.001, lo, hi)
                    else:
                        lo = T.min(T.max(v - 0.25, 0.0), 1.0)
                        hi = T.min(T.max(v + 0.25, lo), 1.0)
                        v = T.min(T.max(v * 1.01 + 0.001, lo), hi)
                Y[i] = v

    return kernel


def make_semantic_kernel(n, constants):
    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), Lo: T.Tensor((n,), "float32"), Hi: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for i in T.Parallel(128):
                j = bx * 128 + i
                if j < n:
                    C[j] = T.clamp(A[j], -1.0 if constants else Lo[j], 1.0 if constants else Hi[j])

    return kernel


def semantic_outputs(backend):
    bits = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x3F000000,
        0xBF000000,
        0x3F800000,
        0xBF800000,
        0x40000000,
        0xC0000000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0xFFC01234,
        0x7F800001,
        0xFF800001,
    ]
    triples = torch.tensor(list(itertools.product(bits, repeat=3)), dtype=torch.int64).to(torch.int32).view(torch.float32)
    a, lo, hi = [column.contiguous().cuda() for column in triples.unbind(1)]
    n = a.numel()
    outputs = []
    for constants in [False, True]:
        kernel = make_semantic_kernel(n, constants)
        for fast in [False, True]:
            compiled = tilelang.compile(kernel, target="cuda", execution_backend=backend, compile_flags=["--use_fast_math"] if fast else [])
            out = torch.empty_like(a)
            launch(compiled, backend, a, lo, hi, out)
            outputs.append(out.cpu())
    return outputs


def main():
    parser = argparse.ArgumentParser(description="TileLang #3515 workload and exact FP32 clamp differential checks.")
    parser.add_argument("--n", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--fast-math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("clamp-poc"))
    parser.add_argument("--execution-backend", choices=["tvm_ffi", "nvrtc"], default="tvm_ffi")
    parser.add_argument("--compare", type=Path, help="Baseline output directory produced with the same backend/options")
    args = parser.parse_args()
    if args.n < 1 or args.steps < 1:
        parser.error("n and steps must be positive")
    if args.compare is not None and args.output.resolve() == args.compare.resolve():
        parser.error("--output and --compare must refer to different directories")
    cuda_math_sha256 = hashlib.sha256((Path(TILELANG_TEMPLATE_PATH) / "tl_templates/cuda/math.h").read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    values = semantic_outputs(args.execution_backend)
    torch.save(values, args.output / "semantics.pt")
    if args.compare:
        baseline = torch.load(args.compare / "semantics.pt", weights_only=True)
        assert len(baseline) == len(values)
        for before, after in zip(baseline, values, strict=True):
            assert torch.equal(torch.isnan(before), torch.isnan(after))
            valid = ~torch.isnan(before)
            assert torch.equal(before.view(torch.int32)[valid], after.view(torch.int32)[valid])
    torch.manual_seed(0)
    x = torch.rand(args.n, device="cuda", dtype=torch.float32)
    # Graph nodes retain kernel handles; keep the owning CUDA libraries alive.
    outputs, graphs, kernels, samples = {}, {}, {}, {"clamp": [], "minmax": []}
    capture_stream = torch.cuda.Stream()
    for name in samples:
        kernel = kernels[name] = tilelang.compile(
            make_kernel(args.n, args.steps, name == "clamp"),
            target="cuda",
            execution_backend=args.execution_backend,
            pass_configs={"tl.enable_fast_math": args.fast_math},
            compile_flags=["--use_fast_math"] if args.fast_math else [],
        )
        (args.output / f"{name}.cu").write_text(kernel.get_kernel_source())
        kernel.export_ptx(str(args.output / f"{name}.ptx"))
        y = outputs[name] = torch.empty_like(x)
        for _ in range(10):
            launch(kernel, args.execution_backend, x, y)
        torch.cuda.synchronize()
        graph = graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(100):
                launch(kernel, args.execution_backend, x, y)
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs["clamp"], outputs["minmax"], rtol=0, atol=0)
    # Each graph contains 100 launches, avoiding per-kernel Python dispatch gaps.
    # Alternate order; compilation and correctness checks are outside timing.
    for repeat in range(9):
        for name in ("clamp", "minmax") if repeat % 2 == 0 else ("minmax", "clamp"):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    result = {
        "python": platform.python_version(),
        "execution_backend": args.execution_backend,
        "semantic_cases": sum(value.numel() for value in values),
        "semantic_baseline_compared": bool(args.compare),
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "cuda_math_sha256": cuda_math_sha256,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "n": args.n,
        "steps": args.steps,
        "fast_math": args.fast_math,
        "samples_us": samples,
        "median_us": {k: statistics.median(v) for k, v in samples.items()},
    }
    result["clamp_over_minmax"] = result["median_us"]["clamp"] / result["median_us"]["minmax"]
    # Deliberately different NaN behavior: NOT a claim of general semantic equivalence.
    x[0] = float("nan")
    for graph in graphs.values():
        graph.replay()
    assert outputs["clamp"][0].isnan().item()
    assert outputs["minmax"][0].isfinite().item()
    result["nan_input_outputs"] = {k: str(v[0].item()) for k, v in outputs.items()}
    if args.compare:
        baseline = json.loads((args.compare / "result.json").read_text())
        for key in ["execution_backend", "gpu", "capability", "python", "torch", "torch_cuda", "n", "steps", "fast_math"]:
            assert baseline[key] == result[key], f"Mismatched benchmark setting: {key}"
        result["clamp_speedup_over_baseline"] = baseline["median_us"]["clamp"] / result["median_us"]["clamp"]
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
