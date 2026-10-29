"""Matched, preallocated, unsorted values+indices benchmark; times are in us."""

import argparse
from contextlib import suppress
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import tilelang

from example_deep_select import prepare_deep_select


def ensure_exclusive_gpu(gpu_uuid):
    """Do not silently report timings from a GPU another process is using."""
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    for row in rows.splitlines():
        uuid, pid, name = (field.strip() for field in row.split(",", 2))
        if uuid.removeprefix("GPU-") == gpu_uuid and int(pid) != os.getpid():
            raise RuntimeError(f"GPU is also used by PID {pid} ({name}); benchmark on an idle GPU")


def measure(fn, *, cold, samples, flush):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    repeats = 1 if cold else 100
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    # JIT compilation can leave an idle GPU at low clocks. Precondition both
    # implementations equally rather than attributing clock ramp-up to a kernel.
    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        flush.zero_()
        graph.replay()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    results = []
    for _ in range(samples):
        # This long, common prelude also keeps CPU launch gaps out of event timing.
        # It is outside the measured interval for BOTH implementations.
        if cold:
            flush.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "both"], default="both")
    parser.add_argument("--strategy", choices=["auto", "stream", "hierarchical"], default="auto")
    parser.add_argument(
        "--cases", default="1:8192:512,16:8192:512,64:8192:512,1:131072:512,6:131072:512,64:131072:512,1:1048576:512,6:1048576:512"
    )
    parser.add_argument("--cold-samples", type=int, default=100)
    parser.add_argument("--warm-samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, default=Path("deep_select_results.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    props = torch.cuda.get_device_properties(0)
    gpu_uuid = str(props.uuid)
    report = dict(
        gpu=props.name,
        gpu_uuid=gpu_uuid,
        compute_capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        tilelang=tilelang.__version__,
        tilelang_path=tilelang.__file__,
        seed=args.seed,
        flush_bytes=256 * 1024**2,
        baseline="torch.topk(largest=True, sorted=False, out=preallocated), values + int64 indices",
        timing="CUDA events around CUDA graph replay; warm: 100 calls/replay; cold: 1 call/replay after 256 MiB flush",
        cold_samples=args.cold_samples,
        warm_samples=args.warm_samples,
        precondition_ms=200,
        kernel_source_sha256=hashlib.sha256(Path(__file__).with_name("example_deep_select.py").read_bytes()).hexdigest(),
        results=[],
    )
    from tilelang.contrib.nvcc import get_nvcc_compiler

    report["nvcc"] = subprocess.check_output([get_nvcc_compiler(), "--version"], text=True).strip()
    with suppress(OSError, subprocess.CalledProcessError):
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
        ).strip()
    flush = torch.empty(report["flush_bytes"], dtype=torch.uint8, device="cuda")
    dtypes = [torch.bfloat16, torch.float32] if args.dtype == "both" else [getattr(torch, args.dtype)]
    for dtype in dtypes:
        for case in args.cases.split(","):
            ensure_exclusive_gpu(gpu_uuid)
            m, n, k = map(int, case.split(":"))
            x = torch.randn(m, n, device="cuda", dtype=dtype)
            run = prepare_deep_select(x, k, strategy=args.strategy, index_dtype=torch.int64)
            values, indices = run()
            torch.testing.assert_close(values.sort().values, x.topk(k).values.sort().values, rtol=0, atol=0)
            torch.testing.assert_close(values, x.gather(1, indices), rtol=0, atol=0)
            ordered = indices.sort().values
            assert (ordered[:, 1:] != ordered[:, :-1]).all()
            ref_values = torch.empty((m, k), device="cuda", dtype=dtype)
            ref_indices = torch.empty((m, k), device="cuda", dtype=torch.int64)

            baseline = partial(torch.topk, x, k, sorted=False, out=(ref_values, ref_indices))

            row = dict(dtype=str(dtype), batch=m, n=n, k=k, **run.configuration)
            for cold, samples in [(False, args.warm_samples), (True, args.cold_samples)]:
                label = "cold" if cold else "warm"
                tl_us = measure(run, cold=cold, samples=samples, flush=flush)
                pt_us = measure(baseline, cold=cold, samples=samples, flush=flush)
                row[label] = dict(tilelang_us=tl_us, torch_us=pt_us, speedup=pt_us / tl_us)
            ensure_exclusive_gpu(gpu_uuid)
            report["results"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
