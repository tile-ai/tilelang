"""C/MIR comparison on real kernels; fresh processes, isolated caches, no speed gate."""

import argparse
from contextlib import suppress
import importlib
import json
import os
from pathlib import Path
import platform
import runpy
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def run_logged(command, env, path, label, timeout):
    """Stream worker output while bounding even a silent or stuck compiler."""
    print(f"START {label} (timeout={timeout:g}s)", flush=True)
    start = time.perf_counter()
    expired = threading.Event()
    result, ready = None, None
    with path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
        )

        def kill():
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()

        def expire():
            expired.set()
            print(f"TIMEOUT {label} after {timeout:g}s", flush=True)
            kill()

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.strip() == "HOST_IR_READY":
                    ready = time.perf_counter() - start
                if line.startswith("HOST_IR_RESULT "):
                    result = json.loads(line.removeprefix("HOST_IR_RESULT "))
            code = proc.wait()
        finally:
            timer.cancel()
            timer.join()
            if proc.poll() is None:
                kill()
            proc.wait()
            proc.stdout.close()
        if expired.is_set():
            log.write(f"TIMEOUT {label} after {timeout:g}s\n")
            code = -1
    failed = bool(code or result is None or ready is None)
    print(f"{'FAILED' if failed else 'DONE'} {label} ({time.perf_counter() - start:.1f}s)", flush=True)
    return failed, result, ready


def workload(target, case, torch):
    import tilelang.language as T

    device = "npu" if target in ("ascend", "pto") else "mps" if target == "metal" else "cuda"
    if case != "gemm":
        rows = 1 if case == "small" else 8
        if target in ("ascend", "pto"):
            program = runpy.run_path(str(ROOT / "testing/ascend/runtime/test_tvm_ffi_host_args.py"))["dynamic_grid_kernel"]
            dtype = torch.bfloat16
        else:
            n = 1 if case == "small" else T.dynamic("rows")

            @T.prim_func
            def program(A: T.Tensor((n, 128), T.float32), B: T.Tensor((n, 128), T.float32)):
                with T.Kernel(n, threads=128) as row:
                    for i in T.Parallel(128):
                        B[row, i] = A[row, i]

            dtype = torch.float32
        a = torch.randn(rows, 128, device=device, dtype=dtype)
        return program, [a, torch.empty_like(a)], a
    if target in ("ascend", "pto"):
        program = runpy.run_path(str(ROOT / "examples/ascend/example_gemm.py"))["gemm"](256, 256, 256)
        a, b = [torch.randn(256, 256, device=device, dtype=torch.bfloat16) for _ in range(2)]
        reference = a.float() @ b.float().T
    else:
        if target == "metal":
            factory = runpy.run_path(str(ROOT / "testing/python/metal/test_metal_cache.py"))["_matmul_gemm"]
            program = factory(128, 128, 128, 16, 16, 8)
        else:

            @T.prim_func
            def program(A: T.Tensor((128, 128), T.float16), B: T.Tensor((128, 128), T.float16), C: T.Tensor((128, 128), T.float32)):
                with T.Kernel(4, 4, threads=128) as (bx, by):
                    a_shared = T.alloc_shared((32, 32), T.float16)
                    b_shared = T.alloc_shared((32, 32), T.float16)
                    accum = T.alloc_fragment((32, 32), T.float32)
                    T.clear(accum)
                    for k in T.serial(4):
                        T.copy(A[by * 32, k * 32], a_shared)
                        T.copy(B[k * 32, bx * 32], b_shared)
                        T.gemm(a_shared, b_shared, accum)
                    T.copy(accum, C[by * 32, bx * 32])

        a, b = [torch.randn(128, 128, device=device, dtype=torch.float16) for _ in range(2)]
        reference = a.float() @ b.float()
    return program, [a, b, torch.empty_like(reference)], reference


def worker(args):
    import torch
    import tilelang
    from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache

    if args.target in ("ascend", "pto"):
        import torch_npu  # noqa: F401
    device = torch.npu if args.target in ("ascend", "pto") else torch.mps if args.target == "metal" else torch.cuda
    if device is torch.cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    program, inputs, reference = workload(args.target, args.case, torch)
    device.synchronize()
    lower = importlib.import_module("tilelang.engine.lower")
    timings = {"host_codegen_s": 0.0, "host_publish_s": 0.0}

    def timed(name, fn):
        def call(*a, **kw):
            start = time.perf_counter()
            result = fn(*a, **kw)
            timings[name] += time.perf_counter() - start
            return result

        return call

    # Includes Host IR lowering/codegen and host compilation/packaging, not device compilation.
    with (
        patch.object(lower, "host_codegen", timed("host_codegen_s", lower.host_codegen)),
        patch.object(TVMFFIKernelCache, "export_library", staticmethod(timed("host_publish_s", TVMFFIKernelCache.export_library))),
    ):
        start = time.perf_counter()
        kernel = tilelang.compile(
            program,
            target="cuda" if args.target == "nvrtc" else args.target,
            target_host=args.host,
            execution_backend="tvm_ffi",
            pass_configs={"tl.cuda_compiler": "nvrtc"} if args.target == "nvrtc" else {},
        )
        timings["compile_or_load_s"] = time.perf_counter() - start
        kernel(*inputs)
        device.synchronize()
    print("HOST_IR_READY", flush=True)
    torch.testing.assert_close(inputs[-1], reference, atol=1e-2 if args.case == "gemm" else 0, rtol=1e-2 if args.case == "gemm" else 0)
    if args.phase == "hit":
        assert timings["host_codegen_s"] == 0 and timings["host_publish_s"] == 0, "Expected disk cache hit, got compilation"
    else:
        assert timings["host_codegen_s"] > 0 and timings["host_publish_s"] > 0, "Cold compile did not publish a cache"
    for _ in range(20):
        kernel(*inputs)
    submit, complete = [], []
    for _ in range(7):
        device.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            kernel(*inputs)
        submitted = time.perf_counter()
        device.synchronize()
        complete.append((time.perf_counter() - start) * 1e6 / 100)
        submit.append((submitted - start) * 1e6 / 100)
    torch.testing.assert_close(inputs[-1], reference, atol=1e-2 if args.case == "gemm" else 0, rtol=1e-2 if args.case == "gemm" else 0)
    files = list(Path(os.environ["TILELANG_CACHE_DIR"]).rglob("*"))
    result = dict(
        timings,
        submit_us=statistics.median(submit),
        complete_us=statistics.median(complete),
        submit_samples_us=submit,
        complete_samples_us=complete,
        cache_bytes=sum(p.stat().st_size for p in files if p.is_file()),
        executable_bytes=sum(p.stat().st_size for p in files if p.name in ("executable.so", "executable.tlm")),
        torch=torch.__version__,
        tilelang=tilelang.__version__,
        platform=platform.platform(),
    )
    print("HOST_IR_RESULT " + json.dumps(result), flush=True)


def report(args):
    rows, errors = [], []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        for case in ("small", "dynamic", "gemm"):
            for trial in range(args.trials):
                for host in ("c", "mir") if trial % 2 == 0 else ("mir", "c"):
                    with tempfile.TemporaryDirectory(prefix="tilelang-host-bench-") as cache:
                        for phase in ("cold", "hit"):
                            env = dict(os.environ, TILELANG_CACHE_DIR=cache, TILELANG_DISABLE_CACHE="0", PYTHONUNBUFFERED="1")
                            command = [
                                sys.executable,
                                "-u",
                                str(Path(__file__).resolve()),
                                "--target",
                                target,
                                "--host",
                                host,
                                "--case",
                                case,
                                "--phase",
                                phase,
                            ]
                            label = f"{target}-{case}-{host}-{trial}-{phase}"
                            failed, result, ready = run_logged(command, env, output / f"{label}.log", label, args.timeout)
                            if failed:
                                errors.append(label)
                                break
                            rows.append(
                                dict(target=target, case=case, host=host, trial=trial, phase=phase, process_to_first_s=ready, **result)
                            )
                            (output / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = [
        "## Host IR: C vs MIR",
        "",
        "Median [min, max]; C/MIR > 1 favors MIR. No speed gate.",
        "Fresh processes and isolated caches; steady calls include real device kernels.",
        "Host codegen excludes external C compilation; publish includes compilation and device-binary packaging.",
        "Executable size includes embedded device modules, not host-only machine code.",
        "",
        "| Target / case / phase | Metric | C | MIR | C/MIR |",
        "|---|---|---:|---:|---:|",
    ]
    for target in args.targets:
        for case in ("small", "dynamic", "gemm"):
            for phase in ("cold", "hit"):
                for metric in (
                    "process_to_first_s",
                    "compile_or_load_s",
                    "host_codegen_s",
                    "host_publish_s",
                    "submit_us",
                    "complete_us",
                    "executable_bytes",
                    "cache_bytes",
                ):
                    groups = [
                        [r[metric] for r in rows if (r["target"], r["case"], r["phase"], r["host"]) == (target, case, phase, host)]
                        for host in ("c", "mir")
                    ]
                    if not all(len(g) == args.trials for g in groups) or not all(statistics.median(g) > 0 for g in groups):
                        continue
                    cells = [f"{statistics.median(g):.4g} [{min(g):.4g}, {max(g):.4g}]" for g in groups]
                    ratio = statistics.median(groups[0]) / statistics.median(groups[1])
                    summary.append(f"| {target}/{case}/{phase} | {metric} | {' | '.join(cells)} | {ratio:.3f}× |")
    summary += ["", "Failures: " + (", ".join(errors) if errors else "none")]
    text = "\n".join(summary) + "\n"
    (output / "summary.md").write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return bool(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=["cuda", "nvrtc", "cutedsl", "hip", "metal", "ascend", "pto"], default=["cuda"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", default="host-ir-report")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds per worker (default: 300)")
    parser.add_argument("--target")
    parser.add_argument("--host", choices=["c", "mir"])
    parser.add_argument("--case", choices=["small", "dynamic", "gemm"])
    parser.add_argument("--phase", choices=["cold", "hit"])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    if args.target:
        worker(args)
    else:
        sys.exit(report(args))
