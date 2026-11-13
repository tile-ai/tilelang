"""Small-M Metal GEMM block sweep; reports synchronized batch latency, not GPU-only time."""

import argparse
from contextlib import suppress
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import tilelang
import tilelang.language as T


def make_kernel(m, n, k, bm, bn, bk, transpose_b=False):
    b_shape = (n, k) if transpose_b else (k, n)
    shared_b_shape = (bn, bk) if transpose_b else (bk, bn)

    @T.prim_func
    def kernel(A: T.Tensor((m, k), T.float16), B: T.Tensor(b_shape, T.float16), C: T.Tensor((m, n), T.float32)):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=128) as (bx, by):
            a = T.alloc_shared((bm, bk), T.float16)
            b = T.alloc_shared(shared_b_shape, T.float16)
            acc = T.alloc_local((bm, bn), T.float32, scope="metal.simdgroup")
            scratch = T.alloc_shared((bm, bn), T.float32)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=0):
                T.copy(A[by * bm, ko * bk], a)
                if transpose_b:
                    T.copy(B[bx * bn, ko * bk], b)
                else:
                    T.copy(B[ko * bk, bx * bn], b)
                T.gemm(a, b, acc, transpose_B=transpose_b)
            T.copy(acc, scratch)
            T.copy(scratch, C[by * bm, bx * bn])

    return kernel


def measure(fn, count):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    for _ in range(count):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter_ns() - start) / count / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nk", type=int, default=1024)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--transpose-b", action="store_true", help="Store weights as contiguous (N,K).")
    parser.add_argument("--focused", action="store_true", help="Compare only 32x32x32 and 8x32x32.")
    parser.add_argument("--torch-reference", action="store_true", help="Include FP32 torch.mm with inputs converted before timing.")
    args = parser.parse_args()
    n = args.n if args.n is not None else args.nk
    k = args.k if args.k is not None else args.nk
    if min(args.rows + [n, k, args.rounds, args.iterations]) <= 0:
        parser.error("dimensions, rounds, and iterations must be positive")
    if not torch.backends.mps.is_available():
        parser.error("this benchmark requires PyTorch MPS on Apple Silicon")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(4028)
    rng = random.Random(4028)
    root = Path(tilelang.__file__).resolve().parent.parent
    head = None
    if (root / ".git").exists():
        with suppress(OSError, subprocess.CalledProcessError):
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    report = dict(
        head=head,
        platform=platform.platform(),
        torch=torch.__version__,
        source=str(root),
        metric="synchronized batch wall time per invocation (us), includes dispatch",
        iterations=args.iterations,
        rounds=args.rounds,
        results=[],
    )
    for m in args.rows:
        a = torch.randn((m, k), dtype=torch.float16, device="mps")
        b = torch.randn((n, k) if args.transpose_b else (k, n), dtype=torch.float16, device="mps")
        expected = a.cpu().float() @ (b.cpu().float().T if args.transpose_b else b.cpu().float())
        fns, outputs, samples, compile_ms = {}, {}, {}, {}
        configs = [(32, 32, 32), (8, 32, 32)]
        if not args.focused:
            configs += [(8, 64, 32), (8, 64, 64), (16, 64, 64)]
        for bm, bn, bk in configs:
            name = f"{bm}x{bn}x{bk}"
            start = time.perf_counter()
            kernel = tilelang.compile(
                make_kernel(m, n, k, bm, bn, bk, args.transpose_b), out_idx=[], target="metal", target_host="c", execution_backend="tvm_ffi"
            )
            compile_ms[name] = (time.perf_counter() - start) * 1000
            backing = torch.full((m * n + 64,), -123.0, device="mps")
            c = backing[: m * n].view(m, n)

            def fn(kernel=kernel, c=c, a=a, b=b):
                return kernel(a, b, c)

            fn()
            torch.mps.synchronize()
            torch.testing.assert_close(c.cpu(), expected, rtol=2e-3, atol=2e-3)
            assert torch.all(backing[m * n :].cpu() == -123)
            fns[name], outputs[name], samples[name] = fn, backing, []
            for _ in range(10):
                fn()
        if args.torch_reference:
            af, bf = a.float(), b.float()
            bf = bf.T if args.transpose_b else bf
            ref_backing = torch.full((m * n + 64,), -123.0, device="mps")
            ref_output = ref_backing[: m * n].view(m, n)

            def torch_fn(af=af, bf=bf, ref_output=ref_output):
                return torch.mm(af, bf, out=ref_output)

            torch_fn()
            torch.mps.synchronize()
            torch.testing.assert_close(ref_output.cpu(), expected, rtol=2e-3, atol=2e-3)
            fns["torch_fp32_precast"] = torch_fn
            outputs["torch_fp32_precast"] = ref_backing
            samples["torch_fp32_precast"] = []
        for fn in fns.values():
            measure(fn, args.iterations)
        for _ in range(args.rounds):
            order = list(fns)
            rng.shuffle(order)
            for name in order:
                samples[name].append(measure(fns[name], args.iterations))
        for backing in outputs.values():
            torch.testing.assert_close(backing[: m * n].view(m, n).cpu(), expected, rtol=2e-3, atol=2e-3)
            assert torch.all(backing[m * n :].cpu() == -123)
        row = dict(
            shape=[m, n, k],
            transpose_b=args.transpose_b,
            samples_us=samples,
            median_us={name: statistics.median(values) for name, values in samples.items()},
            compile_ms=compile_ms,
        )
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(shape=row["shape"], median_us=row["median_us"])), flush=True)


if __name__ == "__main__":
    main()
