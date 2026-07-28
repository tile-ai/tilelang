"""Benchmark torch.mean (last-axis reduction) through the TileLang graph backend.

Measures ``torch.mean(x, dim=-1)`` across three backends:

  - eager     : PyTorch's native ``torch.mean``
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Workloads (from the MeanFwdOp spec):

  - hidden-state-reduce   (2048, 4096)   float16 / bfloat16
  - long-seq-reduce       (64,   32768)  bfloat16

The ``Mean`` module is plain ``torch.mean(x, dim=-1)`` — the fp32
accumulator is applied internally by the emitted kernel via the
``mean_fp32_accum`` pattern, not by user code.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import warnings

warnings.filterwarnings("ignore")
import argparse
import csv
import logging

logging.disable(logging.WARNING)
from pathlib import Path

import torch
import torch._dynamo

import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench
import tilelang.graph.vm_build as _vm_mod


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class Mean(torch.nn.Module):
    """Last-axis mean (``torch.mean(x, dim=-1)``, ``keepdim=False``)."""

    def forward(self, x):
        return torch.mean(x, dim=-1)


NativeMean = Mean


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench(fn):
    return do_bench(fn) * 1000.0  # ms → µs


def _build_runner(model, backend):
    if backend == "eager":
        return model
    if backend == "inductor":
        torch._dynamo.reset()
        return torch.compile(model, backend="inductor", mode="default")
    if backend == "tilelang":
        torch._dynamo.reset()
        clear_cache()
        backend_config.vm_clone_output = False
        return torch.compile(model, backend="tilelang")
    raise ValueError(backend)


def _get_model(model_native, model_tl, backend):
    return model_tl if backend == "tilelang" else model_native


# ---------------------------------------------------------------------------
# Optional TIR / CUDA dump for one config
# ---------------------------------------------------------------------------


def _dump_tilelang_source_once(model, x, label):
    captured_tir: dict = {}
    captured_cuda: str | None = None

    orig = _vm_mod._compile_kernels_tilelang

    def spy(kernel_mod, target):
        from tvm import tirx as _tir

        for gv in kernel_mod.get_global_vars():
            fn = kernel_mod[gv]
            if not isinstance(fn, _tir.PrimFunc):
                continue
            captured_tir[gv.name_hint] = str(fn)
        rt = orig(kernel_mod, target)
        nonlocal captured_cuda
        try:
            captured_cuda = rt.imported_modules[0].get_source()
        except Exception:
            captured_cuda = None
        return rt

    _vm_mod._compile_kernels_tilelang = spy
    try:
        torch._dynamo.reset()
        clear_cache()
        backend_config.vm_clone_output = False
        runner = torch.compile(model, backend="tilelang")
        with torch.no_grad():
            runner(x)
        torch.cuda.synchronize()
    finally:
        _vm_mod._compile_kernels_tilelang = orig

    print(f"\n{'=' * 80}")
    print(f"TileLang lowered output — Mean {label}")
    print(f"{'=' * 80}")

    if not captured_tir:
        print("(no PrimFunc captured — pattern did not fire)")
        return

    for name, body in captured_tir.items():
        print(f"\n--- TIR: {name} ---")
        print(body)

    if captured_cuda:
        print("\n--- CUDA source (head) ---")
        print(captured_cuda[:4000])


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

_CONFIGS = [
    ((2048, 4096), torch.float16, "hidden-state-reduce-fp16"),
    ((2048, 4096), torch.bfloat16, "hidden-state-reduce-bf16"),
    ((64, 32768), torch.bfloat16, "long-seq-reduce-bf16"),
]

_CSV_COLUMNS = [
    "kernel",
    "config",
    "eager_us",
    "inductor_us",
    "tilelang_us",
    "speedup_vs_eager",
    "speedup_vs_ind",
    "max_err",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true", help="Dump TileLang TIR and CUDA source for one config.")
    ap.add_argument("--dump-config", default="2048,4096:bf16", help="Config to dump (format shape_csv:dtype, e.g. 2048,4096:bf16).")
    ap.add_argument("--csv", type=Path, default=None, help="Append results to this CSV file.")
    args = ap.parse_args()

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    if args.dump:
        shape_str, dtype_str = args.dump_config.split(":")
        shape = tuple(int(x) for x in shape_str.split(","))
        dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        dtype = dtype_map[dtype_str]
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dtype, device=dev) * 0.02
        model = Mean().cuda().eval()
        _dump_tilelang_source_once(model, x, f"{shape} {dtype_str}")

    csv_rows: list[dict] = []

    hdr = f"{'config':<32s}  {'eager(µs)':>10s}  {'ind(µs)':>10s}  {'tl(µs)':>10s}  {'TL/Eager':>10s}  {'TL/Ind':>10s}  {'max_err':>10s}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for shape, dt, label in _CONFIGS:
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dt, device=dev) * 0.02

        model = Mean().cuda().eval()
        model_native = NativeMean().cuda().eval()

        with torch.no_grad():
            ref = model_native(x).clone()

        results: dict[str, float] = {}
        err: dict[str, float] = {}
        for backend in backends:
            model_b = _get_model(model_native, model, backend)
            runner = _build_runner(model_b, backend)
            with torch.no_grad():
                runner(x)
                out = runner(x)
            torch.cuda.synchronize()

            err[backend] = (out.float() - ref.float()).abs().max().item() if backend != "eager" else 0.0
            us = _bench(lambda r=runner, x=x: r(x))
            results[backend] = us

        print(
            f"{label:<32s}  "
            f"{results['eager']:>10.2f}  "
            f"{results['inductor']:>10.2f}  "
            f"{results['tilelang']:>10.2f}  "
            f"{results['tilelang'] / results['eager']:>10.3f}  "
            f"{results['tilelang'] / results['inductor']:>10.3f}  "
            f"{max(err['inductor'], err['tilelang']):>10.4f}"
        )
        csv_rows.append(
            {
                "kernel": "mean",
                "config": label,
                "eager_us": f"{results['eager']:.4f}",
                "inductor_us": f"{results['inductor']:.4f}",
                "tilelang_us": f"{results['tilelang']:.4f}",
                "speedup_vs_eager": f"{results['eager'] / results['tilelang']:.4f}",
                "speedup_vs_ind": f"{results['inductor'] / results['tilelang']:.4f}",
                "max_err": f"{max(err['inductor'], err['tilelang']):.4f}",
            }
        )

    if args.csv is not None and csv_rows:
        mode = "a" if args.csv.exists() else "w"
        with args.csv.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            if mode == "w":
                writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nAppended {len(csv_rows)} rows → {args.csv}")


if __name__ == "__main__":
    main()
