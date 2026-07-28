"""Unary elementwise benchmark through the TileLang graph backend.

Measures ``op(x)`` on large 1-D tensors across three backends:

  - eager     : PyTorch's native op
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Ops:    exp, gelu
Shapes: 262_144, 1_048_576, 4_000_000
Dtypes: float16, bfloat16

Type-promotion note
-------------------
Only the TileLang model wraps the input in an explicit fp32 cast::

    op(x.to(fp32)).to(orig_dtype)

Cast is an elementwise op and gets fused into the same kernel by the
ElementWise schedule — no dedicated pattern to register.  The
explicit cast is a source-level convention so that TileLang matches
Inductor's fp32 auto-promotion for transcendental/math functions.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import warnings

warnings.filterwarnings("ignore")
import argparse
import csv
import logging

logging.disable(logging.WARNING)
from itertools import product
from pathlib import Path

import torch
import torch._dynamo
import torch.nn.functional as F

import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench
import tilelang.graph.backend as _be_mod
import tilelang.graph.vm_build as _vm_mod
from tvm import tirx as _tir


# ---------------------------------------------------------------------------
# Workload spec
# ---------------------------------------------------------------------------

SHAPES = [262_144, 1_048_576, 4_000_000]
DTYPES = [torch.float16, torch.bfloat16]
OPS = ["exp", "gelu"]

_DTYPE_LABEL = {torch.float16: "fp16", torch.bfloat16: "bf16"}


# ---------------------------------------------------------------------------
# Model definitions — native (plain) vs TL (explicit fp32 round-trip)
# ---------------------------------------------------------------------------


class _NativeExp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


class _NativeGelu(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)


class _TLExp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x.to(torch.float32)).to(x.dtype)


class _TLGelu(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x.to(torch.float32)).to(x.dtype)


_NATIVE_BY_OP = {"exp": _NativeExp, "gelu": _NativeGelu}
_TL_BY_OP = {"exp": _TLExp, "gelu": _TLGelu}


# ---------------------------------------------------------------------------
# Backend probe — fail fast if TileLang silently falls back to eager
# ---------------------------------------------------------------------------


class _BackendProbe:
    def __init__(self):
        self.backend_calls = 0
        self.tir_kernels = 0
        self._orig_backend = None
        self._orig_compile_tir = None

    def install(self):
        from torch._dynamo.backends.registry import _COMPILER_FNS

        self._orig_backend = _be_mod.tilelang_backend
        self._orig_compile_tir = _vm_mod._compile_tir_for_vm

        def _tracked_backend(gm, example_inputs):
            self.backend_calls += 1
            return self._orig_backend(gm, example_inputs)

        def _tracked_compile_tir(tir_mod, target):
            for _gv, fn in tir_mod.functions.items():
                if isinstance(fn, _tir.PrimFunc):
                    if fn.attrs is not None and fn.attrs.get("tir.is_host_func"):
                        continue
                    self.tir_kernels += 1
            return self._orig_compile_tir(tir_mod, target)

        _COMPILER_FNS["tilelang"] = _tracked_backend
        _be_mod.tilelang_backend = _tracked_backend
        _vm_mod._compile_tir_for_vm = _tracked_compile_tir

    def uninstall(self):
        from torch._dynamo.backends.registry import _COMPILER_FNS

        if self._orig_backend is not None:
            _COMPILER_FNS["tilelang"] = self._orig_backend
            _be_mod.tilelang_backend = self._orig_backend
        if self._orig_compile_tir is not None:
            _vm_mod._compile_tir_for_vm = self._orig_compile_tir

    def reset(self):
        self.backend_calls = 0
        self.tir_kernels = 0


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
    raise ValueError(f"unknown backend {backend!r}")


def _get_model(op_name, backend):
    table = _TL_BY_OP if backend == "tilelang" else _NATIVE_BY_OP
    return table[op_name]().cuda().eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "kernel",
    "op",
    "N",
    "dtype",
    "eager_us",
    "inductor_us",
    "tilelang_us",
    "speedup_vs_eager",
    "speedup_vs_ind",
    "max_err",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, default=None, help="Append results to this CSV file.")
    ap.add_argument("--ops", default=None, help="Comma-separated subset of ops (exp,gelu).")
    args = ap.parse_args()

    if args.ops:
        wanted = {name.strip() for name in args.ops.split(",") if name.strip()}
        ops = [name for name in OPS if name in wanted]
    else:
        ops = OPS

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    hdr = (
        f"{'op':<5s} {'N':>8s} {'dtype':<6s} "
        f"{'eager(µs)':>10s}  {'ind(µs)':>10s}  {'tl(µs)':>10s}  "
        f"{'TL/Eager':>10s}  {'TL/Ind':>10s}  {'max_err':>10s}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))

    csv_rows: list[dict] = []
    probe = _BackendProbe()
    probe.install()
    try:
        for op_name, N, dtype in product(ops, SHAPES, DTYPES):
            torch.manual_seed(0)
            x = torch.randn(N, dtype=dtype, device=dev)

            ref_model = _NATIVE_BY_OP[op_name]().cuda().eval()
            with torch.no_grad():
                ref = ref_model(x).clone()

            results: dict[str, float] = {}
            err: dict[str, float] = {}
            probe.reset()
            for backend in backends:
                model = _get_model(op_name, backend)
                runner = _build_runner(model, backend)
                with torch.no_grad():
                    runner(x)  # compile + cache warm
                    out = runner(x)
                torch.cuda.synchronize()

                out_f = out.float()
                ref_f = ref.float()
                mask = out_f.isfinite() & ref_f.isfinite()
                if mask.any():
                    diff = (out_f[mask] - ref_f[mask]).abs().max().item()
                else:
                    diff = 0.0
                err[backend] = diff if backend != "eager" else 0.0

                us = _bench(lambda r=runner, x=x: r(x))
                results[backend] = us

            if probe.backend_calls == 0 or probe.tir_kernels == 0:
                raise AssertionError(
                    f"[{op_name} N={N} {_DTYPE_LABEL[dtype]}] "
                    "TileLang backend was not exercised "
                    f"(calls={probe.backend_calls}, kernels={probe.tir_kernels})"
                )

            label = _DTYPE_LABEL[dtype]
            print(
                f"{op_name:<5s} {N:>8d} {label:<6s} "
                f"{results['eager']:>10.2f}  "
                f"{results['inductor']:>10.2f}  "
                f"{results['tilelang']:>10.2f}  "
                f"{results['tilelang'] / results['eager']:>10.3f}  "
                f"{results['tilelang'] / results['inductor']:>10.3f}  "
                f"{max(err['inductor'], err['tilelang']):>10.4f}"
            )
            csv_rows.append(
                {
                    "kernel": "unary_elementwise",
                    "op": op_name,
                    "N": str(N),
                    "dtype": label,
                    "eager_us": f"{results['eager']:.4f}",
                    "inductor_us": f"{results['inductor']:.4f}",
                    "tilelang_us": f"{results['tilelang']:.4f}",
                    "speedup_vs_eager": f"{results['eager'] / results['tilelang']:.4f}",
                    "speedup_vs_ind": f"{results['inductor'] / results['tilelang']:.4f}",
                    "max_err": f"{max(err['inductor'], err['tilelang']):.4f}",
                }
            )
    finally:
        probe.uninstall()

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
