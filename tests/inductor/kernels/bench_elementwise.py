"""Binary elementwise benchmark through the TileLang graph backend.

Measures ``x op y`` on last-axis 2D tensors across three backends:

  - eager     : PyTorch's native op
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Ops:  sub, mul, div, max, min
Shapes:  (1024, 4096), (1024, 10240), (1024, 20480)
Dtypes:  float16, bfloat16

Type-promotion note
-------------------
Only the TileLang model wraps the inputs in an explicit fp32 cast::

    op(x.to(fp32), y.to(fp32)).to(orig_dtype)

Cast is itself an elementwise op and gets fused into the same kernel
by the ElementWise schedule — no dedicated pattern to register.  The
explicit cast is a *source-level* convention so that TileLang matches
Inductor's numerical behaviour (Inductor auto-promotes fp16/bf16 math
to fp32 in Triton; our lowering does not).
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

SHAPES = [
    (1024, 4096),
    (1024, 10240),
    (1024, 20480),
]

DTYPES = [torch.float16, torch.bfloat16]

OPS = [torch.sub, torch.mul, torch.div, torch.max, torch.min]

_DTYPE_LABEL = {torch.float16: "fp16", torch.bfloat16: "bf16"}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class _NativeBinary(torch.nn.Module):
    """Plain op — used for eager and inductor.

    No explicit fp32 promotion: inductor auto-promotes for math
    functions; eager stays in the input dtype.
    """

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, y):
        return self.op(x, y)


class _TLBinary(torch.nn.Module):
    """TileLang variant with an explicit fp32 round-trip.

    The ``.to(fp32)`` and ``.to(orig_dtype)`` casts are themselves
    elementwise ops; the ElementWise schedule fuses all three
    (cast → op → cast) into a single kernel.  No pattern rewrite
    required — this is purely a source-code shape for parity with
    inductor's auto-promotion.
    """

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, y):
        return self.op(x.to(torch.float32), y.to(torch.float32)).to(x.dtype)


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


def _get_model(op, backend):
    cls = _TLBinary if backend == "tilelang" else _NativeBinary
    return cls(op).cuda().eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "kernel",
    "op",
    "shape",
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
    ap.add_argument("--ops", default=None, help="Comma-separated subset of ops (sub,mul,div,max,min).")
    args = ap.parse_args()

    if args.ops:
        wanted = {name.strip() for name in args.ops.split(",") if name.strip()}
        ops = [op for op in OPS if op.__name__ in wanted]
    else:
        ops = OPS

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    hdr = (
        f"{'op':<6s} {'shape':<18s} {'dtype':<6s} "
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
        for op, shape, dtype in product(ops, SHAPES, DTYPES):
            torch.manual_seed(0)
            x = torch.randn(*shape, dtype=dtype, device=dev)
            y = torch.randn(*shape, dtype=dtype, device=dev)

            # Eager reference — run the native (no-cast) model for a
            # stable dtype-preserving ground truth.
            ref_model = _NativeBinary(op).cuda().eval()
            with torch.no_grad():
                ref = ref_model(x, y).clone()

            results: dict[str, float] = {}
            err: dict[str, float] = {}
            probe.reset()
            for backend in backends:
                model = _get_model(op, backend)
                runner = _build_runner(model, backend)
                with torch.no_grad():
                    runner(x, y)  # compile + cache warm
                    out = runner(x, y)
                torch.cuda.synchronize()

                if backend != "eager":
                    out_f = out.float()
                    ref_f = ref.float()
                    mask = out_f.isfinite() & ref_f.isfinite()
                    if mask.any():
                        diff = (out_f[mask] - ref_f[mask]).abs().max().item()
                    else:
                        diff = 0.0
                    err[backend] = diff
                else:
                    err[backend] = 0.0
                us = _bench(lambda r=runner, x=x, y=y: r(x, y))
                results[backend] = us

            if probe.backend_calls == 0 or probe.tir_kernels == 0:
                raise AssertionError(
                    f"[{op.__name__} {shape} {_DTYPE_LABEL[dtype]}] "
                    "TileLang backend was not exercised "
                    f"(calls={probe.backend_calls}, kernels={probe.tir_kernels})"
                )

            label = _DTYPE_LABEL[dtype]
            print(
                f"{op.__name__:<6s} {str(tuple(shape)):<18s} {label:<6s} "
                f"{results['eager']:>10.2f}  "
                f"{results['inductor']:>10.2f}  "
                f"{results['tilelang']:>10.2f}  "
                f"{results['tilelang'] / results['eager']:>10.3f}  "
                f"{results['tilelang'] / results['inductor']:>10.3f}  "
                f"{max(err['inductor'], err['tilelang']):>10.4f}"
            )
            csv_rows.append(
                {
                    "kernel": "binary_elementwise",
                    "op": op.__name__,
                    "shape": str(tuple(shape)),
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
