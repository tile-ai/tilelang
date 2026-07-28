"""SiLU-and-multiply benchmark through the TileLang graph backend.

Computes the SwiGLU activation tail:

    y = silu(gate) * value
      = (gate * sigmoid(gate)) * value

Two equally-shaped inputs (``gate`` and ``value``) go through:

  - eager     : PyTorch's native ops
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Workloads
---------
    shape               dtypes
    ------------------  -------------------------
    (1024,  4096)       fp16, bf16, fp32
    (1024,  10240)      fp16, bf16, fp32
    (4096,  4096)       fp16, bf16, fp32

Type-promotion note
-------------------
Only the TileLang model wraps the inputs in an explicit fp32 cast::

    (F.silu(gate.to(fp32)) * value.to(fp32)).to(orig_dtype)

Cast is itself an elementwise op and gets fused into the same kernel
by the ElementWise schedule — no dedicated pattern to register.  The
explicit fp32 round-trip is a source-level convention so that
TileLang matches Inductor's fp32 auto-promotion for transcendental
math (sigmoid/exp).  fp32 inputs make both paths identical.
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
# Workload spec (matches _STRATEGY_SHAPES / _STRATEGY_DTYPES)
# ---------------------------------------------------------------------------

SHAPES = [
    (1024, 4096),
    (1024, 10240),
    (4096, 4096),
]

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_DTYPE_LABEL = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


# ---------------------------------------------------------------------------
# Model definitions — native (plain) vs TL (explicit fp32 round-trip)
# ---------------------------------------------------------------------------


class _NativeSiluAndMul(torch.nn.Module):
    """Plain path used for eager and inductor."""

    def forward(self, gate, value):
        return F.silu(gate) * value


class _TLSiluAndMul(torch.nn.Module):
    """TileLang variant with an explicit fp32 round-trip.

    Cast is an elementwise op, so the emitted kernel fuses::

        cast(gate, fp32) → sigmoid → mul → mul(value) → cast(orig)

    into a single ElementWise-scheduled kernel.  No dedicated pattern
    rewrite required.
    """

    def forward(self, gate, value):
        orig_dtype = gate.dtype
        out = F.silu(gate.to(torch.float32)) * value.to(torch.float32)
        return out.to(orig_dtype)


# _TLSiluAndMul = _NativeSiluAndMul

# ---------------------------------------------------------------------------
# Backend probe — fail fast on silent fallback to eager
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


def _get_model(backend):
    cls = _TLSiluAndMul if backend == "tilelang" else _NativeSiluAndMul
    return cls().cuda().eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "kernel",
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
    args = ap.parse_args()

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    hdr = (
        f"{'shape':<18s} {'dtype':<6s} "
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
        for shape, dtype in product(SHAPES, DTYPES):
            torch.manual_seed(0)
            gate = torch.randn(*shape, dtype=dtype, device=dev)
            value = torch.randn(*shape, dtype=dtype, device=dev)

            # Eager reference — plain path, matches each backend's dtype
            ref_model = _NativeSiluAndMul().cuda().eval()
            with torch.no_grad():
                ref = ref_model(gate, value).clone()

            results: dict[str, float] = {}
            err: dict[str, float] = {}
            probe.reset()
            for backend in backends:
                model = _get_model(backend)
                runner = _build_runner(model, backend)
                with torch.no_grad():
                    runner(gate, value)  # compile + cache warm
                    out = runner(gate, value)
                torch.cuda.synchronize()

                err[backend] = (out.float() - ref.float()).abs().max().item() if backend != "eager" else 0.0
                us = _bench(lambda r=runner, gate=gate, value=value: r(gate, value))
                results[backend] = us

            if probe.backend_calls == 0 or probe.tir_kernels == 0:
                raise AssertionError(
                    f"[{shape} {_DTYPE_LABEL[dtype]}] "
                    "TileLang backend was not exercised "
                    f"(calls={probe.backend_calls}, kernels={probe.tir_kernels})"
                )

            label = _DTYPE_LABEL[dtype]
            print(
                f"{str(tuple(shape)):<18s} {label:<6s} "
                f"{results['eager']:>10.2f}  "
                f"{results['inductor']:>10.2f}  "
                f"{results['tilelang']:>10.2f}  "
                f"{results['tilelang'] / results['eager']:>10.3f}  "
                f"{results['tilelang'] / results['inductor']:>10.3f}  "
                f"{max(err['inductor'], err['tilelang']):>10.4f}"
            )
            csv_rows.append(
                {
                    "kernel": "silu_and_mul",
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
