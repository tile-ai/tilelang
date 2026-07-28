"""Benchmark Softmax through the TileLang graph backend.

Measures ``torch.nn.functional.softmax(x, dim=-1)`` across three backends:

  - eager     : PyTorch's native softmax
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Workloads (from the softmax op spec):

  - attn-weights-4k   (32, 32, 4096)   float16 / bfloat16
  - attn-weights-32k  (32, 32, 32768)  bfloat16
  - lm-head-logits    (4, 102400)      float16 / bfloat16

max_err reports the absolute error vs. eager on the same seed.  Small
deviations (~1e-3 for fp16, ~1e-2 for bf16) are expected as each backend
uses its own accumulator width.
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
import torch.nn.functional as F

import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench
import tilelang.graph.vm_build as _vm_mod


# ---------------------------------------------------------------------------
# Model definition — a single softmax along dim=-1.  Wrapping it in a module
# is just so ``torch.compile`` treats it the same way as the LayerNorm bench.
# ---------------------------------------------------------------------------


class Softmax(torch.nn.Module):
    """Plain softmax along the last dimension.

    The TileLang softmax pattern matches ``R.nn.softmax`` directly on any
    float dtype and upcasts to fp32 internally (inside the generated
    kernel) for numerically stable max/exp/sum — same semantics as
    PyTorch's own ``F.softmax`` on low-precision inputs.  No explicit
    ``x.float()`` round-trip is needed in user code.
    """

    def forward(self, x):
        return F.softmax(x, dim=-1)


# Kept for backward compatibility with earlier benchmark runs that used a
# separate "native" reference model.  Both classes now do the same thing.
NativeSoftmax = Softmax


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
            name = gv.name_hint
            # Capture ALL TIR kernels so we can see whether the leading cast
            # and the softmax+cast-back live in the same or different kernels.
            captured_tir[name] = str(fn)
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
    print(f"TileLang lowered output — Softmax {label}")
    print(f"{'=' * 80}")

    if not captured_tir:
        print("(no softmax-like PrimFunc found — pattern did not fire)")
        return

    for name, body in captured_tir.items():
        print(f"\n--- TIR: {name} ---")
        print(body)

    if captured_cuda:
        print("\n--- CUDA source ---")
        lines = captured_cuda.split("\n")
        in_block, brace_depth, shown = False, 0, 0
        for line in lines:
            if not in_block and "softmax" in line.lower() and ("__global__" in line or "__device__" in line):
                in_block = True
                brace_depth = 0
            if in_block:
                print(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth == 0 and "}" in line and shown > 0:
                    in_block = False
                shown += 1
        if shown == 0:
            print("(no matching kernel; dumping first 4k chars)")
            print(captured_cuda[:4000])


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

# (shape, dtype, label)
_CONFIGS = [
    # Attention weight matrices (batch*heads, seq, seq)
    ((32, 32, 4096), torch.float16, "attn-weights-4k-fp16"),
    ((32, 32, 4096), torch.bfloat16, "attn-weights-4k-bf16"),
    ((32, 32, 32768), torch.bfloat16, "attn-weights-32k-bf16"),
    # LM head logits (batch, vocab_size)
    # ((4, 102400),       torch.float16,  "lm-head-logits-fp16"),
    # ((4, 102400),       torch.bfloat16, "lm-head-logits-bf16"),
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


def _get_model(model_native, model_tl, backend):
    return model_tl if backend == "tilelang" else model_native


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true", help="Dump TileLang TIR and CUDA source for one config.")
    ap.add_argument("--dump-config", default="32,32,32768:bf16", help="Config to dump (format shape_csv:dtype, e.g. 32,32,32768:bf16).")
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
        x = torch.randn(*shape, dtype=dtype, device=dev)
        model = Softmax().cuda().eval()
        _dump_tilelang_source_once(model, x, f"{shape} {dtype_str}")

    csv_rows: list[dict] = []

    hdr = f"{'config':<32s}  {'eager(µs)':>10s}  {'ind(µs)':>10s}  {'tl(µs)':>10s}  {'TL/Eager':>10s}  {'TL/Ind':>10s}  {'max_err':>10s}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for shape, dt, label in _CONFIGS:
        torch.manual_seed(0)
        x = torch.randn(*shape, dtype=dt, device=dev)
        model = Softmax().cuda().eval()
        model_native = NativeSoftmax().cuda().eval()

        # Ground-truth reference: eager softmax.
        with torch.no_grad():
            ref = model_native(x).clone()

        results: dict[str, float] = {}
        err: dict[str, float] = {}
        for backend in backends:
            model_b = _get_model(model_native, model, backend)
            runner = _build_runner(model_b, backend)
            with torch.no_grad():
                runner(x)  # compile + cache warm
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
                "kernel": "softmax",
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
