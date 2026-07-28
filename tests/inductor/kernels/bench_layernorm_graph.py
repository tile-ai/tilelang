"""Benchmark LayerNorm through the TileLang graph backend.

Reuses the ``layernorm`` pattern in ``tilelang/graph/patterns/layernorm.py``.
That pattern matches the explicit two-pass chain:

    astype(fp32) → mean → subtract → power(2) → mean → add(eps)
    → rsqrt → multiply(diff, rstd) → astype(orig) → multiply(w) → add(bias)

Backend routing:
  - eager / inductor  →  NativeLayerNorm  (``F.layer_norm``, no cast overhead)
  - tilelang          →  LayerNorm        (explicit fp32 cast, required for pattern)

Both models share identical weights and bias so ``max_err`` measures the
inter-implementation numerical difference, not TileLang's internal precision.

Workloads (from the LayerNormFwdOp spec):
  - llama-3.1-8b-prefill   (2048, 4096)  float16 / bfloat16
  - llama-3.1-8b-decode    (1,    4096)  bfloat16
  - llama-3.1-70b-prefill  (2048, 8192)  float16 / bfloat16
  - llama-3.1-70b-decode   (1,    8192)  bfloat16
  - llama-3.1-405b-prefill (2048,16384)  float16 / bfloat16
  - llama-3.1-405b-decode  (1,   16384)  bfloat16
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
# Model definitions
# ---------------------------------------------------------------------------


class LayerNorm(torch.nn.Module):
    """Explicit fp32 cast variant required for TileLang pattern matching.

    The two astype nodes and the explicit subtract/power/mean chain are
    what the layernorm pattern rewriter looks for in the Relax IR.
    """

    def __init__(self, n: int, eps: float = 1e-5, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(n, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        diff = x32 - mean
        var = (diff**2).mean(dim=-1, keepdim=True)
        norm = diff * torch.rsqrt(var + self.eps)
        return self.weight * norm.to(in_dtype) + self.bias


class NativeLayerNorm(torch.nn.Module):
    """Native ``F.layer_norm`` variant for eager and Inductor baselines.

    Uses PyTorch's built-in fused kernel; no explicit cast overhead.
    """

    def __init__(self, n: int, eps: float = 1e-5, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.n = n
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(n, dtype=dtype))

    def forward(self, x):
        return F.layer_norm(x, [self.n], weight=self.weight, bias=self.bias, eps=self.eps)


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
# Optional TIR / CUDA dump
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
            if "layernorm" in name.lower() or "layer_norm" in name.lower():
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
    print(f"TileLang lowered output — LayerNorm {label}")
    print(f"{'=' * 80}")

    if not captured_tir:
        print("(no layernorm-like PrimFunc found — pattern did not fire)")
        return

    for name, body in captured_tir.items():
        print(f"\n--- TIR: {name} ---")
        print(body)

    if captured_cuda:
        print("\n--- CUDA source ---")
        lines = captured_cuda.split("\n")
        in_block, brace_depth, shown = False, 0, 0
        for line in lines:
            if (
                not in_block
                and ("layernorm" in line.lower() or "layer_norm" in line.lower())
                and ("__global__" in line or "__device__" in line)
            ):
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

_CONFIGS = [
    # (M,     N,     dtype,            label)
    (2048, 4096, torch.float16, "llama-3.1-8b-prefill-fp16"),
    (2048, 4096, torch.bfloat16, "llama-3.1-8b-prefill-bf16"),
    (1, 4096, torch.bfloat16, "llama-3.1-8b-decode-bf16"),
    (2048, 8192, torch.float16, "llama-3.1-70b-prefill-fp16"),
    (2048, 8192, torch.bfloat16, "llama-3.1-70b-prefill-bf16"),
    (1, 8192, torch.bfloat16, "llama-3.1-70b-decode-bf16"),
    (2048, 16384, torch.float16, "llama-3.1-405b-prefill-fp16"),
    (2048, 16384, torch.bfloat16, "llama-3.1-405b-prefill-bf16"),
    (1, 16384, torch.bfloat16, "llama-3.1-405b-decode-bf16"),
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
    ap.add_argument("--dump-config", default="1:8192:bf16", help="Config to dump (format M:N:dtype, e.g. 1:8192:bf16).")
    ap.add_argument("--csv", type=Path, default=None, help="Append results to this CSV file.")
    args = ap.parse_args()

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    if args.dump:
        dm, dn, dd = args.dump_config.split(":")
        dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        dm, dn = int(dm), int(dn)
        dtype = dtype_map[dd]
        model_tl = LayerNorm(dn, dtype=dtype).cuda().eval()
        x = torch.randn(dm, dn, dtype=dtype, device=dev)
        _dump_tilelang_source_once(model_tl, x, f"M={dm} N={dn} {dd}")

    csv_rows: list[dict] = []

    hdr = f"{'config':<36s}  {'eager(µs)':>10s}  {'ind(µs)':>10s}  {'tl(µs)':>10s}  {'TL/Eager':>10s}  {'TL/Ind':>10s}  {'max_err':>10s}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for M, N, dt, label in _CONFIGS:
        torch.manual_seed(0)
        x = torch.randn(M, N, dtype=dt, device=dev)

        # Two separate model instances; weights and bias are synced.
        model_native = NativeLayerNorm(N, eps=1e-5, dtype=dt).cuda().eval()
        model_tl = LayerNorm(N, eps=1e-5, dtype=dt).cuda().eval()
        model_tl.weight.data.copy_(model_native.weight.data)
        model_tl.bias.data.copy_(model_native.bias.data)

        # Ground-truth reference: native F.layer_norm in eager mode.
        with torch.no_grad():
            ref = model_native(x).clone()

        results: dict[str, float] = {}
        err: dict[str, float] = {}
        for backend in backends:
            model_b = _get_model(model_native, model_tl, backend)
            runner = _build_runner(model_b, backend)
            with torch.no_grad():
                runner(x)  # compile + cache warm
                out = runner(x)
            torch.cuda.synchronize()

            err[backend] = (out.float() - ref.float()).abs().max().item() if backend != "eager" else 0.0
            us = _bench(lambda r=runner, x=x: r(x))
            results[backend] = us

        print(
            f"{label:<36s}  "
            f"{results['eager']:>10.2f}  "
            f"{results['inductor']:>10.2f}  "
            f"{results['tilelang']:>10.2f}  "
            f"{results['tilelang'] / results['eager']:>10.3f}  "
            f"{results['tilelang'] / results['inductor']:>10.3f}  "
            f"{max(err['inductor'], err['tilelang']):>10.4f}"
        )
        csv_rows.append(
            {
                "kernel": "layernorm",
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
