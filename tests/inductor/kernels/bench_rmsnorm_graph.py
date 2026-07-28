"""Benchmark RMSNorm through the TileLang graph backend.

Reuses the existing ``rmsnorm`` pattern in
``tilelang/graph/patterns/rmsnorm.py`` — no extra registration needed.
That pattern matches the chain

    astype(fp32) → power(2) → mean → add(eps) → rsqrt → multiply
    → astype(orig) → multiply(w)

which is exactly what the ``RMSNorm`` module below emits.  Without the
explicit ``x.float() / .to(in_dtype)`` promotion, FuseOps cannot merge
the power/mean/rsqrt chain into a single call_tir group and the
pattern rewriter would not fire.

Backend routing:
  - eager / inductor  →  NativeRMSNorm (``F.rms_norm``, no cast overhead)
  - tilelang          →  RMSNorm       (explicit fp32 cast, required for
                                        pattern matching)

Both models are initialised with identical weights so the error column
measures the numerical difference between the two formulations, not
TileLang's internal precision.

Covers the nine configurations requested:

    M ∈ {2048}, N ∈ {4096, 8192, 16384}, dtype ∈ {float16, bfloat16}
    M ∈ {1},    N ∈ {4096, 8192, 16384}, dtype = bfloat16

For each config, compares eager / torch.compile(inductor) /
torch.compile(tilelang) latency and max-abs error vs eager.
``--dump`` prints the TileLang-generated CUDA source for one
representative config alongside Inductor's Triton output.
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
import tilelang  # noqa: F401  (registers the "tilelang" torch.compile backend)
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench
import tilelang.graph.vm_build as _vm_mod


# ---------------------------------------------------------------------------
# Model definitions — one per code path
# ---------------------------------------------------------------------------


class RMSNorm(torch.nn.Module):
    """Explicit fp32 cast variant required for TileLang pattern matching.

    The two astype nodes in the FX graph delimit the fp32 reduction region
    that the rmsnorm pattern rewriter looks for.  Without them FuseOps
    cannot form a single call_tir group and the pattern never fires.
    """

    def __init__(self, n: int, eps: float = 1e-6, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return self.weight * x32.to(in_dtype)


class NativeRMSNorm(torch.nn.Module):
    """Native ``F.rms_norm`` variant for eager and Inductor baselines.

    Uses PyTorch's built-in fused kernel; no explicit cast overhead.
    This gives a fair eager/inductor baseline without penalising them
    for the extra cast nodes that TileLang requires.
    """

    def __init__(self, n: int, eps: float = 1e-6, dtype=torch.float16):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=dtype))

    def forward(self, x):
        return F.rms_norm(x, [self.weight.shape[0]], weight=self.weight, eps=self.eps)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _bench(fn, warmup_ms=25, rep_ms=100):
    """tilelang.profiler.do_bench → µs/call."""
    return do_bench(fn) * 1000.0


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
    """Route each backend to the appropriate model."""
    return model_tl if backend == "tilelang" else model_native


# ---------------------------------------------------------------------------
# Optional TIR / CUDA dump
# ---------------------------------------------------------------------------


def _dump_tilelang_source_once(model, x, label):
    """Compile once, dump TIR and CUDA source for rmsnorm-like PrimFuncs."""
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
            if "rmsnorm" in name.lower() or "rsqrt" in name.lower():
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
    print(f"TileLang lowered output — RMSNorm {label}")
    print(f"{'=' * 80}")

    if not captured_tir:
        print("(no rmsnorm-like PrimFunc found — pattern did not fire)")
        return

    for name, body in captured_tir.items():
        print(f"\n--- TIR: {name} ---")
        print(body)

    if captured_cuda:
        print("\n--- CUDA source (all device kernels in the VM) ---")
        lines = captured_cuda.split("\n")
        in_block = False
        brace_depth = 0
        shown = 0
        for line in lines:
            if not in_block and ("rmsnorm" in line.lower() or "rsqrt" in line.lower()) and ("__global__" in line or "__device__" in line):
                in_block = True
                brace_depth = 0
            if in_block:
                print(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth == 0 and "}" in line and shown > 0:
                    in_block = False
                shown += 1
        if shown == 0:
            print("(no matching CUDA kernel string; dumping full source)")
            print(captured_cuda[:4000])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-ms", type=float, default=25)
    ap.add_argument("--rep-ms", type=float, default=100)
    ap.add_argument("--dump", action="store_true", help="Dump TileLang TIR and CUDA source for one config.")
    ap.add_argument("--dump-config", default="1:8192:bf16", help="Config to dump (format M:N:dtype).")
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Append results to this CSV file (columns: kernel,config,"
        "eager_us,inductor_us,tilelang_us,speedup_vs_eager,"
        "speedup_vs_ind,max_err).",
    )
    args = ap.parse_args()

    dev = "cuda"
    configs = [
        (2048, 4096, torch.float16),
        (2048, 4096, torch.bfloat16),
        (1, 4096, torch.bfloat16),
        (2048, 8192, torch.float16),
        (2048, 8192, torch.bfloat16),
        (1, 8192, torch.bfloat16),
        (2048, 16384, torch.float16),
        (2048, 16384, torch.bfloat16),
        (1, 16384, torch.bfloat16),
    ]
    backends = ["eager", "inductor", "tilelang"]

    if args.dump:
        dm, dn, dd = args.dump_config.split(":")
        dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        dm, dn = int(dm), int(dn)
        dtype = dtype_map[dd]
        model_tl = RMSNorm(dn, dtype=dtype).cuda().eval()
        x = torch.randn(dm, dn, dtype=dtype, device=dev)
        _dump_tilelang_source_once(model_tl, x, f"M={dm} N={dn} {dd}")

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
    csv_rows: list[dict] = []

    header = f"{'config':<32s}  {'eager(µs)':>10s}  {'ind(µs)':>10s}  {'tl(µs)':>10s}  {'TL/Eager':>10s}  {'TL/Ind':>10s}  {'max_err':>10s}"
    print()
    print(header)
    print("-" * len(header))

    for M, N, dt in configs:
        dt_name = "float16" if dt == torch.float16 else "bfloat16"
        label = f"rms_norm_m{M}_n{N}_{dt_name}"

        torch.manual_seed(0)
        x = torch.randn(M, N, dtype=dt, device=dev)

        # Two separate model instances, identical weights (all-ones init).
        # NativeRMSNorm → eager + inductor (no cast overhead).
        # RMSNorm       → tilelang (explicit cast, required for pattern match).
        model_native = NativeRMSNorm(N, eps=1e-6, dtype=dt).cuda().eval()
        model_tl = RMSNorm(N, eps=1e-6, dtype=dt).cuda().eval()
        # Sync weights so both models compute the exact same function.
        model_tl.weight.data.copy_(model_native.weight.data)

        # Ground-truth reference: native F.rms_norm in eager mode.
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

            # Capture runner by value to avoid closure-over-loop-variable bug.
            us = _bench(
                lambda r=runner, x=x: r(x),
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
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
                "kernel": "rmsnorm",
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
