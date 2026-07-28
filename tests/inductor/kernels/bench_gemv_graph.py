"""Benchmark GEMV through the TileLang graph backend.

Measures matrix-vector products for the canonical GEMV shapes where
either ``M == 1`` (vector × matrix: ``F.linear``-style) or ``N == 1``
(matrix × vector).  For each config we compare:

  - eager     : PyTorch's native ``torch.matmul`` / ``F.linear``
  - inductor  : ``torch.compile(..., backend="inductor")``
  - tilelang  : ``torch.compile(..., backend="tilelang")``

Workloads follow the standard BLAS ``M, N, K`` convention where
``C = op(A) @ op(B)`` with:

  - ``A`` stored as ``(M, K)`` when ``trans_a == False`` else ``(K, M)``
  - ``B`` stored as ``(K, N)`` when ``trans_b == False`` else ``(N, K)``
  - ``C`` always has shape ``(M, N)``

The two ``trans_*`` flags match the ``pytest.param`` convention where
``True`` means the matrix is stored transposed vs. the mathematical
operand.  In the configs below, the side whose outer extent is 1
(``M == 1`` or ``N == 1``) plays the role of the activation vector and
is re-generated per iteration; the other matrix is treated as a
persistent weight stored as a ``nn.Parameter`` inside the module so
``torch.compile`` sees a stable inductor/tilelang trace.
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


class Gemv(torch.nn.Module):
    def forward(self, x, y: torch.Tensor):
        """Compute ``C = op(A) @ op(B)``.

        The caller passes the activation vector as ``x``; the module
        materialises the full product using PyTorch so the backend
        observes a single matmul in the fx graph.
        """
        return x @ y.transpose(0, 1).contiguous()


class NativeGemv(torch.nn.Module):
    def forward(self, x, y: torch.Tensor):
        """Native PyTorch matmul for reference."""
        return x @ y.T


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


def _make_input(N: int, K: int, dtype: torch.dtype, device: str):
    """Build a random activation tensor for the given GEMV shape."""

    left = torch.randn((1, K), dtype=dtype, device=device)
    right = torch.randn((N, K), dtype=dtype, device=device)

    return left, right


# ---------------------------------------------------------------------------
# Optional TIR / CUDA dump for one config
# ---------------------------------------------------------------------------


def _dump_tilelang_source_once(model, inputs, label):
    """Compile ``model`` on the TileLang backend once and dump TIR + CUDA.

    ``inputs`` is a tuple of positional arguments to ``model.forward`` —
    for the two-operand GEMV module this is ``(x, y)``.
    """
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
            runner(*inputs)
        torch.cuda.synchronize()
    finally:
        _vm_mod._compile_kernels_tilelang = orig

    print(f"\n{'=' * 80}")
    print(f"TileLang lowered output — GEMV {label}")
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

# (M, N, K, dtype, label)
#
# All shapes are ``C = A @ B.T`` where ``A: (M, K)`` and ``B: (N, K)`` —
# the standard weight layout for LLM linear projections (``F.linear``).
# With ``M = 1`` these are decode-time GEMVs.  Labels capture the origin
# model and module so regressions can be traced back to real workloads.
# ``trans_a=False, trans_b=True`` is baked into the ``Gemv`` module above
# (``return x @ y.T``) so it isn't listed per-config.
_CONFIGS = [
    # 004: Qwen3 30B A3B moe.gate
    (1, 128, 2048, torch.bfloat16, "004_moe_gate_n128_k2048"),
    # 005: DeepSeek-V3 moe.gate
    (1, 256, 7168, torch.bfloat16, "005_moe_gate_n256_k7168"),
    # 006: Qwen3 30B A3B attn.o_proj
    (1, 2048, 4096, torch.bfloat16, "006_o_proj_n2048_k4096"),
    # 007: Llama 3.1 8B attn.o_proj
    (1, 4096, 4096, torch.bfloat16, "007_o_proj_n4096_k4096"),
    # 008: Llama 3.1 8B mlp.down_proj
    (1, 4096, 14336, torch.bfloat16, "008_down_proj_n4096_k14336"),
    # 009: Qwen3 30B A3B attn.qkv_pro
    (1, 5120, 2048, torch.bfloat16, "009_qkv_proj_n5120_k2048"),
    # 010: Llama 3.1 8B attn.qkv_pro
    (1, 6144, 4096, torch.bfloat16, "010_qkv_proj_n6144_k4096"),
    # 011: Llama 3.1 8B mlp.gate_up_p
    (1, 28672, 4096, torch.bfloat16, "011_gate_up_proj_n28672_k4096"),
]

_CSV_COLUMNS = [
    "kernel",
    "config",
    "eager_us",
    "inductor_us",
    "tilelang_us",
    "speedup_vs_eager",
    "speedup_vs_ind",
    "tilelang_max_abs_fp32",
    "tilelang_rel_linf_fp32",
    "tilelang_rel_rmse_fp32",
    "tilelang_allclose_fp32",
]


def _error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float | bool]:
    out_fp32 = out.float()
    ref_fp32 = ref.float()
    diff = out_fp32 - ref_fp32

    max_abs = diff.abs().max().item()

    ref_linf = ref_fp32.abs().max().item()
    rel_linf = max_abs / max(ref_linf, 1e-12)

    rmse = diff.square().mean().sqrt().item()
    ref_rmse = ref_fp32.square().mean().sqrt().item()
    rel_rmse = rmse / max(ref_rmse, 1e-12)

    return {
        "max_abs": max_abs,
        "rel_linf": rel_linf,
        "rel_rmse": rel_rmse,
        "allclose": torch.allclose(out_fp32, ref_fp32, rtol=5e-3, atol=5e-3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true", help="Dump TileLang TIR and CUDA source for one config.")
    ap.add_argument("--dump-config", default="128:2048:bf16", help="Config to dump (format N:K:dtype, e.g. 128:2048:bf16).")
    ap.add_argument("--csv", type=Path, default=None, help="Write results to this CSV file.")
    args = ap.parse_args()

    dev = "cuda"
    backends = ["eager", "inductor", "tilelang"]

    if args.dump:
        dN, dK, dtype_str = args.dump_config.split(":")
        dN, dK = int(dN), int(dK)
        dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
        ddt = dtype_map[dtype_str]
        torch.manual_seed(0)
        dump_model = Gemv().cuda().eval()
        dump_x, dump_y = _make_input(dN, dK, ddt, dev)
        _dump_tilelang_source_once(
            dump_model,
            (dump_x, dump_y),
            f"N={dN} K={dK} {dtype_str}",
        )

    csv_rows: list[dict] = []

    hdr = (
        f"{'config':<26s}  {'eager(µs)':>10s}  {'ind(µs)':>10s}  "
        f"{'tl(µs)':>10s}  {'TL/Eager':>10s}  {'TL/Ind':>10s}  "
        f"{'rel_linf':>10s}  {'rel_rmse':>10s}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))

    for _M, N, K, dt, label in _CONFIGS:
        torch.manual_seed(0)
        model = Gemv().cuda().eval()
        model_native = NativeGemv().cuda().eval()
        x, y = _make_input(N, K, dt, dev)

        # Ground-truth reference: float32 GEMV computed from float32-cast inputs.
        with torch.no_grad():
            ref = x.float() @ y.float().transpose(0, 1).contiguous()

        results: dict[str, float] = {}
        metrics: dict[str, dict[str, float | bool]] = {}
        for backend in backends:
            model_b = model_native if backend != "tilelang" else model
            runner = _build_runner(model_b, backend)
            with torch.no_grad():
                runner(x, y)  # compile + cache warm
                out = runner(x, y)
            torch.cuda.synchronize()

            metrics[backend] = _error_metrics(out, ref)
            us = _bench(lambda r=runner, x=x, y=y: r(x, y))
            results[backend] = us

        print(
            f"{label:<26s}  "
            f"{results['eager']:>10.2f}  "
            f"{results['inductor']:>10.2f}  "
            f"{results['tilelang']:>10.2f}  "
            f"{results['tilelang'] / results['eager']:>10.3f}  "
            f"{results['tilelang'] / results['inductor']:>10.3f}  "
            f"{metrics['tilelang']['rel_linf']:>10.4e}  "
            f"{metrics['tilelang']['rel_rmse']:>10.4e}"
        )
        csv_rows.append(
            {
                "kernel": "gemv",
                "config": label,
                "eager_us": f"{results['eager']:.4f}",
                "inductor_us": f"{results['inductor']:.4f}",
                "tilelang_us": f"{results['tilelang']:.4f}",
                "speedup_vs_eager": f"{results['eager'] / results['tilelang']:.4f}",
                "speedup_vs_ind": f"{results['inductor'] / results['tilelang']:.4f}",
                "tilelang_max_abs_fp32": f"{metrics['tilelang']['max_abs']:.6f}",
                "tilelang_rel_linf_fp32": f"{metrics['tilelang']['rel_linf']:.6e}",
                "tilelang_rel_rmse_fp32": f"{metrics['tilelang']['rel_rmse']:.6e}",
                "tilelang_allclose_fp32": str(metrics["tilelang"]["allclose"]).lower(),
            }
        )

    if args.csv is not None and csv_rows:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {len(csv_rows)} rows → {args.csv}")


if __name__ == "__main__":
    main()
