"""Example: trace graph-mode TileLang compilation.

This example uses the current ``tilelang.graph`` torch.compile backend and
TileLang's pass-level lower_trace tool. By default it writes the HTML trace and
generated source files under ``examples/inductor/logs``.

Environment knobs:
  TL_GRAPH_TRACE_MODE=html|terminal|both|0
  TL_GRAPH_TRACE_DIR=/path/to/output
  TL_GRAPH_TRACE_CODEGEN=/path/to/codegen.cu  # output base path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_EXAMPLE_DIR.parents[1])
sys.path = [_REPO_ROOT] + [path for path in sys.path if path != _REPO_ROOT]

import torch
import torch._dynamo

import tilelang  # noqa: F401  (loads TileLang and registers the graph backend)
from tilelang.tools.lower_trace import enable as enable_lower_trace


def _trace_config() -> tuple[str, str, str]:
    default_trace_dir = _EXAMPLE_DIR / "logs"
    trace_mode = os.environ.get("TL_GRAPH_TRACE_MODE", "html")
    trace_dir = os.environ.get("TL_GRAPH_TRACE_DIR", str(default_trace_dir))
    codegen_output = os.environ.get(
        "TL_GRAPH_TRACE_CODEGEN",
        str(default_trace_dir / "codegen.cu"),
    )
    return trace_mode, trace_dir, codegen_output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA.")

    trace_mode, trace_dir, codegen_output = _trace_config()
    enable_lower_trace(
        mode=trace_mode,
        trace_dir=trace_dir,
        codegen_output=codegen_output,
    )

    # Clear Dynamo's compile cache so this debug example always emits a fresh
    # TileLang compile trace and per-FFI codegen output.
    torch._dynamo.reset()

    dim = 256
    batch = 32
    out_dim = dim * 4

    @torch.compile(backend="tilelang")
    def casted_matmul(x, weight):
        return (x @ weight).to(torch.bfloat16)

    x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, out_dim, device="cuda", dtype=torch.bfloat16)

    out = casted_matmul(x, weight)
    ref = x @ weight
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    print("=" * 70)
    print("GRAPH TRACE EXAMPLE")
    print("=" * 70)
    print(f"trace mode    : {trace_mode}")
    print(f"trace dir     : {trace_dir}")
    print(f"codegen base  : {codegen_output}")
    print(f"output        : shape={tuple(out.shape)}, dtype={out.dtype}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
