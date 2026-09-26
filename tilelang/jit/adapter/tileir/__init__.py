"""Runtime adapter for the CUDA Tile IR execution backend.

The compiler (structured lowering, Semantic IR, assembly, and toolchain
detection) lives in the top-level ``tilelang.tileir`` package. This package only
hosts the runtime adapter that turns a lowered kernel into a torch-callable
function.
"""

from .adapter import TileIRKernelAdapter  # noqa: F401

__all__ = [
    "TileIRKernelAdapter",
]
