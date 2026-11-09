"""Experimental Torch-managed Ascend graph compilation.

Import this module explicitly before using
``torch.compile(model, backend="tilelang_ascend")``. Importing TileLang alone
does not import Torch Dynamo or register a graph backend.
"""

from torch._dynamo.backends.registry import register_backend

from .backend import make_backend, tilelang_backend
from .config import GraphConfig
from .frontend import UnsupportedGraphError, fx_to_relax
from .runtime import build_runner

register_backend(tilelang_backend, name="tilelang_ascend")

__all__ = ["GraphConfig", "UnsupportedGraphError", "build_runner", "fx_to_relax", "make_backend", "tilelang_backend"]
