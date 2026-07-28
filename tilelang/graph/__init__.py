"""TileLang torch.compile backend.

Usage::

    import tilelang
    compiled = torch.compile(model, backend="tilelang")

Custom dispatch control::

    from tilelang.graph import backend_config, default_extern_dispatch

    def my_dispatch(node):
        return default_extern_dispatch(node, gemm_threshold=256)

    backend_config.extern_dispatch = my_dispatch
    compiled = torch.compile(model, backend="tilelang")
"""

from tilelang.graph.backend import tilelang_backend, backend_config
from tilelang.graph.converter import default_extern_dispatch

try:
    from torch._dynamo.backends.registry import register_backend

    register_backend(name="tilelang")(tilelang_backend)
except ImportError:
    pass

__all__ = ["tilelang_backend", "backend_config", "default_extern_dispatch"]
