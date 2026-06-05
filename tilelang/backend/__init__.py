from .backend import Backend, ExecutionBackendSpec  # noqa: F401
from .pass_pipeline import PassPipeline, register_pipeline, resolve_pipeline  # noqa: F401
from .registry import (  # noqa: F401
    get_backend,
    list_backends,
    register_backend,
    register_lazy_backend,
    resolve_backend,
)

register_lazy_backend("cuda", "tilelang.cuda.backend")
register_lazy_backend("hip", "tilelang.rocm.backend")
register_lazy_backend("c", "tilelang.cpu.backend")
register_lazy_backend("llvm", "tilelang.cpu.backend")
register_lazy_backend("metal", "tilelang.metal.backend")

from . import common as common  # noqa: F401,E402
