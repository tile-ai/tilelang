from .pass_pipeline import PassPipeline, register_pipeline, resolve_pipeline  # noqa: F401
from .execution_backend import (  # noqa: F401
    ExecutionBackendSpec,
    allowed_backends_for_target,
    canonicalize_execution_backend,
    register_execution_backend,
    register_lazy_execution_backends,
    resolve_execution_backend,
    resolve_execution_backend_spec,
)
from .target import auto_detect_target, list_target_detectors, register_lazy_target_detector, register_target_detector  # noqa: F401

register_lazy_target_detector("cuda", "tilelang.cuda.target")
register_lazy_target_detector("hip", "tilelang.rocm.target")
register_lazy_target_detector("metal", "tilelang.metal.target")

register_lazy_execution_backends("cuda", "tilelang.cuda.execution_backend")
register_lazy_execution_backends("hip", "tilelang.rocm.execution_backend")
register_lazy_execution_backends("c", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("llvm", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("metal", "tilelang.metal.execution_backend")

from . import common as common  # noqa: F401,E402
