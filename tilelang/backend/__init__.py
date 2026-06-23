from .pass_pipeline import PassPipeline, register_pipeline, resolve_pipeline  # noqa: F401
from .device_codegen import (  # noqa: F401
    DeviceCodegen,
    allowed_device_codegens_for_target,
    register_device_codegen,
    register_lazy_device_codegen,
    resolve_device_codegen,
)
from .execution_backend import (  # noqa: F401
    ExecutionBackendSpec,
    allowed_backends_for_target,
    canonicalize_execution_backend,
    register_execution_backend,
    register_lazy_execution_backends,
    resolve_execution_backend,
    resolve_execution_backend_spec,
)
from .target import (  # noqa: F401
    auto_detect_target,
    list_target_detectors,
    register_target_detector,
    register_target_normalizer,
)

register_lazy_execution_backends("cuda", "tilelang.cuda.execution_backend")
register_lazy_execution_backends("hip", "tilelang.rocm.execution_backend")
register_lazy_execution_backends("c", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("llvm", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("metal", "tilelang.metal.execution_backend")

register_lazy_device_codegen("cuda", "tilelang.cuda.codegen")
register_lazy_device_codegen("hip", "tilelang.rocm.codegen")
register_lazy_device_codegen("c", "tilelang.cpu.codegen")
register_lazy_device_codegen("llvm", "tilelang.cpu.codegen")
register_lazy_device_codegen("metal", "tilelang.metal.codegen")
register_lazy_device_codegen("webgpu", "tilelang.webgpu.codegen")

from . import common as common  # noqa: F401,E402
