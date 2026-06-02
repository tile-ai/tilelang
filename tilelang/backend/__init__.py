from .backend import Backend, ExecutionBackendSpec  # noqa: F401
from .pass_pipeline import PassPipeline, register_pipeline, resolve_pipeline  # noqa: F401
from .device_codegen import (  # noqa: F401
    DeviceCodegen,
    allowed_device_codegens_for_target,
    register_device_codegen,
    register_lazy_device_codegen,
    resolve_device_codegen,
)
from .host_codegen import (  # noqa: F401
    HostCodegen,
    HostCodegenHook,
    allowed_host_codegens_for_target,
    apply_host_codegen_hooks,
    register_host_codegen,
    register_host_codegen_hook,
    register_lazy_host_codegen,
    register_lazy_host_codegen_hooks,
    resolve_host_codegen,
)
from .execution_backend import (  # noqa: F401
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
from .registry import (  # noqa: F401
    get_backend,
    list_backends,
    register_backend,
    register_lazy_backend,
    resolve_backend,
)

register_lazy_execution_backends("cuda", "tilelang.cuda.execution_backend")
register_lazy_execution_backends("hip", "tilelang.rocm.execution_backend")
register_lazy_execution_backends("c", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("llvm", "tilelang.cpu.execution_backend")
register_lazy_execution_backends("metal", "tilelang.metal.execution_backend")
register_lazy_execution_backends("webgpu", "tilelang.backend.common")

register_lazy_device_codegen("cuda", "tilelang.cuda.codegen")
register_lazy_device_codegen("hip", "tilelang.rocm.codegen")
register_lazy_device_codegen("c", "tilelang.cpu.codegen")
register_lazy_device_codegen("llvm", "tilelang.cpu.codegen")
register_lazy_device_codegen("metal", "tilelang.metal.codegen")
register_lazy_device_codegen("webgpu", "tilelang.webgpu.codegen")

register_lazy_host_codegen("c", "tilelang.cpu.codegen")
register_lazy_host_codegen("llvm", "tilelang.cpu.codegen")
register_lazy_host_codegen_hooks("metal", "tilelang.metal.codegen")

register_lazy_backend("cuda", "tilelang.cuda.backend")
register_lazy_backend("hip", "tilelang.rocm.backend")
register_lazy_backend("c", "tilelang.cpu.backend")
register_lazy_backend("llvm", "tilelang.cpu.backend")
register_lazy_backend("metal", "tilelang.metal.backend")
register_lazy_backend("webgpu", "tilelang.backend.common")

from . import common as common  # noqa: F401,E402
