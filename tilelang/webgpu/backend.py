"""WebGPU backend registration module."""

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend
from tilelang.backend.module import BackendModule, register_backend_module

from . import codegen as codegen  # noqa: F401
from . import pipeline as pipeline  # noqa: F401

register_execution_backend("webgpu", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "webgpu",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)

BACKEND_MODULE = register_backend_module(BackendModule("webgpu", ("webgpu",)))
