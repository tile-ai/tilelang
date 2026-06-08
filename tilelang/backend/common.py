from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody


register_pipeline(PassPipeline("webgpu", CPUPassPipelineBody))
register_execution_backend("webgpu", ExecutionBackendSpec("cython", priority=100), override=True)
register_execution_backend(
    "webgpu",
    ExecutionBackendSpec("tvm_ffi", priority=10, enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
