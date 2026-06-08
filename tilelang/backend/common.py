from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.backend.target import register_supported_target
from tilelang.cpu.pipeline import CPUPassPipelineBody


register_supported_target("c", "C source backend.", override=True)
register_supported_target("llvm", "LLVM CPU target. Use dict options such as {'kind': 'llvm', 'mcpu': 'native'}.", override=True)
register_supported_target("webgpu", "WebGPU target for browser/WebGPU runtimes.", override=True)
register_pipeline(PassPipeline("webgpu", CPUPassPipelineBody))
register_execution_backend("webgpu", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "webgpu",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
