from __future__ import annotations

from tvm import IRModule, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.execution_backend import (
    ExecutionBackendSpec,
    register_execution_backend,
)
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBodyAfterKernelLaunch


def WebGPUPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.LowerKernelLaunchToThreadBinding()(mod)
    return CPUPassPipelineBodyAfterKernelLaunch(mod, target)


register_pipeline(PassPipeline("webgpu", WebGPUPassPipelineBody))
register_execution_backend("webgpu", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "webgpu",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
