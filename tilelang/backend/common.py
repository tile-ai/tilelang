from __future__ import annotations

from tvm.target import Target

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, register_backend
from tilelang.backend.codegen import build_device_with_global_func
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.cpu.pipeline import CPUPassPipelineBody


register_pipeline(PassPipeline("webgpu", CPUPassPipelineBody))


def webgpu_device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return build_device_with_global_func(device_mod, target, "target.build.webgpu")


webgpu_backend = Backend(
    name="webgpu",
    target_kinds=("webgpu",),
    import_path="tilelang.backend.common",
    pipeline=CPUPassPipelineBody,
    device_codegen_without_compile=webgpu_device_codegen_without_compile,
    execution_backends={
        "tvm_ffi": ExecutionBackendSpec(
            name="tvm_ffi",
            enable_host_codegen=True,
            enable_device_compile=False,
        ),
    },
    default_execution_backend="tvm_ffi",
    cmake_name="WEBGPU",
)

register_backend(webgpu_backend, override=True)
