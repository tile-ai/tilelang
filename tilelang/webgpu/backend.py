"""WebGPU backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, pipeline

BACKEND = register_backend(
    BackendModule(
        name="webgpu",
        target_kinds=("webgpu",),
        pipelines={"webgpu": PassPipeline("webgpu", pipeline.WebGPUPassPipelineBody)},
        device_codegens={
            "webgpu": (
                DeviceCodegen(
                    "webgpu",
                    build=codegen.build_webgpu,
                    build_without_compile=codegen.build_webgpu,
                ),
            )
        },
        execution_backends=(ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),),
    )
)
