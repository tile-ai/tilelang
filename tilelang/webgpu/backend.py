"""WebGPU backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.capabilities import target_limits
from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, pipeline


def _capabilities(target):
    return target_limits(target, subgroup_width=1)


BACKEND = register_backend(
    BackendModule(
        name="webgpu",
        target_kinds=("webgpu",),
        pipelines={"webgpu": PassPipeline("webgpu", pipeline.WebGPUPassPipelineBody)},
        device_codegens={
            "webgpu": DeviceCodegen(
                "webgpu",
                build=codegen.build_webgpu,
                build_without_compile=codegen.build_webgpu,
            )
        },
        execution_backends=(
            ExecutionBackendSpec(
                "tvm_ffi",
                enable_host_codegen=True,
                enable_device_compile=True,
                native_multi_launch=True,
                native_argument_binding=True,
            ),
        ),
        capabilities=_capabilities,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
