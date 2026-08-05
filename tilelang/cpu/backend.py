"""CPU backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline

BACKEND = register_backend(
    BackendModule(
        name="cpu",
        target_kinds=("c", "llvm"),
        pipelines={
            "c": PassPipeline("c", pipeline.CPUPassPipelineBody),
            "llvm": PassPipeline("llvm", pipeline.CPUPassPipelineBody),
        },
        device_codegens={
            "c": (DeviceCodegen("c", build_without_compile=codegen.build_c),),
            "llvm": (
                DeviceCodegen(
                    "llvm",
                    build=codegen.build_llvm,
                    build_without_compile=codegen.build_llvm,
                ),
            ),
        },
        host_codegens=STANDARD_HOST_CODEGENS,
        execution_backends=execution_backend.EXECUTION_BACKENDS,
    )
)
