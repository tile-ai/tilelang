"""Metal backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import HostCodegenHook, STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline

BACKEND = register_backend(
    BackendModule(
        name="metal",
        target_kinds=("metal",),
        pipelines={"metal": PassPipeline("metal", pipeline.MetalPassPipelineBody)},
        device_codegens={
            "metal": DeviceCodegen(
                "metal",
                build=codegen.build_metal,
                build_without_compile=codegen.build_metal_without_compile,
            )
        },
        host_codegen_hooks={"metal": (HostCodegenHook("metal_context", codegen.mark_host_metal_context),)},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
