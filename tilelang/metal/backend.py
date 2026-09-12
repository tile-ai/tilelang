"""Metal backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.capabilities import MatrixInstruction, target_limits
from tilelang.backend.host_codegen import HostCodegenHook, STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline


def _capabilities(target):
    return target_limits(
        target,
        subgroup_width=32,
        matrix_instructions=(
            MatrixInstruction(8, 8, 8, "float16", "float32"),
            MatrixInstruction(8, 8, 8, "bfloat16", "float32"),
        ),
        features=frozenset({"subgroup_exchange"}),
    )


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
        capabilities=_capabilities,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
