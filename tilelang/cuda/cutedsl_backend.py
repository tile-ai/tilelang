"""CuTeDSL backend manifest sharing the CUDA lowering pipeline."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.capabilities import MatrixInstruction, target_limits
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline


def _capabilities(target):
    return target_limits(
        target,
        subgroup_width=32,
        matrix_instructions=(
            MatrixInstruction(16, 16, 16, "float16", "float32"),
            MatrixInstruction(16, 16, 16, "bfloat16", "float32"),
        ),
        features=frozenset({"async_copy", "subgroup_exchange", "atomic.add.float32", "atomic.add.int32"}),
    )


BACKEND = register_backend(
    BackendModule(
        name="cutedsl",
        target_kinds=("cuda",),
        supports_target=codegen.is_cutedsl_target,
        pipelines={"cuda": pipeline.CUDA_PIPELINE},
        device_codegens={
            "cuda": DeviceCodegen(
                "cutedsl",
                build=codegen.build_cutedsl,
                build_without_compile=codegen.build_cutedsl_without_compile,
            )
        },
        execution_backends=execution_backend.CUTEDSL_EXECUTION_BACKENDS,
        capabilities=_capabilities,
    )
)
