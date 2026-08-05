"""CuTeDSL backend manifest sharing the CUDA lowering pipeline."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline

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
    )
)
