"""CuTeDSL backend manifest sharing the CUDA lowering pipeline."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import nvcc

from . import codegen, execution_backend, pipeline


def _toolchain_stamp() -> dict[str, str]:
    from tilelang.jit.adapter.cutedsl.checks import get_cutedsl_version

    # NVCC builds the host launcher and CuTeDSL compiles the device kernel.
    return {**nvcc.toolchain_stamp(), "cutedsl": get_cutedsl_version() or "unknown"}


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
        toolchain_stamp=_toolchain_stamp,
    )
)
