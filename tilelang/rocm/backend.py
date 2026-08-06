from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import hipcc
from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.rocm.target import target_get_mcpu

from . import codegen, execution_backend, pipeline


def tilelang_callback_hip_compile(code, target):
    """Compile generated HIP source into an HSACO binary."""

    return hipcc.compile_hip(
        code,
        target_format="hsaco",
        arch=target_get_mcpu(target),
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        ],
        verbose=False,
    )


BACKEND = register_backend(
    BackendModule(
        name="rocm",
        target_kinds=("hip",),
        pipelines={"hip": PassPipeline("hip", pipeline.ROCMPassPipelineBody)},
        device_codegens={
            "hip": DeviceCodegen(
                "hip",
                build=codegen.build_hip,
                build_without_compile=codegen.build_hip_without_compile,
            )
        },
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_hip_compile": tilelang_callback_hip_compile},
    )
)
