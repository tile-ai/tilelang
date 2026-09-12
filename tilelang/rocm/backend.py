from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.capabilities import MatrixInstruction, target_limits
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.rocm.target import target_get_mcpu, target_get_warp_size

from . import codegen, execution_backend, pipeline


def _capabilities(target):
    return target_limits(
        target,
        subgroup_width=target_get_warp_size(target),
        matrix_instructions=(
            MatrixInstruction(16, 16, 16, "float16", "float32"),
            MatrixInstruction(16, 16, 16, "bfloat16", "float32"),
        ),
        features=frozenset({"subgroup_exchange", "atomic.add.float32", "atomic.add.int32"}),
    )


def tilelang_callback_hip_compile(code, target):
    """Compile generated HIP source into an HSACO binary."""

    return hipcc.compile_hip(
        code,
        target_format="hsaco",
        arch=target_get_mcpu(target),
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
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
        capabilities=_capabilities,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_hip_compile": tilelang_callback_hip_compile},
    )
)
