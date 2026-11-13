from __future__ import annotations

import shlex
import warnings

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend
from tilelang.contrib import hipcc
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.rocm.target import target_get_mcpu
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline


# nvcc spellings that clang rejects. A kernel written for both vendors may carry
# them, so they are dropped here instead of failing the HIP build.
_NVCC_ONLY_FLAGS = frozenset({"--use_fast_math", "-use_fast_math", "--extra-device-vectorization", "-extra-device-vectorization"})


def hip_compile_flags(flags):
    """Turn compile_flags / tl.device_compile_flags into hipcc arguments.

    Each entry is split like a shell, so one string can hold "-mllvm <option>".
    Flags that only nvcc understands are dropped with a warning.
    """
    if not flags:
        return []
    if isinstance(flags, str):
        flags = [flags]
    tokens = [token for flag in flags for token in shlex.split(str(flag))]
    dropped = [token for token in tokens if token in _NVCC_ONLY_FLAGS]
    if dropped:
        warnings.warn(f"hipcc does not accept the nvcc flag(s) {dropped}; ignored on HIP", stacklevel=2)
    return [token for token in tokens if token not in _NVCC_ONLY_FLAGS]


def tilelang_callback_hip_compile(code, target, pass_config=None):
    """Compile generated HIP source into an HSACO binary."""

    cfg = pass_config or {}
    return hipcc.compile_hip(
        code,
        target_format="hsaco",
        arch=target_get_mcpu(target),
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            *hip_compile_flags(cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)),
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
