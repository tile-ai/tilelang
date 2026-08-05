from __future__ import annotations

import tvm_ffi

from tilelang.backend.module import BackendModule, register_backend_module
from tilelang.contrib import hipcc
from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.rocm.target import target_get_mcpu

from . import codegen as codegen  # noqa: F401
from . import execution_backend as execution_backend  # noqa: F401
from . import pipeline as pipeline  # noqa: F401


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
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


BACKEND_MODULE = register_backend_module(BackendModule("rocm", ("hip",)))
