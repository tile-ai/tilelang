from __future__ import annotations

import tvm_ffi
from tvm.target import Target

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, register_backend
from tilelang.backend.codegen import build_device_with_global_func
from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.rocm.pipeline import rocm_pipeline
from tilelang.rocm.target import target_get_mcpu, target_get_warp_size

_CALLBACKS_REGISTERED = False


def _tilelang_callback_hip_compile(code, target):
    from tilelang.contrib import hipcc

    arch = target_get_mcpu(target)
    return hipcc.compile_hip(
        code,
        target_format="hsaco",
        arch=arch,
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        ],
        verbose=False,
    )


def register_rocm_callbacks() -> None:
    global _CALLBACKS_REGISTERED
    if _CALLBACKS_REGISTERED:
        return
    tvm_ffi.register_global_func("tilelang_callback_hip_compile", f=_tilelang_callback_hip_compile, override=True)
    _CALLBACKS_REGISTERED = True


def rocm_device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return build_device_with_global_func(device_mod, target, "target.build.tilelang_hip")


def rocm_device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return build_device_with_global_func(device_mod, target, "target.build.tilelang_hip_without_compile")


rocm_backend = Backend(
    name="rocm",
    target_kinds=("hip",),
    import_path="tilelang.rocm",
    pipeline=rocm_pipeline,
    device_codegen=rocm_device_codegen,
    device_codegen_without_compile=rocm_device_codegen_without_compile,
    register_callbacks=register_rocm_callbacks,
    features={
        "mcpu": target_get_mcpu,
        "warp_size": target_get_warp_size,
    },
    execution_backends={
        "tvm_ffi": ExecutionBackendSpec(
            name="tvm_ffi",
            enable_host_codegen=True,
            enable_device_compile=True,
        ),
        "cython": ExecutionBackendSpec(name="cython"),
    },
    default_execution_backend="tvm_ffi",
    cmake_name="ROCM",
)

register_backend(rocm_backend, override=True)
