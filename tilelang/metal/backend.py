from __future__ import annotations

from tvm.target import Target

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, register_backend
from tilelang.backend.codegen import build_device_with_global_func
from tilelang.metal.pipeline import metal_pipeline
from tilelang.metal.transform import MarkHostMetalContext


def metal_device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return build_device_with_global_func(device_mod, target, "target.build.tilelang_metal")


def metal_host_pre_codegen(host_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return MarkHostMetalContext()(host_mod)


metal_backend = Backend(
    name="metal",
    target_kinds=("metal",),
    import_path="tilelang.metal",
    pipeline=metal_pipeline,
    device_codegen=metal_device_codegen,
    device_codegen_without_compile=metal_device_codegen,
    host_pre_codegen=metal_host_pre_codegen,
    execution_backends={
        "tvm_ffi": ExecutionBackendSpec(
            name="tvm_ffi",
            enable_host_codegen=True,
            enable_device_compile=True,
        ),
        "torch": ExecutionBackendSpec(name="torch"),
    },
    default_execution_backend="tvm_ffi",
    cmake_name="METAL",
)

register_backend(metal_backend, override=True)
