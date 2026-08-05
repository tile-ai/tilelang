from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from tilelang.backend.device_codegen import global_func_device_codegen


build_metal = global_func_device_codegen("target.build.tilelang_metal")
build_metal_without_compile = global_func_device_codegen("target.build.tilelang_metal_without_compile")


def mark_host_metal_context(mod: IRModule, target_host: Target, target: Target) -> IRModule:
    from tilelang.metal.transform import MarkHostMetalContext

    return MarkHostMetalContext()(mod)
