from __future__ import annotations

from tvm import IRModule
from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen
from tilelang.backend.host_codegen import HostCodegenHook, register_host_codegen_hook


_build_metal = global_func_device_codegen("target.build.tilelang_metal")


def _mark_host_metal_context(mod: IRModule, target_host: Target, target: Target) -> IRModule:
    from tilelang.metal.transform import MarkHostMetalContext

    return MarkHostMetalContext()(mod)


register_device_codegen(
    "metal",
    DeviceCodegen(
        "metal",
        build=_build_metal,
        build_without_compile=_build_metal,
    ),
    override=True,
)

register_host_codegen_hook(
    "metal",
    HostCodegenHook("metal_context", _mark_host_metal_context),
    override=True,
)
