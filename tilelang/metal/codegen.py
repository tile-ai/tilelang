from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


_build_metal = global_func_device_codegen("target.build.tilelang_metal")


register_device_codegen(
    "metal",
    DeviceCodegen(
        "metal",
        build=_build_metal,
        build_without_compile=_build_metal,
    ),
    override=True,
)
