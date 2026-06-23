from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


register_device_codegen(
    "metal",
    DeviceCodegen(
        "metal",
        build=global_func_device_codegen("target.build.tilelang_metal"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_metal"),
    ),
    override=True,
)
