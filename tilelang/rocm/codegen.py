from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


register_device_codegen(
    "hip",
    DeviceCodegen(
        "hip",
        build=global_func_device_codegen("target.build.tilelang_hip"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_hip_without_compile"),
    ),
    override=True,
)
