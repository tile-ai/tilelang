from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen

ASCEND_CODEGEN = DeviceCodegen(
    "ascend",
    build=global_func_device_codegen("target.build.tilelang_ascend"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_ascend_without_compile"),
)
