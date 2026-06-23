from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


register_device_codegen(
    "c",
    DeviceCodegen(
        "c",
        build_without_compile=global_func_device_codegen("target.build.tilelang_c"),
    ),
    override=True,
)

register_device_codegen(
    "llvm",
    DeviceCodegen(
        "llvm",
        build=global_func_device_codegen("target.build.llvm"),
        build_without_compile=global_func_device_codegen("target.build.llvm"),
    ),
    override=True,
)
