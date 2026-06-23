from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


_build_llvm = global_func_device_codegen("target.build.llvm")


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
        build=_build_llvm,
        build_without_compile=_build_llvm,
    ),
    override=True,
)
