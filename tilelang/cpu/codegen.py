from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen
from tilelang.backend.host_codegen import HostCodegen, global_func_host_codegen, register_host_codegen


_build_llvm = global_func_device_codegen("target.build.llvm")
_build_host_llvm = global_func_host_codegen("target.build.llvm")
_build_host_c = global_func_host_codegen("target.build.tilelang_c_host")


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

register_host_codegen(
    "c",
    HostCodegen("c", build=_build_host_c),
    override=True,
)

register_host_codegen(
    "llvm",
    HostCodegen("llvm", build=_build_host_llvm),
    override=True,
)
