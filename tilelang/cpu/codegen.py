from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_c = global_func_device_codegen("target.build.tilelang_c")
build_llvm = global_func_device_codegen("target.build.llvm")
