from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_hip = global_func_device_codegen("target.build.tilelang_hip")
build_hip_without_compile = global_func_device_codegen("target.build.tilelang_hip_without_compile")
