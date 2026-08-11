from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import global_func_device_codegen


def is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" in target.keys


def is_plain_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" not in target.keys


build_cuda = global_func_device_codegen("target.build.tilelang_cuda")
build_cuda_without_compile = global_func_device_codegen("target.build.tilelang_cuda_without_compile")
build_cutedsl = global_func_device_codegen("target.build.tilelang_cutedsl")
build_cutedsl_without_compile = global_func_device_codegen("target.build.tilelang_cutedsl_without_compile")
