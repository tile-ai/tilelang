from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


def _is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" in target.keys


def _is_plain_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" not in target.keys


register_device_codegen(
    "cuda",
    DeviceCodegen(
        "cuda",
        build=global_func_device_codegen("target.build.tilelang_cuda"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_cuda_without_compile"),
        supports_target=_is_plain_cuda_target,
    ),
    override=True,
)

register_device_codegen(
    "cuda",
    DeviceCodegen(
        "cutedsl",
        build=global_func_device_codegen("target.build.tilelang_cutedsl"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_cutedsl_without_compile"),
        supports_target=_is_cutedsl_target,
    ),
    override=True,
)
