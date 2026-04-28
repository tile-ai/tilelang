from __future__ import annotations

from tvm.target import Target


def target_kind(target: Target) -> str:
    return target.kind.name


def target_keys(target: Target) -> set[str]:
    return {str(key) for key in target.keys}


def is_cuda_target(target: Target) -> bool:
    return target_kind(target) == "cuda"


def is_cutedsl_target(target: Target) -> bool:
    return is_cuda_target(target) and "cutedsl" in target_keys(target)


def is_hip_target(target: Target) -> bool:
    return target_kind(target) == "hip"


def is_metal_target(target: Target) -> bool:
    return target_kind(target) == "metal"


def is_c_target(target: Target) -> bool:
    return target_kind(target) == "c"


def is_llvm_target(target: Target) -> bool:
    return target_kind(target) == "llvm"


def is_webgpu_target(target: Target) -> bool:
    return target_kind(target) == "webgpu"
