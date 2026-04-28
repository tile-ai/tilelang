from __future__ import annotations

from tvm.target import Target

from tilelang.backend.common.target import is_cuda_target, is_cutedsl_target


def is_tilelang_cuda_target(target: Target) -> bool:
    return is_cuda_target(target) and not is_cutedsl_target(target)

