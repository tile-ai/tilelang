# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
from typing import Union, Optional
from tilelang import tvm as tvm
from tvm import IRModule, tir
from tvm.target import Target
from tilelang.engine.lower import (
    is_device_call,
    determine_target,
    canon_target_host,
)
from tilelang.engine.phase import (
    LowerAndLegalize,
    OptimizeForTarget,
)


def match_global_kernel(source: str) -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    matched = re.findall(pattern, source)
    assert len(matched) >= 1  # may have statement before kernel
    return source.index(matched[0])


def is_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda"


def is_hip_target(target: Target) -> bool:
    return target.kind.name == "hip"


def get_annotated_device_mod(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
) -> "IRModule":

    mod = func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)
    device_mod = tir.transform.Filter(is_device_call)(mod)

    return device_mod
