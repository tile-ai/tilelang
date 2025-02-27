# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
from typing import Union, Optional, Literal
from tilelang import tvm as tvm
from tvm import IRModule, tir
from tvm.target import Target
from tilelang.engine.lower import (
    get_device_call,
    get_host_call,
    determine_target,
    canon_target_host,
    is_cpu_device_backend,
)
from tilelang.engine.phase import (
    LowerAndLegalize,
    OptimizeForTarget,
)


def match_global_kernel(source: str, annotation: str = "__global__") -> int:
    pattern = r"__global__\s+void\s+[__launch_bounds__\(\d+\)\s+]\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0])
    raise ValueError("No global kernel found in the source code")


def match_declare_kernel(source: str, annotation: str = "__global__") -> int:
    pattern = r"__global__\s+void\s+\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0] + "(")
    raise ValueError("No global kernel found in the source code")


def is_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda"


def is_hip_target(target: Target) -> bool:
    return target.kind.name == "hip"


def get_annotated_mod(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    model_type: Literal["device", "host", "all"] = "all",
) -> Union[IRModule, tuple[IRModule, IRModule]]:

    # Validate model_type early
    if model_type not in {"device", "host", "all"}:
        raise ValueError(f"Invalid model type: {model_type}")

    # Convert PrimFunc to IRModule if needed
    mod = func_or_mod
    if isinstance(func_or_mod, tir.PrimFunc):
        mod = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})

    # Handle target and target_host
    if isinstance(target, str):
        target = determine_target(target)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, target_host))
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Apply transformations
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

    # Define dispatch dictionary for different model types
    dispatch = {
        "device":
            lambda m: tir.transform.Filter(_is_device_call)(m),
        "host":
            lambda m: tir.transform.Filter(_is_host_call)(m),
        "all":
            lambda m: (tir.transform.Filter(_is_device_call)(m), tir.transform.Filter(_is_host_call)
                       (m)),
    }

    return dispatch[model_type](mod)
