# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from __future__ import annotations

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
    pattern = r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?\w+"
    for line in source.split("\n"):
        if annotation in line:
            matched = re.findall(pattern, line)
            if len(matched) >= 1:
                return source.index(matched[0] + "(")
    raise ValueError("No global kernel found in the source code")


def match_declare_kernel_cpu(source: str, annotation: str = "int32_t") -> int:
    pattern = r"int32_t\s+\w+"
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


def is_cpu_target(target: Target) -> bool:
    return target.kind.name in ["c"]


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


def pythonic_expr(expr: tvm.tir.PrimExpr) -> str:
    if not isinstance(expr, tvm.tir.PrimExpr):
        return str(expr)
    python_str = ""
    node_to_str_map = {}  # Stores string representation for each node

    def _pythonic_visitor(node):
        if isinstance(node, tvm.tir.Var):
            s = node.name
        elif isinstance(node, (tvm.tir.IntImm, tvm.tir.FloatImm)):
            # Integer constant: use value directly (ignore type)
            s = str(node.value)
        elif isinstance(node, tvm.tir.Cast):
            # Type cast: represent as (type)value
            dtype_map = {"int64": "int64_t", "int32": "int32_t", "int8": "int8_t"}
            dtype = dtype_map.get(str(node.dtype), str(node.dtype))
            value_str = node_to_str_map.get(node.value, str(node.value))
            s = f"({dtype}){value_str}"
        elif isinstance(node, tvm.tir.Mul):
            # Multiplication: format as 'left * right'
            a_str = node_to_str_map.get(node.a, str(node.a))
            b_str = node_to_str_map.get(node.b, str(node.b))
            s = f"{a_str} * {b_str}"
        else:
            # Other nodes: use default string representation
            s = str(node)

        # Store current node's string representation
        node_to_str_map[node] = s
        nonlocal python_str
        python_str = s  # Update global string (retain root node in the end)

    # Perform post-order traversal
    tvm.tir.stmt_functor.post_order_visit(expr, _pythonic_visitor)
    return python_str
