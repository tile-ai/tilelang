from __future__ import annotations

import re
from typing import Literal
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


def is_metal_target(target: Target) -> bool:
    return target.kind.name == "metal"


def get_annotated_mod(
    func_or_mod: tir.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    model_type: Literal["device", "host", "all"] = "all",
) -> IRModule | tuple[IRModule, IRModule]:

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


def pythonic_expr(expr: tvm.tir.PrimExpr, dtype_map: dict[str, str] | None = None) -> str:
    """
    Converts a TVM PrimExpr into a Python-style string, correctly handling operator precedence.

    Args:
        expr: The TVM PrimExpr to convert.

    Returns:
        A string representation of the expression.
    """
    if not isinstance(expr, tvm.tir.PrimExpr):
        return str(expr)

    # 1. Define operator precedence (higher value means higher precedence)
    # Based on Python's operator precedence
    PRECEDENCE = {
        tvm.tir.Call: 20,  # Includes min, max
        tvm.tir.Cast: 20,  # Treated like a function call
        tvm.tir.Mul: 13,
        tvm.tir.FloorDiv: 13,
        tvm.tir.Div: 13,  # For tvm.tir.Div if it appears
        tvm.tir.FloorMod: 13,
        tvm.tir.Add: 12,
        tvm.tir.Sub: 12,
        tvm.tir.LT: 10,
        tvm.tir.LE: 10,
        tvm.tir.GT: 10,
        tvm.tir.GE: 10,
        tvm.tir.EQ: 10,
        tvm.tir.NE: 10,
        tvm.tir.And: 5,
        tvm.tir.Or: 4,
        # Atoms (Var, IntImm) have the highest precedence implicitly
    }
    # By default, atomic expressions (variables, constants) have the highest precedence
    ATOMIC_PRECEDENCE = 100

    node_to_result_map = {}  # Stores (string, precedence) for each node

    def _visitor(node):
        # 2. Visitor returns (str, precedence) tuple
        if node in node_to_result_map:
            return

        if isinstance(node, tvm.tir.Var):
            s, p = node.name, ATOMIC_PRECEDENCE
        elif isinstance(node, (tvm.tir.IntImm, tvm.tir.FloatImm)):
            s, p = str(node.value), ATOMIC_PRECEDENCE
        elif isinstance(node, tvm.tir.Cast):
            # C-style cast has high precedence
            value_str, _ = node_to_result_map[node.value]
            if dtype_map is None:
                s = f"({node.dtype}){value_str}"
            else:
                s = f"({dtype_map[node.dtype]}){value_str}"
            p = PRECEDENCE.get(type(node), ATOMIC_PRECEDENCE)
        elif isinstance(
                node,
            (tvm.tir.Mul, tvm.tir.FloorDiv, tvm.tir.Add, tvm.tir.Sub, tvm.tir.FloorMod, tvm.tir.LT,
             tvm.tir.LE, tvm.tir.GT, tvm.tir.GE, tvm.tir.EQ, tvm.tir.NE, tvm.tir.And, tvm.tir.Or)):
            op_map = {
                tvm.tir.Mul: "*",
                tvm.tir.FloorDiv: "/",
                tvm.tir.Add: "+",
                tvm.tir.Sub: "-",
                tvm.tir.FloorMod: "%",
                tvm.tir.LT: "<",
                tvm.tir.LE: "<=",
                tvm.tir.GT: ">",
                tvm.tir.GE: ">=",
                tvm.tir.EQ: "==",
                tvm.tir.NE: "!=",
                tvm.tir.And: "and",
                tvm.tir.Or: "or",
            }
            op_str = f" {op_map[type(node)]} "
            my_precedence = PRECEDENCE[type(node)]

            a_str, a_precedence = node_to_result_map[node.a]
            b_str, b_precedence = node_to_result_map[node.b]

            # 3. Add parentheses intelligently
            # Add parentheses if the left operand's precedence is lower than the current operator
            if a_precedence < my_precedence:
                a_str = f"({a_str})"
            # Add parentheses if the right operand's precedence is lower than or equal to the current operator
            # 'Equal' is to handle non-associative operations, e.g., a - (b - c)
            if b_precedence <= my_precedence:
                b_str = f"({b_str})"

            s = f"{a_str}{op_str}{b_str}"
            p = my_precedence
        elif isinstance(node, (tvm.tir.Min, tvm.tir.Max)):
            op_name = "min" if isinstance(node, tvm.tir.Min) else "max"
            a_str, _ = node_to_result_map[node.a]
            b_str, _ = node_to_result_map[node.b]
            s = f"{op_name}({a_str}, {b_str})"
            # Function calls have high precedence
            p = PRECEDENCE.get(tvm.tir.Call, ATOMIC_PRECEDENCE)
        else:
            # Fallback for unhandled expression types
            s, p = str(node), 0

        node_to_result_map[node] = (s, p)

    # Perform post-order traversal
    tvm.tir.stmt_functor.post_order_visit(expr, _visitor)

    return next(iter(node_to_result_map[expr]), "")
