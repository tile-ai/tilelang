from __future__ import annotations

import re
from typing import Literal, Callable, Any
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


def match_declare_kernel_cutedsl(source: str, annotation: str = "@cute.kernel") -> int:
    # Match decorator followed by function definition across lines
    # \s+ allows any whitespace including newlines between decorator and def
    pattern = r"@cute\.kernel\s+def\s+(\w+)"
    matched = re.search(pattern, source, re.MULTILINE)
    if matched:
        # Find the position of the opening parenthesis after the function name
        # matched.start(1) gives position of function name
        func_name_pos = matched.start(1)
        # Find the '(' after function name
        paren_pos = source.find("(", func_name_pos)
        if paren_pos != -1:
            return paren_pos
    raise ValueError("No global kernel found in the source code")


def extract_python_func_declaration(source: str, func_name: str) -> str:
    """Extract the full Python function declaration from decorator to colon.

    Args:
        source: Source code containing the function
        func_name: Name of the function to extract (can include '(' suffix)

    Returns:
        The function declaration from 'def' to ':', including parameters

    Example:
        For code:
            @cute.kernel
            def kernel(arg1: cute.Tensor, arg2: int):
                ...
        Returns: "def kernel(arg1: cute.Tensor, arg2: int)"
    """
    # Remove '(' suffix if present
    if func_name.endswith("("):
        func_name = func_name[:-1]

    # Match from def to the closing ) followed by :
    # This handles multi-line function signatures
    pattern = rf"def\s+{re.escape(func_name)}\s*\([^)]*\)"
    matched = re.search(pattern, source, re.DOTALL)
    if matched:
        return matched.group(0)

    raise ValueError(f"No function declaration found for {func_name}")


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


def is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" in target.keys


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
        "device": lambda m: tir.transform.Filter(_is_device_call)(m),
        "host": lambda m: tir.transform.Filter(_is_host_call)(m),
        "all": lambda m: (tir.transform.Filter(_is_device_call)(m), tir.transform.Filter(_is_host_call)(m)),
    }

    return dispatch[model_type](mod)


def pythonic_expr(
    expr: tvm.tir.PrimExpr, dtype_map: dict[str, str] | None = None, ignore_cast: bool = False, floor_div_op: str = "/"
) -> str:
    """
    Converts a TVM PrimExpr into a Python-style string, correctly handling operator precedence.

    Args:
        expr: The TVM PrimExpr to convert.
        dtype_map: A dictionary mapping data types to their string representations.
        ignore_cast: Whether to ignore the cast operator and return the string representation of the value without the cast.
        floor_div_op: Operator to use for tvm.tir.FloorDiv. Default '/' preserves prior
                      behavior (suitable for generating C/C++ expressions). For generating
                      Python code where integer division is required (e.g. grid/block),
                      pass '//' explicitly.
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
            if ignore_cast:
                s = value_str
            else:
                type_str = node.dtype if dtype_map is None else dtype_map[node.dtype]
                s = f"({type_str}){value_str}"
            p = PRECEDENCE.get(type(node), ATOMIC_PRECEDENCE)
        elif isinstance(
            node,
            (
                tvm.tir.Mul,
                tvm.tir.FloorDiv,
                tvm.tir.Add,
                tvm.tir.Sub,
                tvm.tir.FloorMod,
                tvm.tir.LT,
                tvm.tir.LE,
                tvm.tir.GT,
                tvm.tir.GE,
                tvm.tir.EQ,
                tvm.tir.NE,
                tvm.tir.And,
                tvm.tir.Or,
            ),
        ):
            op_map = {
                tvm.tir.Mul: "*",
                tvm.tir.FloorDiv: floor_div_op,
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


def maybe_desc_name(name: str, matches: list[str], i: int, desc_name_map: dict[str, str] | None = None) -> bool:
    """
    Check if a parameter name corresponds to a TMA descriptor.

    Args:
        name: The parameter name to check.
        matches: List of all matched parameter names.
        i: Index of the current match.
        desc_name_map: Optional mapping to store descriptor name relationships.

    Returns:
        True if the parameter is a TMA descriptor.
    """
    match = matches[i]
    if not (match == name + "_desc" or match.startswith(name + "_desc_")):
        return False
    desc_decls = []
    if desc_name_map is not None:
        desc_name_map[match] = name
    if i > 0:
        desc_decls.append(matches[i - 1])
    if i < len(matches) - 1:
        desc_decls.append(matches[i + 1])
    return any([decl == "CUtensorMap" for decl in desc_decls])


def parse_function_call_args(
    declaration: str,
    function_args: list[dict[str, str]],
    function_params: list[Any],
    desc_name_map: dict[str, str] | None = None,
    desc_name_var_map: dict[str, tvm.tir.Var] | None = None,
    transform_arg: Callable[[str, str], Any] | None = None,
) -> list[Any]:
    """
    Parse function call arguments from a kernel declaration.

    Args:
        declaration: The kernel function declaration string.
        function_args: List of function argument specifications.
        function_params: List of function parameters from TVM IR.
        desc_name_map: Optional mapping for descriptor names.
        desc_name_var_map: Optional mapping from descriptor names to TVM variables.
        transform_arg: Optional function to transform each argument (name, type) -> result.

    Returns:
        List of parsed call arguments.
    """
    pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
    matches = re.findall(pattern, declaration)
    call_args = []

    for i, match in enumerate(matches):
        for arg in function_args:
            if arg["name"] == match:
                if transform_arg is not None:
                    call_args.append(transform_arg(match, arg["type"]))
                else:
                    call_args.append(match)
            elif maybe_desc_name(arg["name"], matches, i, desc_name_map):
                if transform_arg is not None:
                    call_args.append(transform_arg(match, "None"))
                else:
                    call_args.append(match)
                if desc_name_var_map is not None and function_params is not None:
                    assert len(call_args) <= len(function_params), f"Too many arguments: {len(call_args)} > {len(function_params)}"
                    desc_name_var_map[match] = function_params[len(call_args) - 1]

    return call_args


class TMADescriptorParams:
    """Parsed TMA descriptor parameters."""

    def __init__(self, handle_name: str, dtype: str, tensor_rank: int, global_address: Any, is_img2col: bool = False):
        self.handle_name = handle_name
        self.dtype = dtype
        self.tensor_rank = tensor_rank
        self.global_address = global_address
        self.is_img2col = is_img2col

        # Common fields
        self.global_dim: list[str] = []
        self.global_stride: list[str] = []
        self.element_strides: list[str] = []
        self.interleave: str = ""
        self.swizzle: str = ""
        self.l2_promotion: str = ""
        self.oob_fill: str = ""

        # Tiled-specific fields
        self.box_dim: list[str] = []

        # Im2col-specific fields
        self.lower_corner: list[str] = []
        self.upper_corner: list[str] = []
        self.smem_box_channel: str = ""
        self.smem_box_pixel: str = ""


def parse_tma_descriptor_args(
    tma_descriptor_args: dict[tvm.tir.Var, list[Any]],
    desc_name_map: dict[str, str],
    desc_name_var_map: dict[str, tvm.tir.Var],
    pythonic_expr_func: Callable[[Any], str],
) -> list[TMADescriptorParams]:
    """
    Parse TMA descriptor arguments into structured parameters.

    Args:
        tma_descriptor_args: Dictionary mapping TMA descriptor variables to their arguments.
        desc_name_map: Mapping from descriptor handles to parameter names.
        desc_name_var_map: Mapping from descriptor handles to TVM variables.
        pythonic_expr_func: Function to convert TVM expressions to strings.

    Returns:
        List of parsed TMA descriptor parameters.
    """
    if not tma_descriptor_args:
        return []

    results = []

    for handle_name, _ in desc_name_map.items():
        assert handle_name in desc_name_var_map, f"Handle name {handle_name} not found in desc_name_var_map"
        desc_var = desc_name_var_map[handle_name]

        assert desc_var in tma_descriptor_args, f"TMA descriptor {desc_var} not found in {tma_descriptor_args}"
        args = tma_descriptor_args[desc_var]

        # Skip __tvm_tensormap_create_tiled and second element (like CUDA version)
        if len(args) < 3:
            raise ValueError(f"TMA descriptor args too short: {len(args)} elements, expected at least 3")

        tma_create_str, _, dtype, tensor_rank, global_address, *remaining_args = args

        is_img2col = tma_create_str.value == "__tvm_tensormap_create_im2col"

        # Convert basic fields
        dtype = pythonic_expr_func(dtype)
        tensor_rank = int(pythonic_expr_func(tensor_rank))

        # Validate tensor_rank
        if not isinstance(tensor_rank, int) or tensor_rank <= 0:
            raise ValueError(f"Invalid tensor_rank: {tensor_rank}. Must be a positive integer")

        params = TMADescriptorParams(handle_name, dtype, tensor_rank, global_address, is_img2col)

        if not is_img2col:
            # Tiled mode
            expected_args_len = 4 * tensor_rank + 4
            if len(remaining_args) < expected_args_len:
                raise ValueError(
                    f"Insufficient remaining args: got {len(remaining_args)}, expected {expected_args_len} for tensor_rank {tensor_rank}"
                )

            # Extract dimensions and strides
            params.global_dim = [pythonic_expr_func(i) for i in remaining_args[:tensor_rank]]
            params.global_stride = [pythonic_expr_func(i) for i in remaining_args[tensor_rank : 2 * tensor_rank]]
            params.box_dim = [pythonic_expr_func(i) for i in remaining_args[2 * tensor_rank : 3 * tensor_rank]]
            params.element_strides = [pythonic_expr_func(i) for i in remaining_args[3 * tensor_rank : 4 * tensor_rank]]

            # Extract remaining parameters
            try:
                interleave, swizzle, l2_promotion, oob_fill = remaining_args[4 * tensor_rank : 4 * tensor_rank + 4]
                params.interleave = pythonic_expr_func(interleave)
                params.swizzle = pythonic_expr_func(swizzle)
                params.l2_promotion = pythonic_expr_func(l2_promotion)
                params.oob_fill = pythonic_expr_func(oob_fill)
            except ValueError as e:
                raise ValueError("Failed to unpack the final 4 TMA parameters (interleave, swizzle, l2Promotion, oobFill)") from e
        else:
            # Im2col mode
            expected_args_len = 5 * tensor_rank + 2
            if len(remaining_args) < expected_args_len:
                raise ValueError(
                    f"Insufficient remaining args: got {len(remaining_args)}, expected {expected_args_len} for tensor_rank {tensor_rank}"
                )

            # Extract dimensions and strides
            params.global_dim = [pythonic_expr_func(i) for i in remaining_args[:tensor_rank]]
            params.global_stride = [pythonic_expr_func(i) for i in remaining_args[tensor_rank : 2 * tensor_rank]]
            params.element_strides = [pythonic_expr_func(i) for i in remaining_args[2 * tensor_rank : 3 * tensor_rank]]
            params.lower_corner = [pythonic_expr_func(i) for i in remaining_args[3 * tensor_rank : 4 * tensor_rank - 2]]
            params.upper_corner = [pythonic_expr_func(i) for i in remaining_args[4 * tensor_rank - 2 : 5 * tensor_rank - 4]]

            # Extract remaining parameters
            try:
                smem_box_pixel, smem_box_channel, interleave, swizzle, l2_promotion, oob_fill = remaining_args[
                    5 * tensor_rank - 4 : 5 * tensor_rank + 2
                ]
                params.smem_box_pixel = pythonic_expr_func(smem_box_pixel)
                params.smem_box_channel = pythonic_expr_func(smem_box_channel)
                params.interleave = pythonic_expr_func(interleave)
                params.swizzle = pythonic_expr_func(swizzle)
                params.l2_promotion = pythonic_expr_func(l2_promotion)
                params.oob_fill = pythonic_expr_func(oob_fill)
            except ValueError as e:
                raise ValueError(
                    "Failed to unpack the final 6 TMA parameters "
                    "(smem_box_pixel, smem_box_channel, interleave, swizzle, l2Promotion, oobFill)"
                ) from e

        results.append(params)

    return results
