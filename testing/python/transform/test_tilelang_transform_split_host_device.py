# Copyright (c) Tile-AI Corporation. All Rights Reserved.
"""Tests for SplitHostDevice pass."""
# ruff: noqa

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm import tir


def run_split_host_device_passes(func: tvm.tir.PrimFunc):
    """Run the necessary passes before and including SplitHostDevice."""
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("cuda", "c"))(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.AnnotateDeviceRegions()(mod)
    mod = tl.transform.SplitHostDevice()(mod)
    return mod


def get_device_func(mod: tvm.IRModule):
    """Get the device kernel function from the module."""
    for gvar, func in mod.functions.items():
        if "kernel" in gvar.name_hint:
            return func
    return None


def get_host_func(mod: tvm.IRModule):
    """Get the host function from the module."""
    for gvar, func in mod.functions.items():
        if "kernel" not in gvar.name_hint:
            return func
    return None


def collect_assume_vars(func: tvm.tir.PrimFunc):
    """Collect all variables used in assume statements."""
    assume_vars = set()
    in_assume = [False]  # Use list to allow mutation in nested function
    assume_nodes = []

    def collect_assumes(stmt):
        if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "tl.assume":
            assume_nodes.append(stmt.node)

    tir.stmt_functor.post_order_visit(func.body, collect_assumes)

    # Now collect variables from assume nodes
    def collect_vars_from_expr(expr):
        if isinstance(expr, tir.Var):
            assume_vars.add(expr)

    for node in assume_nodes:
        tir.stmt_functor.post_order_visit(node, collect_vars_from_expr)

    return assume_vars


def get_var_name(var):
    """Get the name of a Var, handling different TVM versions."""
    if hasattr(var, "name_hint"):
        return var.name_hint
    elif hasattr(var, "name"):
        return var.name
    else:
        # Try to get name from string representation
        return str(var).split(":")[0].strip()


def get_param_by_name(func: tvm.tir.PrimFunc, name: str):
    """Get a parameter by name_hint."""
    for param in func.params:
        if get_var_name(param) == name:
            return param
    return None


@tilelang.testing.requires_cuda
def test_split_host_device_with_user_assume():
    """Test that user-defined assumes are correctly copied to device function
    with proper variable substitution.

    This test verifies that:
    1. Assumes are copied from host to device function
    2. Variables in assumes refer to the device function parameters (not dangling)
    3. Host function correctly calls the kernel with original variables
    """
    n = T.dynamic("n")

    @T.prim_func
    def main(a: T.Tensor[(n,), T.int32]):
        T.assume(n >= 233 and n <= 1000)
        with T.Kernel(1, threads=128):
            for i in T.serial(T.ceildiv(n - 233, 123)):
                a[i] = 1

    mod = run_split_host_device_passes(main)

    # Check that we have both host and device functions
    assert len(mod.functions) == 2, "Expected 2 functions (host and device)"

    device_func = get_device_func(mod)
    host_func = get_host_func(mod)

    assert device_func is not None, "Device function not found"
    assert host_func is not None, "Host function not found"

    # Check that device function has assume statements
    device_str = str(device_func)
    assert "tl.assume" in device_str, "Device function should have assume statements"

    # Check that assume variables are the same objects as function parameters
    # (not dangling variables like n_1)
    assume_vars = collect_assume_vars(device_func)
    param_n = get_param_by_name(device_func, "n")

    assert param_n is not None, "Device function should have parameter 'n'"

    # All 'n' variables in assumes should be the same object as the parameter
    for var in assume_vars:
        if get_var_name(var) == "n":
            assert var.same_as(param_n), (
                f"Assume variable 'n' (id={id(var)}) should be the same object "
                f"as parameter 'n' (id={id(param_n)}). "
                "This indicates a variable substitution bug in SplitHostDevice."
            )


@tilelang.testing.requires_cuda
def test_split_host_device_with_buffer_shape_assume():
    """Test that buffer shape assumes (auto-generated) are correctly handled."""
    n = T.dynamic("n")
    m = T.dynamic("m")

    @T.prim_func
    def main(a: T.Tensor[(n, m), T.float32]):
        with T.Kernel(1, threads=128):
            for i in T.serial(n):
                for j in T.serial(m):
                    a[i, j] = 1.0

    mod = run_split_host_device_passes(main)

    device_func = get_device_func(mod)
    assert device_func is not None

    # Check that assumes exist
    device_str = str(device_func)
    assert "tl.assume" in device_str, "Device function should have assume statements"

    # Check that assume variables match parameters
    assume_vars = collect_assume_vars(device_func)
    param_n = get_param_by_name(device_func, "n")
    param_m = get_param_by_name(device_func, "m")

    for var in assume_vars:
        if get_var_name(var) == "n" and param_n is not None:
            assert var.same_as(param_n), "Assume 'n' should match parameter 'n'"
        elif get_var_name(var) == "m" and param_m is not None:
            assert var.same_as(param_m), "Assume 'm' should match parameter 'm'"


@tilelang.testing.requires_cuda
def test_split_host_device_multiple_assumes():
    """Test with multiple user assumes on the same variable."""
    n = T.dynamic("n")

    @T.prim_func
    def main(a: T.Tensor[(n,), T.int32]):
        T.assume(n > 0)
        T.assume(n < 10000)
        T.assume(n % 128 == 0)
        with T.Kernel(1, threads=128):
            for i in T.serial(n):
                a[i] = i

    mod = run_split_host_device_passes(main)

    device_func = get_device_func(mod)
    assert device_func is not None

    device_str = str(device_func)
    # Should have multiple assume statements
    assert device_str.count("tl.assume") >= 3, "Should have at least 3 assume statements"

    # All assumes should use the parameter, not dangling variables
    assume_vars = collect_assume_vars(device_func)
    param_n = get_param_by_name(device_func, "n")

    assert param_n is not None
    for var in assume_vars:
        if get_var_name(var) == "n":
            assert var.same_as(param_n), "All assume variables should match parameter"


@tilelang.testing.requires_cuda
def test_split_host_device_no_dangling_vars():
    """Verify that no dangling variable declarations (like n_1 = T.int32())
    appear in the device function due to incorrect variable handling.
    """
    n = T.dynamic("n")

    @T.prim_func
    def main(a: T.Tensor[(n,), T.int32]):
        T.assume(n >= 100)
        with T.Kernel(1, threads=128):
            for i in T.serial(n):
                a[i] = 1

    mod = run_split_host_device_passes(main)

    device_func = get_device_func(mod)
    assert device_func is not None

    device_str = str(device_func)

    # Check for common patterns of dangling variables
    # These patterns indicate that ConvertSSA created separate variables
    # for assumes that should have used the function parameters
    import re

    # Look for patterns like "n_1 = T.int32()" which indicate dangling vars
    dangling_pattern = r"\bn_\d+\s*=\s*T\.int32\(\)"
    matches = re.findall(dangling_pattern, device_str)

    # Filter out legitimate uses (like in blocks that might have their own scope)
    # We're specifically looking for dangling declarations at function level
    lines = device_str.split("\n")
    dangling_decls = []
    for line in lines:
        # Check if this is a top-level dangling declaration
        # (not inside a block's T.reads()/T.writes())
        stripped = line.strip()
        if re.match(r"^n_\d+\s*=\s*T\.int32\(\)$", stripped):
            dangling_decls.append(stripped)

    # If assume is immediately followed by a dangling var declaration, that's the bug
    assume_indices = [i for i, line in enumerate(lines) if "tl.assume" in line]
    for idx in assume_indices:
        # Check if line before assume has dangling var
        if idx > 0:
            prev_line = lines[idx - 1].strip()
            if re.match(r"^n_\d+\s*=\s*T\.int32\(\)$", prev_line):
                raise AssertionError(
                    f"Found dangling variable declaration '{prev_line}' before assume. "
                    "This indicates SplitHostDevice did not properly substitute variables."
                )


if __name__ == "__main__":
    tilelang.testing.main()
