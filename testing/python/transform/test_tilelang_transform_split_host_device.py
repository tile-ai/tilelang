# Copyright (c) Tile-AI Corporation. All Rights Reserved.
"""Tests for SplitHostDevice pass."""
# ruff: noqa

import tvm_tirx
import tvm_ffi
from tvm_tirx import ir
from tilelang.testing.ir import collect_nodes, collect_calls
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tvm_tirx import tirx


def run_split_host_device_passes(func: tirx.Function):
    """Run the necessary passes before and including SplitHostDevice."""
    mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})
    mod = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target("cuda", "c"))(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.transform.InjectAssumes()(mod)
    mod = tl.transform.AnnotateDeviceRegions()(mod)
    mod = tl.transform.SplitHostDevice()(mod)
    return mod


def get_device_func(mod: tvm_tirx.IRModule):
    """Get the device kernel function from the module."""
    for gvar, func in mod.functions.items():
        if "kernel" in gvar.name_hint:
            return func
    return None


def get_host_func(mod: tvm_tirx.IRModule):
    """Get the host function from the module."""
    for gvar, func in mod.functions.items():
        if "kernel" not in gvar.name_hint:
            return func
    return None


def collect_assume_vars(func: tirx.Function):
    """Collect typed variables from native optimizer facts by identity."""
    return {var for condition in collect_assume_conditions(func) for var in collect_nodes(condition, ir.Var)}


def collect_assume_conditions(func: tirx.Function):
    return [call.args[0] for call in collect_calls(func, op="prim.assume")]


def collect_runtime_check_conditions(func: tirx.Function):
    # The native split boundary has already materialized checked assumptions
    # as assertions. Their message preserves the user-assume origin.
    return [
        stmt.condition
        for stmt in collect_nodes(func, ir.AssertStmt)
        if any(isinstance(part, ir.StringImm) and part.value.startswith("Assume:") for part in stmt.message_parts)
    ]


def get_var_name(var):
    """Get the name of a Var, handling different TVM versions."""
    if hasattr(var, "name_hint"):
        return var.name_hint
    elif hasattr(var, "name"):
        return var.name
    else:
        # Try to get name from string representation
        return str(var).split(":")[0].strip()


def get_param_by_name(func: tirx.Function, name: str):
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
    assert collect_runtime_check_conditions(host_func)
    assert not collect_runtime_check_conditions(device_func)

    # Check that device function has assume statements
    device_str = str(device_func)
    assert collect_assume_conditions(device_func), "Device function should have assume statements"

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

    host_func = get_host_func(mod)
    device_func = get_device_func(mod)
    assert host_func is not None
    assert device_func is not None
    assert not collect_runtime_check_conditions(host_func)
    assert not collect_runtime_check_conditions(device_func)

    # Check that assumes exist
    device_str = str(device_func)
    assert collect_assume_conditions(device_func), "Device function should have assume statements"

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
    assert len(collect_assume_conditions(device_func)) >= 3, "Should have at least 3 assume statements"

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

    assert tirx.analysis.check_well_formed(device_func)
    parameter = get_param_by_name(device_func, "n")
    assert parameter is not None
    named = [var for var in collect_assume_vars(device_func) if get_var_name(var) == "n"]
    assert named and all(var.same_as(parameter) for var in named)


@tilelang.testing.requires_cuda
def test_split_host_device_lifts_host_evaluable_device_assume():
    @T.prim_func
    def main(a: T.Tensor[(1,), T.int32], n: T.int32):
        with T.Kernel(1, threads=128):
            T.assume(n % 4 == 0)
            a[0] = n

    mod = run_split_host_device_passes(main)
    host_func = get_host_func(mod)
    device_func = get_device_func(mod)
    assert host_func is not None
    assert device_func is not None

    expected = main.params[1] % 4 == 0
    assert any(tvm_ffi.structural_equal(condition, expected) for condition in collect_assume_conditions(host_func))
    assert any(tvm_ffi.structural_equal(condition, expected, map_free_vars=True) for condition in collect_assume_conditions(device_func))
    assert any(tvm_ffi.structural_equal(condition, expected) for condition in collect_runtime_check_conditions(host_func))
    assert not collect_runtime_check_conditions(device_func)


@tilelang.testing.requires_cuda
def test_split_host_device_keeps_device_local_assume_off_host():
    @T.prim_func
    def main(a: T.Tensor[(1,), T.int32]):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            T.assume(tx < 128)
            if tx == 0:
                a[0] = tx

    mod = run_split_host_device_passes(main)
    host_func = get_host_func(mod)
    device_func = get_device_func(mod)
    assert host_func is not None
    assert device_func is not None

    assert not collect_assume_conditions(host_func)
    assert collect_assume_conditions(device_func)
    assert not collect_runtime_check_conditions(host_func)
    assert not collect_runtime_check_conditions(device_func)


@tilelang.testing.requires_cuda
def test_split_host_device_does_not_lift_conditional_device_assume():
    @T.prim_func
    def main(a: T.Tensor[(1,), T.int32], n: T.int32):
        with T.Kernel(1, threads=128):
            if n % 4 == 0:
                T.assume(n > 0)
                a[0] = n

    mod = run_split_host_device_passes(main)
    host_func = get_host_func(mod)
    device_func = get_device_func(mod)
    assert host_func is not None
    assert device_func is not None

    assert not collect_assume_conditions(host_func)
    assert collect_assume_conditions(device_func)
    assert not collect_runtime_check_conditions(host_func)
    assert not collect_runtime_check_conditions(device_func)


if __name__ == "__main__":
    tilelang.testing.main()
