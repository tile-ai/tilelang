"""Native CPU launch metadata and ordered serial atomic semantics."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.cpu.language as T
from tilelang.testing.packed import compile_packed_function


def walk(value, kind):
    nodes = []
    tvm_ffi.structural_walk(value, (kind, nodes.append), order="pre")
    return nodes


def cpu_pass(name, module):
    return tvm_ffi.get_global_func("tl.cpu.transform." + name)()(module)


def lower_launch(function, parallel=False):
    module = ir.IRModule.from_expr(function)
    with ir.transform.PassContext(config={"tl.cpu_parallel": parallel}):
        module = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, False, None, ["cluster_dims"], None)(module)
        module = cpu_pass("LowerCPUKernelLaunch", module)
    assert tirx.analysis.check_well_formed(module)
    return module


@pytest.mark.parametrize("parallel", [False, True])
def test_launch_siblings_keep_own_thread_count_and_grid_identity(parallel):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32")):
        with T.Kernel(2, 2, cpu_num_threads=3) as (bx, by):
            A[bx * 2 + by] = 1
        with T.Kernel(2, 2, cpu_num_threads=5) as (bx, by):
            A[4 + bx * 2 + by] = 2

    result = lower_launch(kernel, parallel)
    loops = walk(result, ir.For)
    assert len(loops) == 4
    assert [loop.annotations.get("tl.cpu_num_threads") for loop in loops] == [3, None, 5, None]
    assert [loop.annotations.get("tl.cpu_grid_dim") for loop in loops] == ([0, 1, 0, 1] if parallel else [None] * 4)
    output = tvm_tirx.runtime.tensor(np.zeros(8, dtype="int32"))
    compile_packed_function(result, c_codegen="compute")(output)
    np.testing.assert_array_equal(output.numpy(), np.array([1] * 4 + [2] * 4, dtype="int32"))


def test_nested_launch_headers_remain_independent():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32")):
        with T.Kernel(2, cpu_num_threads=3) as bx:  # noqa: SIM117 - nested AST scope is under test
            with T.Kernel(2, cpu_num_threads=7) as by:
                A[bx * 2 + by] = 1

    result = lower_launch(kernel, True)
    loops = walk(result, ir.For)
    assert [loop.annotations["tl.cpu_num_threads"] for loop in loops] == [3, 7]


def test_cpu_launch_default_thread_count_and_pass_order():
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32")):
        with T.Kernel(4) as bx:
            A[bx] = bx

    result = lower_launch(kernel, True)
    assert all("tl.cpu_num_threads" not in loop.annotations for loop in walk(result, ir.For))
    with pytest.raises(ValueError, match="after MaterializeKernelLaunch"):
        cpu_pass("LowerCPUKernelLaunch", ir.IRModule.from_expr(kernel))


def compile_atomics(function):
    module = lower_launch(function)
    marked = cpu_pass("MarkCPUAtomics", module)
    assert next(iter(marked.functions.values())).attrs["tl.cpu_had_atomics"]
    result = cpu_pass("LowerCPUAtomics", marked)
    assert tirx.analysis.check_well_formed(result)
    assert not any(isinstance(call.op, ir.Op) and call.op.name.startswith("tl.atomic") for call in walk(result, ir.Call))
    return compile_packed_function(result, c_codegen="compute")


@pytest.mark.parametrize("operation", ["add", "max", "min"])
def test_scalar_return_previous_preserves_running_values(operation):
    emit = {"add": T.atomic_add, "max": T.atomic_max, "min": T.atomic_min}[operation]

    @T.prim_func
    def kernel(A: T.Tensor((5,), "int32"), B: T.Tensor((1,), "int32"), P: T.Tensor((5,), "int32")):
        for i in T.serial(5):
            P[i] = emit(B[0], A[i], return_prev=True)

    values = np.array([2, -7, 11, -5, 3], dtype="int32")
    expected = []
    running = 4
    combine = {"add": lambda x, y: x + y, "max": max, "min": min}[operation]
    for value in values:
        expected.append(running)
        running = combine(running, int(value))
    result = tvm_tirx.runtime.tensor(np.array([4], dtype="int32"))
    previous = tvm_tirx.runtime.tensor(np.empty(5, dtype="int32"))
    compile_atomics(kernel)(tvm_tirx.runtime.tensor(values), result, previous)
    np.testing.assert_array_equal(previous.numpy(), np.array(expected, dtype="int32"))
    np.testing.assert_array_equal(result.numpy(), np.array([running], dtype="int32"))


def test_return_previous_keeps_earlier_sibling_reads():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), P: T.Tensor((2,), "int32")):
        P[0] = B[0] + T.atomic_add(B[0], 1, return_prev=True)
        P[1] = T.atomic_add(B[0], 2, return_prev=True) + T.atomic_add(B[0], 3, return_prev=True)

    result = tvm_tirx.runtime.tensor(np.array([2], dtype="int32"))
    previous = tvm_tirx.runtime.tensor(np.empty(2, dtype="int32"))
    compile_atomics(kernel)(result, previous)
    np.testing.assert_array_equal(previous.numpy(), np.array([4, 8], dtype="int32"))
    np.testing.assert_array_equal(result.numpy(), np.array([8], dtype="int32"))


def test_branch_atomics_do_not_execute_on_untaken_branch():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), P: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            P[i] = T.if_then_else(i % 2 == 0, T.atomic_add(B[0], 1, return_prev=True), T.int32(-1))

    result = tvm_tirx.runtime.tensor(np.array([0], dtype="int32"))
    previous = tvm_tirx.runtime.tensor(np.empty(4, dtype="int32"))
    compile_atomics(kernel)(result, previous)
    np.testing.assert_array_equal(previous.numpy(), np.array([0, -1, 1, -1], dtype="int32"))
    np.testing.assert_array_equal(result.numpy(), np.array([2], dtype="int32"))


def test_while_condition_atomic_executes_on_every_test():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), P: T.Tensor((1,), "int32")):
        P[0] = 0
        while T.atomic_add(B[0], 1, return_prev=True) < 4:
            P[0] = P[0] + 1

    result = tvm_tirx.runtime.tensor(np.array([0], dtype="int32"))
    count = tvm_tirx.runtime.tensor(np.empty(1, dtype="int32"))
    compile_atomics(kernel)(result, count)
    np.testing.assert_array_equal(result.numpy(), np.array([5], dtype="int32"))
    np.testing.assert_array_equal(count.numpy(), np.array([4], dtype="int32"))


def test_atomic_load_store_or_and_ignored_serial_memory_order():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32"), P: T.Tensor((1,), "int32")):
        T.atomic_store(B[0], T.int32(3), memory_order="release")
        T.atomic_or(B[0], T.int32(8))
        P[0] = T.atomic_load(B[0], memory_order="acquire")

    result = tvm_tirx.runtime.tensor(np.array([-1], dtype="int32"))
    read = tvm_tirx.runtime.tensor(np.empty(1, dtype="int32"))
    compile_atomics(kernel)(result, read)
    np.testing.assert_array_equal(result.numpy(), np.array([11], dtype="int32"))
    np.testing.assert_array_equal(read.numpy(), np.array([11], dtype="int32"))


def test_vector_return_previous_remains_explicitly_unsupported():
    @T.prim_func
    def kernel(A: T.Tensor((2,), "float16"), B: T.Tensor((2,), "float16"), P: T.Tensor((2,), "float16")):
        P[0:2] = T.atomic_addx2(B[0:2], A[0:2], return_prev=True)

    with pytest.raises(ValueError, match="return_prev=True is not supported on CPU"):
        compile_atomics(kernel)


def test_void_external_statement_is_preserved_without_a_void_value_binding():
    @T.prim_func
    def kernel(B: T.Tensor((1,), "int32")):
        T.call_extern("void", "external_observe", T.address_of(B[0]))
        T.atomic_add(B[0], 1)

    module = cpu_pass("LowerCPUAtomics", cpu_pass("MarkCPUAtomics", lower_launch(kernel)))
    assert tirx.analysis.check_well_formed(module)
    calls = walk(module, ir.Call)
    assert any(isinstance(call.op, ir.Op) and call.op.name == "tirx.call_extern" for call in calls)
    assert all(binding.var.ty != ir.PrimType("void") for binding in walk(module, ir.Bind))
