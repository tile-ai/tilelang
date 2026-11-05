"""Adjacent native guards may merge only when their conditions are replayable."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, runtime, tirx

import tilelang
import tilelang.cpu.language as T
import tilelang.ir as tl_ir
from tilelang.testing.packed import compile_packed_function


def merge(function):
    before = ir.save_json(function)
    module = tilelang.transform.MergeIfStmt()(IRModule({"main": function}))
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(module)
    branches = []
    tvm_ffi.structural_walk(module, (ir.If, branches.append), order="pre")
    return module, branches


def execute(module, values, *args):
    executable = compile_packed_function(module, c_codegen="compute")
    array = runtime.tensor(np.array(values, dtype="int32"))
    executable(*args, array)
    return array.numpy()


@pytest.mark.parametrize("flag", [0, 1])
def test_equal_scalar_guards_merge_and_keep_scoped_definitions(flag):
    @T.prim_func
    def main(flag: T.int32, A: T.Tensor((2,), "int32")):
        if flag > 0:
            first = flag + 2
            A[0] = first
        if flag > 0:
            second = flag + 3
            A[1] = second

    module, branches = merge(main)
    assert len(branches) == 1
    np.testing.assert_array_equal(execute(module, [-5, -5], flag), [3, 4] if flag else [-5, -5])


@pytest.mark.parametrize("initial", [0, 1])
def test_guard_load_is_reevaluated_after_first_body_write(initial):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32")):
        if A[0] > 0:
            A[0] = 0
        if A[0] > 0:
            A[1] = 9

    module, branches = merge(main)
    assert len(branches) == 2
    np.testing.assert_array_equal(execute(module, [initial, -5]), [0, -5])


@pytest.mark.parametrize("flag", [0, 1])
def test_else_branch_keeps_independent_control_flow(flag):
    @T.prim_func
    def main(flag: T.int32, A: T.Tensor((2,), "int32")):
        if flag > 0:
            A[0] = 1
        else:
            A[0] = -1
        if flag > 0:
            A[1] = A[0] + 1

    module, branches = merge(main)
    assert len(branches) == 2
    np.testing.assert_array_equal(execute(module, [-5, -5], flag), [1, 2] if flag else [-1, -5])


def test_same_variable_name_does_not_merge_distinct_identities():
    left, right = ir.Var("flag", "int32"), ir.Var("flag", "int32")
    output = ir.Var("output", tirx.TensorType((2,), "int32"))
    function = tirx.Function(
        [left, right, output],
        [
            ir.If(left > 0, [ir.TensorStore(output, [0], T.int32(1))], None),
            ir.If(right > 0, [ir.TensorStore(output, [1], T.int32(2))], None),
        ],
        ir.NoneType(),
    ).with_attr("global_symbol", "main")
    module, branches = merge(function)
    assert len(branches) == 2
    np.testing.assert_array_equal(execute(module, [-5, -5], 1, 0), [1, -5])


def test_first_guard_and_each_body_keep_source_locations():
    flag = ir.Var("flag", "int32")
    output = ir.Var("output", tirx.TensorType((2,), "int32"))
    first, second = tl_ir.make_span("merge.py", 10), tl_ir.make_span("merge.py", 12)
    a = ir.TensorStore(output, [0], T.int32(1), first)
    b = ir.TensorStore(output, [1], T.int32(2), second)
    function = tirx.Function(
        [flag, output],
        [ir.If(flag > 0, [a], None, loc=first), ir.If(flag > 0, [b], None, loc=second)],
    )
    _, branches = merge(function)
    assert len(branches) == 1
    assert branches[0].loc.same_as(first)
    assert branches[0].then_case.seq[0].loc.same_as(first)
    assert branches[0].then_case.seq[1].loc.same_as(second)
