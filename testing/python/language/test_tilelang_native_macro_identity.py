"""Macro expansion must not mutate the IR definitions supplied by its caller."""

import numpy as np
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.testing.packed import compile_packed_function

import tilelang.language as T


def test_nested_macro_preserves_captured_input_function():
    thread = ir.Var("threadIdx_x", ir.PrimType("int32"))
    owner = tirx.Function([thread], ir.SeqStmt([ir.Evaluate(thread)]))
    before = ir.save_json(owner)

    @T.macro
    def inner(thread_binding):
        return thread_binding + 1

    @T.macro
    def outer(thread_argument):
        return inner(thread_argument)

    @T.prim_func
    def lowered(A: T.Tensor((1,), "int32")):
        for _unused in T.serial(1):
            A[0] = outer(thread)

    assert ir.save_json(owner) == before
    assert str(thread.name) == "threadIdx_x"
    lowered = tirx.Function([lowered.params[0], thread], lowered.body, lowered.ret_type, lowered.attrs, lowered.loc)
    assert tirx.analysis.check_well_formed(lowered)
    compiled = compile_packed_function(lowered, tilelang_owned=False)
    output = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"), tvm_tirx.cpu())
    compiled(output, 37)
    np.testing.assert_array_equal(output.numpy(), [38])


def test_macro_reference_keeps_caller_index_definition():
    index = ir.Var("original_index", ir.PrimType("int32"))
    owner = tirx.Function([index], ir.SeqStmt([ir.Evaluate(index)]))
    before = ir.save_json(owner)

    @T.macro
    def update(value: T.Ref):
        value = value + 2

    @T.prim_func
    def lowered(A: T.Tensor((4,), "int32")):
        update(A[index])

    assert ir.save_json(owner) == before
    lowered = tirx.Function([lowered.params[0], index], lowered.body, lowered.ret_type, lowered.attrs, lowered.loc)
    assert tirx.analysis.check_well_formed(lowered)
    compiled = compile_packed_function(lowered, tilelang_owned=False)
    output = tvm_tirx.runtime.tensor(np.arange(4, dtype="int32"), tvm_tirx.cpu())
    compiled(output, 2)
    np.testing.assert_array_equal(output.numpy(), [0, 1, 4, 3])
