"""Raw native For compares i < end even when its increment is negative."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, runtime, tirx

import tilelang
from tilelang.testing.packed import compile_packed_function


def execute(function):
    executable = compile_packed_function(IRModule({"main": function}), c_codegen="compute")
    output = runtime.tensor(np.array([-1], dtype="int32"))
    executable(output, 1)
    return output.numpy()


def test_raw_negative_step_with_break_keeps_first_iteration():
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    flag, index = ir.Var("flag", "int32"), ir.Var("i", "int32")
    branch = ir.If(flag > 0, [ir.TensorStore(output, [0], 7), ir.Break()], None)
    loop = ir.For(index, 0, 4, -1, ir.ForKind.DEFAULT, [branch])
    function = tirx.Function([output, flag], [loop]).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(function)
    # The real native body executes once, then breaks before applying -1.
    original = execute(function)
    np.testing.assert_array_equal(original, [7])
    after = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(after)
    print("original", original.tolist(), "\nafter IR", after.script())
    rewritten = execute(after)
    print("after execution", rewritten.tolist())
    np.testing.assert_array_equal(rewritten, [7])


def test_raw_negative_step_with_false_entry_does_not_speculate_read():
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    source = ir.Var("predicate", tirx.TensorType((0,), "int32"))
    index = ir.Var("i", "int32")
    loop = ir.For(index, 5, -1, -2, ir.ForKind.DEFAULT, [ir.If(ir.TensorLoad(source, [0]) > 0, [ir.TensorStore(output, [0], 7)], None)])
    function = tirx.Function([output, source], [loop]).with_attr("tirx.noalias", True)
    after = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))["main"]
    top = after.body.seq[0]
    print("zero-trip after IR", after.script())
    if isinstance(top, ir.If):
        loads = []
        tvm_ffi.structural_walk(top.condition, (ir.TensorLoad, loads.append), order="pre")
        assert not loads, "Inactive raw loop must not evaluate the predicate read"


@pytest.mark.parametrize("allow_else", [False, True])
def test_continue_before_guard_prevents_speculative_read(allow_else):
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    source = ir.Var("predicate", tirx.TensorType((0,), "int32"))
    index = ir.Var("i", "int32")
    branch = ir.If(ir.TensorLoad(source, [0]) > 0, [ir.TensorStore(output, [0], 7)], None)
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [ir.Continue(), branch])
    function = tirx.Function([output, source], [loop]).with_attr("tirx.noalias", True)
    assert tirx.analysis.check_well_formed(function)
    with ir.transform.PassContext(config={"tl.loop_unswitching_allow_non_trivial_else": allow_else}):
        after = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))["main"]
    print("unreachable guard after IR", after.script())
    assert isinstance(after.body.seq[0], ir.For), "Continue makes the later read unreachable"


@pytest.mark.parametrize("allow_else", [False, True])
def test_nonterminating_prefix_prevents_speculative_pure_guard(allow_else):
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    denominator, index = ir.Var("denominator", "int32"), ir.Var("i", "int32")
    condition = ir.const(1, "int32") // denominator > 0
    branch = ir.If(condition, [ir.TensorStore(output, [0], 7)], None)
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [ir.While(ir.const(True, "bool"), [ir.Evaluate(0)]), branch])
    function = tirx.Function([output, denominator], [loop])
    assert tirx.analysis.check_well_formed(function)
    with ir.transform.PassContext(config={"tl.loop_unswitching_allow_non_trivial_else": allow_else}):
        after = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))["main"]
    print("nonterminating prefix after IR", after.script())
    assert isinstance(after.body.seq[0], ir.For), "The division is unreachable after while True"


@pytest.mark.parametrize("control", ["break", "continue"])
def test_nested_prefix_control_preserves_unreachable_load(control):
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    source = ir.Var("predicate", tirx.TensorType((0,), "int32"))
    index = ir.Var("i", "int32")
    skip = ir.Break() if control == "break" else ir.Continue()
    prefix = ir.If(index >= 0, [skip], None)
    branch = ir.If(ir.TensorLoad(source, [0]) > 0, [ir.TensorStore(output, [0], 7)], None)
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [prefix, branch])
    function = tirx.Function([output, source], [loop]).with_attr("tirx.noalias", True)
    original = ir.save_json(function)
    with ir.transform.PassContext(config={"tl.loop_unswitching_allow_non_trivial_else": True}):
        after = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))["main"]
    assert isinstance(after.body.seq[0], ir.For)
    assert tirx.analysis.check_well_formed(after)
    assert ir.save_json(function) == original
