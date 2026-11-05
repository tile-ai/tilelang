"""Unswitching obeys alias, zero-trip, SSA and SIMT boundaries."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, runtime, tirx

import tilelang
import tilelang.cpu.language as T
from tilelang.testing.packed import compile_packed_function


def run_pass(function, config=None):
    before = ir.save_json(function)
    with ir.transform.PassContext(config=config or {}):
        result = tilelang.transform.LoopUnswitching()(IRModule({"main": function}))
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    again = tilelang.transform.LoopUnswitching()(result)
    assert tirx.analysis.check_well_formed(again)
    ir.assert_structural_equal(again, result)
    return result


def nodes(module, kind):
    result = []
    tvm_ffi.structural_walk(module, (kind, result.append), order="pre")
    return result


def execute(module, *args):
    executable = compile_packed_function(module, c_codegen="compute")
    return executable(*args)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_dynamic_zero_trip_guards_the_predicate(count):
    @T.prim_func
    def main(out: T.Tensor((4,), "int32"), predicate: T.Tensor((1,), "int32"), n: T.int32):
        for i in T.serial(n):
            gate = T.bind(predicate[0])
            if gate > 0:
                out[i] = i + 7

    module = run_pass(main.with_attr("tirx.noalias", True))
    # The predicate load and its Bind must stay below the n > 0 guard.
    outer = module["main"].body.seq[0]
    assert isinstance(outer, ir.If)
    assert not nodes(outer.condition, ir.TensorLoad)
    assert nodes(outer.then_case, ir.TensorLoad)
    assert any(isinstance(stmt, ir.Bind) for stmt in outer.then_case.seq)
    output = runtime.tensor(np.full(4, -9, dtype="int32"))
    execute(module, output, runtime.tensor(np.ones(1, dtype="int32")), count)
    expected = np.full(4, -9, dtype="int32")
    expected[:count] = np.arange(count) + 7
    np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("exceptions", [False, True])
def test_aliasing_parameters_keep_iteration_dependent_predicate(exceptions):
    @T.prim_func
    def main(out: T.Tensor((4,), "int32"), predicate: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            if predicate[0] > 0:
                out[i] = 0

    function = main.with_attr("tirx.noalias", exceptions)
    if exceptions:
        function = function.with_attr("tl.non_restrict_params", list(main.params))
    module = run_pass(function)
    assert isinstance(module["main"].body.seq[0], ir.For)
    if not exceptions:
        # One tensor serves both parameters; the first write changes all later
        # predicate observations. An unsafely hoisted guard would clear all four.
        shared = runtime.tensor(np.ones(4, dtype="int32"))
        execute(module, shared, shared)
        np.testing.assert_array_equal(shared.numpy(), np.array([0, 1, 1, 1], dtype="int32"))


def test_store_through_a_view_blocks_hoisting():
    storage = ir.Var("storage", tirx.TensorType((4,), "int32"))
    alias = ir.Var("alias", storage.ty)
    i = ir.Var("i", "int32")
    condition = ir.TensorLoad(storage, [0]) > 0
    loop = ir.For(i, 0, 4, 1, ir.ForKind.DEFAULT, [ir.If(condition, [ir.TensorStore(alias, [i], ir.IntImm("int32", 0))], None)])
    function = (
        tirx.Function([storage], [ir.Bind(alias, ir.Call("tirx.decl_tensor", [storage.data], ty_args=[alias.ty])), loop])
        .with_attr("global_symbol", "main")
        .with_attr("tirx.noalias", True)
    )
    module = run_pass(function)
    assert isinstance(module["main"].body.seq[1], ir.For)
    data = runtime.tensor(np.ones(4, dtype="int32"))
    execute(module, data)
    np.testing.assert_array_equal(data.numpy(), np.array([0, 1, 1, 1], dtype="int32"))


@pytest.mark.parametrize("begin,end,step", [(5, -1, -2), (1, 7, 2), (-1, -1, -1)])
@pytest.mark.parametrize("enabled", [0, 1])
def test_signed_steps_and_empty_ranges(begin, end, step, enabled):
    @T.prim_func
    def main(out: T.Tensor((8,), "int32"), flag: T.int32):
        for i in T.serial(begin, end, step):
            if flag > 0:
                out[i] = i + 11

    module = run_pass(main)
    output = runtime.tensor(np.zeros(8, dtype="int32"))
    execute(module, output, enabled)
    expected = np.zeros(8, dtype="int32")
    if enabled:
        for i in range(begin, end, step):
            expected[i] = i + 11
    np.testing.assert_array_equal(output.numpy(), expected)


def test_thread_dependent_outer_bind_is_not_uniform():
    @T.prim_func
    def main(out: T.Tensor((32,), "int32")):
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            alias = T.bind(tx + 1)
            for i in T.serial(4):
                if alias == 1:
                    out[i] = i

    module = run_pass(main)
    # The only If must remain inside the serial loop, below the thread region.
    loop = nodes(module, ir.For)[0]
    assert len(nodes(loop.body, ir.If)) == 1


@pytest.mark.parametrize("enabled", [0, 1])
def test_branch_renewal_updates_variable_types(enabled):
    @T.prim_func
    def main(out: T.Tensor((4,), "int32"), flag: T.int32):
        for i in T.serial(4):
            size = T.bind(i + 1)
            if flag > 0:
                scratch = T.alloc_local((size,), "int32")
                scratch[i] = i + 10
                out[i] = scratch[i]
            else:
                other = T.alloc_local((size,), "int32")
                other[i] = i - 10
                out[i] = other[i]

    config = {"tl.loop_unswitching_allow_non_trivial_else": True}
    module = run_pass(main, config)
    assert isinstance(module["main"].body.seq[0], ir.If)
    definitions = nodes(module, ir.Bind)
    assert len({bind.var.__chandle__() for bind in definitions}) == len(definitions)
    output = runtime.tensor(np.zeros(4, dtype="int32"))
    execute(module, output, enabled)
    expected = np.arange(4, dtype="int32") + (10 if enabled else -10)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_nested_conditional_does_not_speculate_a_load():
    @T.prim_func
    def main(out: T.Tensor((4,), "int32"), predicate: T.Tensor((1,), "int32"), limit: T.int32):
        for i in T.serial(4):
            if i < limit:
                gate = T.bind(predicate[0])
                if gate > 0:
                    out[i] = 1

    module = run_pass(main.with_attr("tirx.noalias", True))
    assert isinstance(module["main"].body.seq[0], ir.For)
    assert not nodes(module["main"].body.seq[0].body.seq[0].condition, ir.TensorLoad)


def test_disable_option_keeps_source_loop():
    @T.prim_func
    def main(out: T.Tensor((4,), "int32"), flag: T.int32):
        for i in T.serial(4):
            if flag > 0:
                out[i] = 1

    with ir.transform.PassContext(config={"tl.disable_loop_unswitching": True}):
        module = tilelang.transform.LoopUnswitching()(IRModule({"main": main}))
    ir.assert_structural_equal(module["main"], main)


def test_raw_negative_step_has_native_less_than_entry():
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    flag, index = ir.Var("flag", "int32"), ir.Var("i", "int32")
    branch = ir.If(flag > 0, [ir.TensorStore(output, [0], 7), ir.Break()], None)
    function = tirx.Function([output, flag], [ir.For(index, 0, 4, -1, ir.ForKind.DEFAULT, [branch])]).with_attr("global_symbol", "main")
    # Both versions execute once and leave through Break before decrementing.
    for module in (IRModule({"main": function}), run_pass(function)):
        data = runtime.tensor(np.array([-1], dtype="int32"))
        execute(module, data, 1)
        np.testing.assert_array_equal(data.numpy(), [7])


def test_raw_negative_step_with_false_entry_does_not_speculate_load():
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    source = ir.Var("predicate", tirx.TensorType((0,), "int32"))
    index = ir.Var("i", "int32")
    loop = ir.For(index, 5, -1, -2, ir.ForKind.DEFAULT, [ir.If(ir.TensorLoad(source, [0]) > 0, [ir.TensorStore(output, [0], 7)], None)])
    function = tirx.Function([output, source], [loop]).with_attr("tirx.noalias", True)
    module = run_pass(function)
    top = module["main"].body.seq[0]
    assert isinstance(top, ir.If)
    assert not nodes(top.condition, ir.TensorLoad)


@pytest.mark.parametrize("exit_type", [ir.Break, ir.Continue])
@pytest.mark.parametrize("aggressive", [False, True])
def test_preceding_control_exit_does_not_speculate_read(exit_type, aggressive):
    output = ir.Var("output", tirx.TensorType((1,), "int32"))
    source = ir.Var("predicate", tirx.TensorType((0,), "int32"))
    index = ir.Var("i", "int32")
    loop = ir.For(
        index,
        0,
        1,
        1,
        ir.ForKind.DEFAULT,
        [
            exit_type(),
            ir.If(ir.TensorLoad(source, [0]) > 0, [ir.TensorStore(output, [0], 7)], None),
        ],
    )
    function = tirx.Function([output, source], [loop]).with_attr("tirx.noalias", True)
    config = {"tl.loop_unswitching_allow_non_trivial_else": aggressive}
    module = run_pass(function, config)
    assert isinstance(module["main"].body.seq[0], ir.For)
