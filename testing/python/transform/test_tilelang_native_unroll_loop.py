"""Native unrolling preserves loop policy, iteration values and definition identity."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
from tilelang.testing.packed import compile_packed_function


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape=(16,), scope="global"):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope))


def function(parameters, body):
    return tirx.Function(parameters, body, ir.NoneType()).with_attr("global_symbol", "main")


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def rewrite(value, config=None):
    before = ir.save_json(value)
    with ir.transform.PassContext(config=config or {}):
        after = tilelang.transform.UnrollLoop()(ir.IRModule({"main": value}))["main"]
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(after)
    assert not tirx.analysis.undefined_vars(after)
    return after


def run(value, expected):
    executable = compile_packed_function(value, c_codegen="compute")
    result = tvm_tirx.runtime.tensor(np.full(len(expected), -99, dtype="int32"))
    executable(result)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.parametrize("bounds", [(0, 6, 2), (1, 8, 3), (8, 0, -3), (5, 5, 2), (9, 2, 1)])
@pytest.mark.parametrize("explicit", [False, True])
def test_iteration_values_and_empty_loop(bounds, explicit):
    begin, end, step = bounds
    output, index = tensor("A"), ir.Var("i", "int32")
    before = function(
        [output],
        [
            ir.For(
                index,
                integer(begin),
                integer(end),
                integer(step),
                ir.ForKind.UNROLLED,
                [ir.TensorStore(output, [index], index)],
                annotations={"pragma_unroll_explicit": explicit},
            )
        ],
    )
    after = rewrite(before)
    assert len(objects(after, ir.For)) == (0 if explicit else 1)
    # The LLVM compiler's later native pipeline has its own unroller. Execute
    # the expanded version here; inspect the retained loop's step above.
    if explicit:
        expected = np.full(16, -99, "int32")
        # Raw ir.For always tests index < end. The TileLang frontend's
        # SerialForWithStep is a distinct Python-range normalization boundary.
        for value in range(begin, end, step) if begin < end else ():
            expected[value] = value
        run(after, expected)


def test_unsigned_reversed_endpoints_are_empty_before_subtraction():
    output, index = tensor("A"), ir.Var("i", "uint8")
    value = function(
        [output],
        [
            ir.For(
                index,
                integer(254, "uint8"),
                integer(1, "uint8"),
                integer(1, "uint8"),
                ir.ForKind.UNROLLED,
                [ir.TensorStore(output, [0], integer(7))],
                annotations={"pragma_unroll_explicit": True},
            )
        ],
    )
    after = rewrite(value)
    assert not objects(after, ir.TensorStore)
    run(after, np.full(16, -99, "int32"))


def test_narrow_signed_loop_does_not_narrow_the_iteration_ordinal():
    output, index = tensor("A", [200]), ir.Var("i", "int8")
    widened = index.astype("int32")
    value = function(
        [output],
        [
            ir.For(
                index,
                integer(-100, "int8"),
                integer(100, "int8"),
                integer(1, "int8"),
                ir.ForKind.UNROLLED,
                [ir.TensorStore(output, [widened + 100], widened)],
                annotations={"pragma_unroll_explicit": True},
            )
        ],
    )
    after = rewrite(value)
    assert len(objects(after, ir.TensorStore)) == 200
    run(after, np.arange(-100, 100, dtype="int32"))


def test_wrapping_final_increment_is_not_replaced_by_a_finite_sequence():
    index = ir.Var("i", "uint8")
    value = function(
        [],
        [
            ir.For(
                index,
                integer(254, "uint8"),
                integer(255, "uint8"),
                integer(2, "uint8"),
                ir.ForKind.UNROLLED,
                [ir.Evaluate(index)],
                annotations={"pragma_unroll_explicit": True},
            )
        ],
    )
    with pytest.raises(ValueError, match="overflows its index type"):
        rewrite(value)


def test_stationary_raw_loop_with_break_keeps_its_runtime_semantics():
    output, index = tensor("A"), ir.Var("i", "int32")
    value = function(
        [output],
        [
            ir.For(
                index,
                integer(0),
                integer(4),
                integer(0),
                ir.ForKind.DEFAULT,
                [ir.TensorStore(output, [index], integer(7)), ir.Break()],
            )
        ],
    )
    after = rewrite(value, {"tl.UnrollLoop": {"auto_max_step": 32, "explicit_unroll": True}})
    assert len(objects(after, ir.For)) == 1
    assert len(objects(after, ir.Break)) == 1
    run(after, np.array([7] + [-99] * 15, "int32"))


def test_stationary_raw_loop_requires_known_count_for_explicit_expansion():
    index = ir.Var("i", "int32")
    value = function(
        [],
        [
            ir.For(
                index,
                integer(0),
                integer(4),
                integer(0),
                ir.ForKind.UNROLLED,
                [ir.Evaluate(index)],
                annotations={"pragma_unroll_explicit": True},
            )
        ],
    )
    with pytest.raises(ValueError, match="Cannot unroll non-constant loop"):
        rewrite(value)


def test_nonexplicit_dynamic_and_factor_loops_remain():
    output, size, i, j = tensor("A"), ir.Var("size", "int32"), ir.Var("i", "int32"), ir.Var("j", "int32")
    value = function(
        [size, output],
        [
            ir.For(i, integer(0), size, integer(1), ir.ForKind.UNROLLED, [ir.TensorStore(output, [i], i)]),
            ir.For(
                j,
                integer(0),
                integer(8),
                integer(1),
                ir.ForKind.UNROLLED,
                [ir.TensorStore(output, [j], j)],
                annotations={"pragma_unroll_factor": 4},
            ),
        ],
    )
    after = rewrite(value)
    loops = objects(after, ir.For)
    assert len(loops) == 2
    assert loops[0].end.same_as(size)
    assert loops[1].annotations["pragma_unroll_factor"] == 4
    with pytest.raises(ValueError, match="non-constant loop"):
        rewrite(value, {"tl.UnrollLoop": {"explicit_unroll": True}})


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_explicit_expansion_rejects_control_targeting_this_loop(control):
    index = ir.Var("i", "int32")
    value = function(
        [],
        [ir.For(index, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, [control()], annotations={"pragma_unroll_explicit": True})],
    )
    with pytest.raises(ValueError, match="cannot be fully expanded"):
        rewrite(value)


@pytest.mark.parametrize("control", [ir.Break, ir.Continue])
def test_expansion_retains_nested_loop_control_and_freshens_its_var(control):
    outer, inner = ir.Var("i", "int32"), ir.Var("i", "int32")
    nested = ir.For(inner, integer(0), integer(2), integer(1), ir.ForKind.DEFAULT, [control()])
    value = function(
        [], [ir.For(outer, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, [nested], annotations={"pragma_unroll_explicit": True})]
    )
    after = rewrite(value)
    loops = objects(after, ir.For)
    assert len(loops) == 2
    assert not loops[0].var.same_as(loops[1].var)
    assert all(len(objects(loop.body, control)) == 1 for loop in loops)


def test_shape_and_layout_definitions_follow_each_iteration():
    output, index = tensor("A"), ir.Var("i", "int32")
    layout = tirx.TileLayout(tirx.layout.S[(1,) : (1,)] + index)
    view = ir.Var("view", tirx.TensorType([index + 1], "int32", layout=layout))
    pointer = ir.Call("tirx.tensor_data_ptr", [output])
    body = [ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])), ir.TensorStore(view, [0], index + 7)]
    value = function(
        [output],
        [ir.For(index, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, body, annotations={"pragma_unroll_explicit": True})],
    )
    after = rewrite(value)
    bindings, stores = objects(after, ir.Bind), objects(after, ir.TensorStore)
    assert len(bindings) == len(stores) == 2
    assert not bindings[0].var.same_as(bindings[1].var)
    for position in range(2):
        assert stores[position].dest.same_as(bindings[position].var)
        assert all(not var.same_as(index) for var in objects(bindings[position].var.ty, ir.Var))
        assert bindings[position].value.args[0].args[0].same_as(output)
    run(after, np.array([7, 8] + [-99] * 14, "int32"))


def test_scalar_bind_and_alloc_definitions_do_not_cross_iterations():
    output, index, size = tensor("A"), ir.Var("i", "int32"), ir.Var("size", "int32")
    scratch = tensor("scratch", [size], "local")
    body = [
        ir.Bind(size, index + 1),
        ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])),
        ir.TensorStore(scratch, [0], index + 11),
        ir.TensorStore(output, [index], ir.TensorLoad(scratch, [0])),
    ]
    value = function(
        [output],
        [ir.For(index, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, body, annotations={"pragma_unroll_explicit": True})],
    )
    after = rewrite(value)
    bindings = objects(after, ir.Bind)
    assert len(bindings) == 4
    assert not bindings[0].var.same_as(bindings[2].var)
    assert not bindings[1].var.same_as(bindings[3].var)
    assert bindings[1].var.ty.shape[0].same_as(bindings[0].var)
    assert bindings[3].var.ty.shape[0].same_as(bindings[2].var)
    run(after, np.array([11, 12] + [-99] * 14, "int32"))


def test_public_unroll_spelling_and_policy_are_preserved():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        for i in T.unroll(1, 8, step=3, explicit=True):
            A[i] = i
        for j in T.unroll(4, annotations={"pragma_unroll_factor": 2}):
            A[j] = j

    after = rewrite(kernel)
    loops = objects(after, ir.For)
    assert len(loops) == 1 and loops[0].annotations["pragma_unroll_factor"] == 2
    assert len(objects(after, ir.TensorStore)) == 4


def test_pass_context_default_can_be_overridden_per_loop():
    output, i, j = tensor("A"), ir.Var("i", "int32"), ir.Var("j", "int32")
    value = function(
        [output],
        [
            ir.For(
                i,
                integer(0),
                integer(2),
                integer(1),
                ir.ForKind.UNROLLED,
                [ir.TensorStore(output, [i], i)],
                annotations={"pragma_unroll_explicit": False},
            ),
            ir.For(j, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, [ir.TensorStore(output, [j], j)]),
        ],
    )
    after = rewrite(value, {"tl.UnrollLoop": {"explicit_unroll": True}})
    assert len(objects(after, ir.For)) == 1
    assert len(objects(after, ir.TensorStore)) == 3


def test_outer_explicit_annotation_does_not_force_inner_dynamic_loop():
    output, size = tensor("A"), ir.Var("size", "int32")
    outer, inner = ir.Var("i", "int32"), ir.Var("j", "int32")
    nested = ir.For(inner, integer(0), size, integer(1), ir.ForKind.UNROLLED, [ir.TensorStore(output, [inner], outer)])
    value = function(
        [size, output],
        [ir.For(outer, integer(0), integer(2), integer(1), ir.ForKind.UNROLLED, [nested], annotations={"pragma_unroll_explicit": True})],
    )
    after = rewrite(value)
    loops = objects(after, ir.For)
    assert len(loops) == 2 and all(loop.end.same_as(size) for loop in loops)
    assert not loops[0].var.same_as(loops[1].var)
