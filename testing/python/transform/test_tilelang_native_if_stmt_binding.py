"""Native guard distribution preserves storage, definitions, and evaluation."""

import numpy as np
import pytest
import torch
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cpu.language as T
from tilelang.testing.packed import compile_packed_function


def integer(value):
    return ir.IntImm("int32", value)


def tensor(name, shape=(2,), scope="global"):
    return ir.Var(name, tirx.TensorType(shape, "int32", scope=scope))


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def conditional(condition, then_case, else_case=None):
    return ir.If(condition, then_case, else_case)


def function(params, body):
    return tirx.Function(params, body, ir.NoneType()).with_attr("global_symbol", "main")


def rewrite(value, inline=True):
    before = ir.save_json(value)
    with ir.transform.PassContext(config={"tl.if_stmt_binding_inline_replayable_binds": inline}):
        after = tilelang.transform.IfStmtBinding()(ir.IRModule({"main": value}))["main"]
    assert ir.save_json(value) == before
    assert tirx.analysis.check_well_formed(after)
    assert not tirx.analysis.undefined_vars(after)
    return after


def flat(body):
    result = []
    for stmt in body.seq:
        result.extend(flat(stmt) if isinstance(stmt, ir.SeqStmt) else [stmt])
    return result


def compile_llvm(value):
    return compile_packed_function(value, c_codegen="compute")


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("initial", [-2, 0, 9])
def test_written_condition_is_evaluated_once(inline, initial):
    data = tensor("A")
    original = function(
        [data],
        [
            conditional(
                ir.TensorLoad(data, [0]) >= 0,
                [
                    ir.TensorStore(data, [0], integer(-1)),
                    ir.TensorStore(data, [1], integer(7)),
                ],
            )
        ],
    )
    after = rewrite(original, inline)
    statements = flat(after.body)
    assert isinstance(statements[0], ir.Bind)
    assert len(statements) == 3
    assert all(node.condition.same_as(statements[0].var) for node in statements[1:])
    actual = tvm_tirx.runtime.tensor(np.array([initial, -99], dtype="int32"))
    compile_llvm(after)(actual)
    np.testing.assert_array_equal(actual.numpy(), [-1, 7] if initial >= 0 else [initial, -99])


def test_pure_scalar_condition_needs_no_snapshot():
    flag, data = ir.Var("flag", "int32"), tensor("A")
    after = rewrite(
        function(
            [flag, data],
            [
                conditional(
                    flag > 0,
                    [
                        ir.TensorStore(data, [0], integer(1)),
                        ir.TensorStore(data, [1], integer(2)),
                    ],
                )
            ],
        )
    )
    assert len(objects(after, ir.Bind)) == 0
    assert len(flat(after.body)) == 2


def test_unannotated_parameter_identities_do_not_prove_non_aliasing():
    source, destination = tensor("A"), tensor("B")
    after = rewrite(
        function(
            [source, destination],
            [
                conditional(
                    ir.TensorLoad(source, [0]) >= 0,
                    [
                        ir.TensorStore(destination, [0], integer(-1)),
                        ir.TensorStore(destination, [1], integer(7)),
                    ],
                )
            ],
        )
    )
    assert isinstance(flat(after.body)[0], ir.Bind)
    actual = tvm_tirx.runtime.tensor(np.array([0, -99], dtype="int32"))
    compile_llvm(after.with_attr("tirx.noalias", False))(actual, actual)
    np.testing.assert_array_equal(actual.numpy(), [-1, 7])


def test_read_bind_keeps_original_value_across_alias_writes():
    flag = ir.Var("flag", "int32")
    source, destination = tensor("A"), tensor("B")
    snapshot = ir.Var("snapshot", "int32")
    after = rewrite(
        function(
            [flag, source, destination],
            [
                conditional(
                    flag > 0,
                    [
                        ir.Bind(snapshot, ir.TensorLoad(source, [0])),
                        ir.TensorStore(destination, [0], integer(-5)),
                        ir.TensorStore(destination, [1], snapshot),
                    ],
                )
            ],
        )
    )
    assert len(objects(after, ir.Bind)) == 1
    assert len(flat(after.body)) == 1
    actual = tvm_tirx.runtime.tensor(np.array([13, 0], dtype="int32"))
    compile_llvm(after.with_attr("tirx.noalias", False))(1, actual, actual)
    np.testing.assert_array_equal(actual.numpy(), [-5, 13])


@pytest.mark.parametrize("inline", [False, True])
def test_type_only_scalar_use_retypes_definition_and_all_uses(inline):
    count, size = ir.Var("count", "int32"), ir.Var("size", "int32")
    scratch, output = tensor("scratch", [size], "local"), tensor("output")
    body = [
        ir.Bind(size, count + 1),
        ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])),
        ir.TensorStore(scratch, [0], integer(7)),
        ir.TensorStore(output, [0], ir.TensorLoad(scratch, [0])),
        ir.TensorStore(output, [1], count),
    ]
    after = rewrite(function([count, output], [conditional(count >= 0, body)]), inline)
    definitions = [node for node in objects(after, ir.Bind) if isinstance(node.var.ty, tirx.TensorType)]
    assert len(definitions) == 1
    rebuilt = definitions[0].var
    loads = [node for node in objects(after, ir.TensorLoad) if node.source.name == "scratch"]
    stores = [node for node in objects(after, ir.TensorStore) if node.dest.name == "scratch"]
    assert all(node.source.same_as(rebuilt) for node in loads)
    assert all(node.dest.same_as(rebuilt) for node in stores)
    if inline:
        assert all(node.name != "size" for node in objects(after, ir.Var))
    output_value = tvm_tirx.runtime.tensor(np.full(2, -99, dtype="int32"))
    compile_llvm(after)(3, output_value)
    np.testing.assert_array_equal(output_value.numpy(), [7, 3])


def test_effectful_scalar_bind_stays_single_and_guarded():
    flag, position = ir.Var("flag", "int32"), ir.Var("position", "int32")
    counter, output = tensor("counter", [1]), tensor("output")
    atomic = ir.Call("tl.atomic_add_ret_elem_op", [ir.Call("tirx.tensor_data_ptr", [counter]), integer(1)])
    after = rewrite(
        function(
            [flag, counter, output],
            [
                conditional(
                    flag > 0,
                    [
                        ir.Bind(position, atomic),
                        ir.TensorStore(output, [0], position),
                        ir.TensorStore(output, [1], position + 1),
                    ],
                )
            ],
        )
    )
    assert len(flat(after.body)) == 1 and isinstance(flat(after.body)[0], ir.If)
    assert len(objects(after, ir.Bind)) == 1
    assert sum(isinstance(node.op, ir.Op) and node.op.name == "tl.atomic_add_ret_elem_op" for node in objects(after, ir.Call)) == 1


def test_pointer_bind_preserves_lexical_scope():
    source, output = tensor("source"), tensor("output")
    pointer = ir.Var("pointer", ir.PtrType(ir.PrimType("int32"), scope="global"))
    view = tensor("view")
    after = rewrite(
        function(
            [source, output],
            [
                conditional(
                    ir.TensorLoad(source, [0]) >= 0,
                    [
                        ir.Bind(pointer, ir.Call("tirx.tensor_data_ptr", [source])),
                        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
                        ir.TensorStore(output, [0], ir.TensorLoad(view, [0])),
                        ir.TensorStore(output, [1], ir.TensorLoad(view, [1])),
                    ],
                )
            ],
        )
    )
    assert len(flat(after.body)) == 1
    assert len(objects(after, ir.Bind)) == 2


def test_else_branch_remains_one_control_flow_choice():
    flag, output = ir.Var("flag", "int32"), tensor("output")
    before = function(
        [flag, output], [conditional(flag > 0, [ir.TensorStore(output, [0], integer(1))], [ir.TensorStore(output, [0], integer(2))])]
    )
    after = rewrite(before)
    assert tvm_ffi.structural_equal(before, after)


def test_repeated_specialization_keeps_bound_identity():
    flag, value, output = ir.Var("flag", "int32"), ir.Var("value", "int32"), tensor("output")
    before = function(
        [flag, output],
        [
            conditional(
                flag > 0,
                [
                    ir.Bind(value, flag + 1),
                    ir.TensorStore(output, [0], value),
                    ir.TensorStore(output, [1], value + 1),
                ],
            )
        ],
    )
    after = rewrite(before)
    again = rewrite(after)
    assert tvm_ffi.structural_equal(after, again)


def test_public_dsl_aliases_execute_after_guard_distribution():
    @T.prim_func
    def main(source: T.Tensor((2,), "int32"), output: T.Tensor((2,), "int32")):
        T.func_attr({"tirx.noalias": False})
        with T.Kernel(1):
            if source[0] >= 0:
                output[0] = -1
                output[1] = 7

    after = rewrite(main)
    target = tvm_tirx.target.Target("c")
    kernel = tilelang.compile(after, target=target, target_host=target, execution_backend="tvm_ffi")
    value = torch.tensor([0, -99], dtype=torch.int32)
    kernel(value, value)
    torch.testing.assert_close(value, torch.tensor([-1, 7], dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("default", [False, True])
def test_compile_alias_default_preserves_explicit_function_choice(default):
    from tilelang.engine.lower import _make_parameter_alias_contract_explicit

    plain = function([], [])
    explicit = plain.with_attr("tirx.noalias", not default)
    original = ir.IRModule({"plain": plain, "explicit": explicit}, attrs={"keep": 17})
    with ir.transform.PassContext(config={"tirx.noalias": default}):
        result = _make_parameter_alias_contract_explicit(original)
    assert result["plain"].attrs["tirx.noalias"] is default
    assert result["explicit"].attrs["tirx.noalias"] is (not default)
    assert "tirx.noalias" not in original["plain"].attrs
    assert result.attrs["keep"] == 17
