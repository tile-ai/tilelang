import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


@pytest.fixture(params=["ConfigIndexBitwidth", "FlattenBuffer"])
def index_pass(request):
    return getattr(tilelang.transform, request.param)()


def _nodes(stmt, kind):
    result = []
    tirx.stmt_functor.post_order_visit(stmt, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _apply(func, index_pass):
    result = index_pass(tvm.IRModule({"main": func}))["main"]
    assert not tirx.analysis.undefined_vars(result.body, list(result.params))
    assert tirx.analysis.verify_well_formed(result)
    return result


def _program(buffers, params, *statements):
    body = tirx.SeqStmt([*(tirx.DeclBuffer(buffer) for buffer in buffers), *statements])
    return tirx.PrimFunc([*(buffer.data for buffer in buffers), *params], body, buffer_map={buffer.data: buffer for buffer in buffers})


def _pointer(buffer, offset, extent):
    return tirx.call_intrin(buffer.data.dtype, "tirx.tvm_access_ptr", tirx.type_annotation(buffer.dtype), buffer.data, offset, extent, 1)


def _lazy_condition(body):
    return next(call.args[0] for call in _nodes(body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.if_then_else")))


def _assert_wide_products(expr):
    products = _nodes(expr, tirx.Mul)
    assert products and all(product.dtype == T.int64 for product in products)


def test_promoted_access_preserves_data_and_control_flow(index_pass):
    i, j, flag = [tirx.Var(name, T.int32) for name in ("i", "j", "flag")]
    source, output = [tirx.decl_buffer((16,), T.int32, name=name) for name in ("A", "B")]
    product = i * j
    check = tirx.call_extern(T.bool, "check_once", flag)
    fallback = tirx.BufferStore(output, tirx.call_extern(T.int32, "fallback"), [0])
    branch = tirx.IfThenElse(
        tirx.And(check, i * j + 1 <= 16),
        tirx.BufferStore(output, source[product] + product, [product]),
        fallback,
    )
    after = _apply(_program([source, output], [i, j, flag], branch), index_pass)
    (actual,) = _nodes(after.body, tirx.IfThenElse)
    index = _nodes(actual, tirx.BufferLoad)[0].indices[0]
    assert index.dtype == T.int64
    _assert_wide_products(index)
    expected = tirx.IfThenElse(
        tirx.And(check, index + 1 <= 16),
        tirx.BufferStore(output, source[index] + product, [index]),
        fallback,
    )
    tvm.ir.assert_structural_equal(actual, expected)
    assert all(param.dtype == T.int32 for param in list(after.params)[2:])


def test_safe_index_and_guard_remain_int32(index_pass):
    i = tirx.Var("i", T.int32)
    buffer = tirx.decl_buffer((32,), T.int32, name="A")
    condition = i * 4 < 16
    body = tirx.For(i, 0, 8, tirx.ForKind.SERIAL, tirx.IfThenElse(condition, tirx.Evaluate(buffer[i * 4]), None))
    after = _apply(_program([buffer], [], body), index_pass)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, condition)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == T.int32


def test_unrelated_and_business_conditions_remain_int32(index_pass):
    i, j = tirx.Var("i", T.int32), tirx.Var("j", T.int32)
    flag = tirx.Var("flag", T.bool)
    source = tirx.decl_buffer((16,), T.int32, name="A")
    output = tirx.decl_buffer((1,), T.int32, name="B")
    data = tirx.Select(i * j < 16, 7, 9)
    business = ((i + j) << 31) < 0
    before = _program(
        [source, output],
        [i, j, flag],
        tirx.IfThenElse(flag, tirx.Evaluate(source[i * j]), None),
        tirx.BufferStore(output, data, [0]),
        tirx.IfThenElse(business, tirx.Evaluate(source[i + j]), None),
    )
    after = _apply(before, index_pass)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.BufferStore)[0].value, data)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[-1].condition, business)
    assert all(load.indices[0].dtype == T.int64 for load in _nodes(after.body, tirx.BufferLoad))


def test_guard_value_boundaries_are_preserved(index_pass):
    i, j = tirx.Var("i", T.int32), tirx.Var("j", T.int32)
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    conditions = [
        tirx.Cast(T.int32, i * j) < 16,
        tirx.call_extern(T.int32, "check_i32", i * j) < 16,
        tirx.call_intrin(T.int32, "tirx.clz", tirx.Select(i * j < 16, 1, -(1 << 31))) < 16,
    ]
    statements = [tirx.IfThenElse(condition, tirx.Evaluate(buffer[i * j]), None) for condition in conditions]
    after = _apply(_program([buffer], [i, j], *statements), index_pass)
    assert len(_nodes(after.body, tirx.IfThenElse)) == len(conditions)
    for branch, condition in zip(_nodes(after.body, tirx.IfThenElse), conditions):
        tvm.ir.assert_structural_equal(branch.condition, condition)
    assert all(load.indices[0].dtype == T.int64 for load in _nodes(after.body, tirx.BufferLoad))


def test_memory_guard_inside_fixed_abi_argument_is_promoted(index_pass):
    i, j = tirx.Var("i", T.int32), tirx.Var("j", T.int32)
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    value = tirx.if_then_else(i * j < 16, buffer[i * j], 1)
    call = tirx.call_intrin(T.int32, "tirx.clz", value)
    after = _apply(_program([buffer], [i, j], tirx.Evaluate(call)), index_pass)
    _assert_wide_products(_lazy_condition(after.body))
    assert _nodes(after.body, tirx.BufferLoad)[0].dtype == T.int32
    assert _nodes(after.body, tirx.Call)[-1].dtype == T.int32


@pytest.mark.parametrize("scope", ["following", "preceding", "conditional"])
def test_wrapped_assert_tracks_only_controlled_accesses(index_pass, scope):
    i, j, k = [tirx.Var(name, T.int32) for name in ("i", "j", "k")]
    flag = tirx.Var("flag", T.bool)
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    condition = tirx.And(i * j < 16, k * j < 16)
    assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("IndexError"), [], None)
    wrapped = tirx.AttrStmt(0, "lexical_alloc_scope", 1, tirx.SeqStmt([assertion, tirx.Evaluate(buffer[i * j])]))
    if scope == "conditional":
        wrapped = tirx.IfThenElse(flag, wrapped, None)
    statements = [wrapped, tirx.Evaluate(buffer[k * j])]
    if scope == "preceding":
        statements.reverse()
    after = _apply(_program([buffer], [i, j, k, flag], *statements), index_pass)
    products = _nodes(_nodes(after.body, tirx.AssertStmt)[0].condition, tirx.Mul)
    assert [product.dtype for product in products] == [T.int64, T.int64 if scope == "following" else T.int32]


def test_shared_guard_keeps_unpromoted_address_arithmetic(index_pass):
    i, j = tirx.Var("i", T.int32), tirx.Var("j", T.int32)
    source = tirx.decl_buffer((16,), T.int32, name="A")
    other = tirx.decl_buffer((256,), T.int32, name="B")
    other_index = tirx.Cast(T.int64, (i * j) << 31) >> 24
    condition = tirx.all(i * j < 16, other_index >= 0, other_index < 256)
    accesses = tirx.SeqStmt([tirx.Evaluate(source[i * j]), tirx.Evaluate(other[other_index])])
    after = _apply(_program([source, other], [i, j], tirx.IfThenElse(condition, accesses, None)), index_pass)
    loads = _nodes(after.body, tirx.BufferLoad)
    assert loads[0].indices[0].dtype == T.int64
    tvm.ir.assert_structural_equal(loads[1].indices[0], other_index)
    expected = tirx.all(loads[0].indices[0] < 16, other_index >= 0, other_index < 256)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, expected)


def test_rebuilt_buffer_preserves_guard_matching_and_identity(index_pass):
    i, j, s = [tirx.Var(name, T.int32) for name in ("i", "j", "s")]
    source = tirx.decl_buffer((32,), T.int32, name="A")
    shape = (tirx.if_then_else(i * j < 16, source[i * j], 16),)
    view = tirx.decl_buffer(shape, T.int32, name="view", data=source.data)
    access = tirx.IfThenElse(view[0] * s < 16, tirx.Evaluate(source[view[0] * s]), None)
    after = _apply(_program([source], [i, j, s], tirx.DeclBuffer(view), access), index_pass)
    rebuilt = _nodes(after.body, tirx.DeclBuffer)[-1].buffer
    guard = _nodes(after.body, tirx.IfThenElse)[0].condition
    assert guard.a.dtype == T.int64
    assert _nodes(guard, tirx.BufferLoad)[0].buffer.same_as(rebuilt)
    _assert_wide_products(rebuilt.shape[0].args[0])


@pytest.mark.parametrize("selected_extent", [False, True])
def test_pointer_range_tracks_offset_but_preserves_extent(selected_extent):
    i, j, n = [tirx.Var(name, T.int32) for name in ("i", "j", "n")]
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    extent = tirx.Select(i * j < 16, 1, (1 << 31) - 1) if selected_extent else n
    pointer = _pointer(buffer, i * j, extent)
    call = tirx.Evaluate(tirx.call_extern(T.int32, "consume", pointer))
    before = _program([buffer], [i, j, n], tirx.IfThenElse(extent + i * j <= 16, call, None))
    after = _apply(before, tilelang.transform.FlattenBuffer())
    pointer = next(call for call in _nodes(after.body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.tvm_access_ptr")))
    assert pointer.args[2].dtype == T.int64
    _assert_wide_products(pointer.args[2])
    tvm.ir.assert_structural_equal(pointer.args[3], extent)
    assert pointer.args[4].dtype == T.int32
    expected = pointer.args[2] + tirx.Cast(T.int64, extent) <= 16
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, expected)


def test_gather_simplification_preserves_inner_guard(index_pass):
    i, j = tirx.Var("i", T.int32), tirx.Var("j", T.int32)
    source, other = [tirx.decl_buffer((16,), T.int32, name=name) for name in ("A", "B")]
    index = tirx.if_then_else(i * j < 16, source[i * j], 0) & 15
    after = _apply(_program([source, other], [i, j], tirx.Evaluate(other[index])), index_pass)
    _assert_wide_products(_lazy_condition(after.body))
    loads = _nodes(after.body, tirx.BufferLoad)
    assert loads[0].indices[0].dtype == T.int64
    assert all(load.dtype == T.int32 for load in loads)


def _conflicting_accesses(value=False):
    i, j, k, l = [tirx.Var(name, T.int32) for name in ("i", "j", "k", "l")]
    source = tirx.decl_buffer((16,), T.int32, name="A")
    output = tirx.decl_buffer((1,), T.int32, name="Out")
    bound = tirx.Var("v", T.int32)
    condition = tirx.Let(bound, 16, i * j + k * l < bound)
    true_value = tirx.call_extern(T.int32, "consume", _pointer(source, i * j, k * l))
    false_value = source[i * j + k * l - 16]
    if value:
        body = tirx.BufferStore(output, tirx.if_then_else(condition, true_value, false_value), [0])
    else:
        body = tirx.IfThenElse(condition, tirx.BufferStore(output, true_value, [0]), tirx.BufferStore(output, false_value, [0]))
    return _program([source, output], [i, j, k, l], body)


@pytest.mark.parametrize("value", [False, True])
def test_conflicting_branch_predicates_are_rejected(value):
    before = _conflicting_accesses(value)
    assert tirx.analysis.verify_well_formed(before)
    with pytest.raises(RuntimeError, match="conflicting index widths"):
        _apply(before, tilelang.transform.FlattenBuffer())


def test_unrelated_conflict_preserves_control_flow_and_bindings():
    before = _conflicting_accesses()
    flag = tirx.Var("flag", T.bool)
    condition = tirx.Let(tirx.Var("v", T.int32), 16, flag)
    branch = before.body.seq[-1]
    body = tirx.SeqStmt([*before.body.seq[:-1], tirx.IfThenElse(condition, branch.then_case, branch.else_case)])
    before = tirx.PrimFunc([*before.params, flag], body, buffer_map=before.buffer_map)
    after = _apply(before, tilelang.transform.FlattenBuffer())
    (branch,) = _nodes(after.body, tirx.IfThenElse)
    tvm.ir.assert_structural_equal(branch.condition, condition)
    assert branch.else_case is not None and len(_nodes(after.body, tirx.BufferStore)) == 2
    assert len(_nodes(after.body, tirx.Let)) == 1


def test_explicit_extent_binding_preserves_declared_integer_semantics():
    i, j, k, l, n = [tirx.Var(name, T.int32) for name in ("i", "j", "k", "l", "n")]
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    pointer = _pointer(buffer, i * j, n)
    accesses = tirx.SeqStmt([tirx.Evaluate(tirx.call_extern(T.int32, "consume", pointer)), tirx.Evaluate(buffer[i * j + n])])
    before = _program([buffer], [i, j, k, l], tirx.Bind(n, k * l), tirx.IfThenElse(i * j + n < 16, accesses, None))
    after = _apply(before, tilelang.transform.FlattenBuffer())
    binding = _nodes(after.body, tirx.Bind)[0]
    tvm.ir.assert_structural_equal(binding.value, k * l)
    assert binding.var.dtype == T.int32
    pointer = next(call for call in _nodes(after.body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.tvm_access_ptr")))
    tvm.ir.assert_structural_equal(pointer.args[3], n)
    end = pointer.args[2] + tirx.Cast(T.int64, n)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, end < 16)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.BufferLoad)[0].indices[0], end)


def test_vector_endpoints_use_promoted_arithmetic(index_pass):
    i, j, s = [tirx.Var(name, T.int32) for name in ("i", "j", "s")]
    buffer = tirx.decl_buffer((16,), T.int32, name="A")
    base = i * j + 7
    endpoints = [(base + 3 * s < 16, s), (s * 3 + base < 16, s), ((3 * s + i * j) + 7 < 16, s), (base - 3 >= 0, -1)]
    statements = [tirx.IfThenElse(guard, tirx.Evaluate(buffer[tirx.Ramp(base, stride, 4)]), None) for guard, stride in endpoints]
    after = _apply(_program([buffer], [i, j, s], *statements), index_pass)
    branches = _nodes(after.body, tirx.IfThenElse)
    assert len(branches) == len(endpoints)
    for branch in branches:
        assert branch.condition.a.dtype == T.int64
        _assert_wide_products(branch.condition)
        assert _nodes(branch, tirx.BufferLoad)[0].indices[0].dtype == T.int64x4


@pytest.mark.skipif(
    tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True) is None,
    reason="CUDA codegen is not enabled",
)
def test_cuda_lowering_keeps_generated_guard_consistent_with_address():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float32), B: T.Tensor((1,), T.float32), i: T.int32, j: T.int32):
        with T.Kernel(1, threads=1):
            B[0] = A[i * j]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        artifact = tilelang.lower(main, target=target, target_host="c")
    (func,) = artifact.device_mod.functions.values()
    index = _nodes(func.body, tirx.BufferLoad)[0].indices[0]
    assert index.dtype == T.int64
    _assert_wide_products(index)
    assert all(param.dtype == T.int32 for param in list(func.params)[-2:])
    expected = tirx.And(index < 16, tirx.IntImm(T.int64, 0) <= index)
    tvm.ir.assert_structural_equal(_lazy_condition(func.body), expected)


if __name__ == "__main__":
    tilelang.testing.main()
