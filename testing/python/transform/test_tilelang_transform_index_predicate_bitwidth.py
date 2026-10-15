import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _nodes(stmt, kind):
    result = []
    tirx.stmt_functor.post_order_visit(stmt, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _apply(func, pass_name):
    result = getattr(tilelang.transform, pass_name)()(tvm.IRModule({"main": func}))["main"]
    assert not tirx.analysis.undefined_vars(result.body, list(result.params))
    assert tirx.analysis.verify_well_formed(result)
    return result


def _guarded_access(kind, index_scale=1, guard="upper"):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    index = i * j * index_scale
    # Construct the guard separately, so matching must be structural.
    condition = i * j < 16 if guard == "upper" else i * j >= 0
    if kind == "widening_cast":
        condition = tirx.Cast("int64", i * j) < 16
    load = tirx.BufferLoad(buffer, [index])
    if kind == "store":
        body = tirx.IfThenElse(condition, tirx.BufferStore(buffer, 1, [index]), None)
    elif kind == "likely":
        body = tirx.IfThenElse(tirx.call_intrin("bool", "tirx.likely", condition), tirx.Evaluate(load), None)
    elif kind.endswith("assert"):
        assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("IndexError"), [], None)
        if kind == "attr_seq_assert":
            assertion = tirx.SeqStmt([tirx.Evaluate(0), assertion])
        if kind != "assert":
            assertion = tirx.AttrStmt(0, "lexical_alloc_scope", 1, assertion)
        if kind == "nested_attr_assert":
            assertion = tirx.AttrStmt(0, "lexical_alloc_scope", 1, assertion)
        body = tirx.SeqStmt([assertion, tirx.Evaluate(load)])
    elif kind == "block":
        region = tirx.BufferRegion(buffer, [tvm.ir.Range(0, 16)])
        block = tirx.SBlock([], [region], [], "guarded", tirx.Evaluate(load))
        body = tirx.SBlockRealize([], condition, block)
    elif kind == "while":
        body = tirx.While(condition, tirx.Evaluate(load))
    elif kind == "if_then_else":
        body = tirx.Evaluate(tirx.if_then_else(condition, load, 0))
    elif kind == "select":
        body = tirx.Evaluate(tirx.BufferLoad(buffer, [tirx.Select(condition, index, 0)]))
    elif kind == "load_predicate":
        body = tirx.Evaluate(tirx.BufferLoad(buffer, [index], predicate=condition))
    elif kind == "store_predicate":
        body = tirx.BufferStore(buffer, 1, [index], predicate=condition)
    elif kind == "access_ptr":
        ptr = tirx.call_intrin("handle", "tirx.tvm_access_ptr", tirx.type_annotation("int32"), buffer.data, index, 1, 1)
        body = tirx.IfThenElse(condition, tirx.Evaluate(ptr), None)
    else:
        body = tirx.IfThenElse(condition, tirx.Evaluate(load), None)
    func = tirx.PrimFunc([buffer.data, i, j], tirx.SeqStmt([tirx.DeclBuffer(buffer), body]), buffer_map={buffer.data: buffer})
    return func, i, j


def _condition(func, kind):
    if kind.endswith("assert"):
        return _nodes(func.body, tirx.AssertStmt)[0].condition
    if kind == "block":
        return _nodes(func.body, tirx.SBlockRealize)[0].predicate
    if kind == "while":
        return _nodes(func.body, tirx.While)[0].condition
    if kind == "if_then_else":
        return next(call.args[0] for call in _nodes(func.body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.if_then_else")))
    if kind == "select":
        return _nodes(func.body, tirx.Select)[0].condition
    if kind == "load_predicate":
        return _nodes(func.body, tirx.BufferLoad)[0].predicate
    if kind == "store_predicate":
        return _nodes(func.body, tirx.BufferStore)[0].predicate
    return _nodes(func.body, tirx.IfThenElse)[0].condition


def _compile_predicate(condition, params):
    # Execute only the predicate; a regression must not dereference a huge index.
    body = tirx.Evaluate(tirx.ret(tirx.Cast("int64", condition)))
    func = tirx.PrimFunc(params, body, ret_type=tvm.ir.PrimType("int64")).with_attr("global_symbol", "main")
    return tvm.compile(func, target="llvm")


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize(
    "kind",
    [
        "load",
        "store",
        "likely",
        "assert",
        "attr_assert",
        "nested_attr_assert",
        "attr_seq_assert",
        "block",
        "while",
        "widening_cast",
        "if_then_else",
        "select",
        "load_predicate",
        "store_predicate",
    ],
)
def test_index_promotion_updates_matching_guard_arithmetic(pass_name, kind):
    before, i, j = _guarded_access(kind)
    after = _apply(before, pass_name)
    condition = _condition(after, kind)
    multiplications = _nodes(condition, tirx.Mul)
    assert multiplications
    assert all(expr.dtype == "int64" for expr in multiplications)
    assert after.params[1].same_as(i) and after.params[2].same_as(j)
    access = _nodes(after.body, tirx.BufferStore if kind.startswith("store") else tirx.BufferLoad)[0]
    assert access.indices[0].dtype == "int64"
    if isinstance(access, tirx.BufferLoad):
        assert access.dtype == "int32"


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize("kind,guard", [("load", "upper"), ("load", "lower"), ("widening_cast", "upper"), ("attr_assert", "upper")])
def test_overflowing_guard_rejects_wide_address(pass_name, kind, guard):
    before, i, j = _guarded_access(kind, guard=guard)
    after = _apply(before, pass_name)
    executable = _compile_predicate(_condition(after, kind), [i, j])
    assert executable(1, 4) == 1
    assert executable(4, 4) == (guard == "lower")
    assert executable(1 << 30, 4) == (guard == "lower")
    assert executable(-(1 << 30), 4) == (guard == "upper")


def test_flattened_offset_updates_guard_of_index_subexpression():
    before, _, _ = _guarded_access("load", index_scale=4)
    after = _apply(before, "FlattenBuffer")
    assert all(expr.dtype == "int64" for expr in _nodes(_condition(after, "load"), tirx.Mul))


def test_access_ptr_offset_promotion_updates_guard():
    before, _, _ = _guarded_access("access_ptr")
    after = _apply(before, "FlattenBuffer")
    condition = _condition(after, "access_ptr")
    assert all(expr.dtype == "int64" for expr in _nodes(condition, tirx.Mul))
    call = next(call for call in _nodes(after.body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.tvm_access_ptr")))
    assert call.args[2].dtype == "int64"
    assert call.args[3].dtype == call.args[4].dtype == "int32"


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_index_promotion_preserves_data_and_predicate_value_boundaries(pass_name):
    i, j, other = [tirx.Var(name, "int32") for name in ("i", "j", "other")]
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    product = i * j
    conditions = [
        tirx.Cast("int32", product) < 16,
        tirx.Cast("int64", tirx.Cast("int32", product)) < 16,
        tirx.call_extern("int32", "check_i32", product) < 16,
        other * j < 16,
    ]
    statements = [tirx.IfThenElse(condition, tirx.BufferStore(buffer, product, [product]), None) for condition in conditions]
    func = tirx.PrimFunc([buffer.data, i, j, other], tirx.SeqStmt([tirx.DeclBuffer(buffer), *statements]), buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    for stmt, condition in zip(_nodes(after.body, tirx.IfThenElse), conditions):
        tvm.ir.assert_structural_equal(stmt.condition, condition)
        tvm.ir.assert_structural_equal(stmt.then_case.value, product)


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_safe_index_and_guard_remain_int32(pass_name):
    i = tirx.Var("i", "int32")
    buffer = tirx.decl_buffer((32,), "int32", name="A")
    condition = i * 4 < 16
    body = tirx.IfThenElse(condition, tirx.Evaluate(tirx.BufferLoad(buffer, [i * 4])), None)
    body = tirx.For(i, 0, 8, tirx.ForKind.SERIAL, body)
    func = tirx.PrimFunc([buffer.data], tirx.SeqStmt([tirx.DeclBuffer(buffer), body]), buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, condition)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int32"


@pytest.mark.parametrize("manual_guard", [False, True])
def test_cpu_lowering_keeps_guards_consistent_with_addresses(manual_guard):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((1,), "float32"), i: T.int32, j: T.int32):
        with T.Kernel(1, threads=1):
            if manual_guard:
                if i * j < 16:
                    B[0] = A[i * j]
            else:
                B[0] = A[i * j]

    target = tvm.target.Target("c")
    with target:
        artifact = tilelang.lower(main, target=target, target_host="c")
    multiplications = [expr for func in artifact.device_mod.functions.values() for expr in _nodes(func.body, tirx.Mul)]
    assert multiplications
    assert all(expr.dtype == "int64" for expr in multiplications)


def _data_condition_program(selector):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    flag = tirx.Var("flag", "bool")
    source = tirx.decl_buffer((16,), "int32", name="A")
    output = tirx.decl_buffer((2,), "int32", name="B")
    access = tirx.IfThenElse(flag, tirx.BufferStore(output, tirx.BufferLoad(source, [i * j]), [1]), None)
    condition = i * j < 16
    if selector == "statement":
        data = tirx.IfThenElse(condition, tirx.BufferStore(output, 7, [0]), tirx.BufferStore(output, 9, [0]))
        expected = condition
    else:
        expected = tirx.Select(condition, 7, 9) if selector == "select" else tirx.if_then_else(condition, 7, 9)
        data = tirx.BufferStore(output, expected, [0])
    body = tirx.SeqStmt([tirx.DeclBuffer(source), tirx.DeclBuffer(output), access, data])
    return tirx.PrimFunc([source.data, output.data, i, j, flag], body, buffer_map={source.data: source, output.data: output}), expected


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize("selector", ["select", "if_then_else", "statement"])
def test_unrelated_data_conditions_are_preserved(pass_name, selector):
    before, expected = _data_condition_program(selector)
    after = _apply(before, pass_name)
    data = after.body.seq[-1]
    actual = data.condition if selector == "statement" else data.value
    tvm.ir.assert_structural_equal(actual, expected)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_unrelated_data_select_keeps_int32_result(pass_name):
    before, _ = _data_condition_program("select")
    after = _apply(before, pass_name)
    executable = _compile_predicate(after.body.seq[-1].value, [before.params[2], before.params[3]])
    assert executable(1 << 30, 4) == 7


def _fixed_boundary_condition(boundary, selector="select"):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    condition = i * j < 16
    value = tirx.Select(condition, 1, -(1 << 31)) if selector == "select" else tirx.if_then_else(condition, 1, -(1 << 31))
    if boundary == "clz":
        return tirx.call_intrin("int32", "tirx.clz", value) < 16, i, j
    if boundary == "cast":
        return tirx.Cast("int32", value) < 0, i, j
    if boundary == "selection":
        return value < 0, i, j
    if boundary == "conditional_value":
        return tirx.if_then_else(condition, 1, -(1 << 31)) < 0, i, j
    var = tirx.Var("v", "int32")
    return tirx.Let(var, value, var < 0), i, j


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize("boundary", ["clz", "cast", "let", "selection", "conditional_value"])
def test_fixed_value_boundaries_keep_nested_data_conditions(pass_name, boundary):
    condition, i, j = _fixed_boundary_condition(boundary)
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    body = tirx.IfThenElse(condition, tirx.Evaluate(tirx.BufferLoad(buffer, [i * j])), None)
    func = tirx.PrimFunc([buffer.data, i, j], tirx.SeqStmt([tirx.DeclBuffer(buffer), body]), buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.IfThenElse)[0].condition, condition)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize("selector", ["select", "if_then_else"])
def test_clz_guard_does_not_open_after_index_promotion(pass_name, selector):
    condition, i, j = _fixed_boundary_condition("clz", selector)
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    body = tirx.IfThenElse(condition, tirx.Evaluate(tirx.BufferLoad(buffer, [i * j])), None)
    func = tirx.PrimFunc([buffer.data, i, j], tirx.SeqStmt([tirx.DeclBuffer(buffer), body]), buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    executable = _compile_predicate(_nodes(after.body, tirx.IfThenElse)[0].condition, [i, j])
    assert executable(1 << 30, 4) == 0


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_accesses_before_assert_do_not_widen_it(pass_name, wrapped):
    before, i, j = _guarded_access("load")
    condition = i * j < 16
    assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("ValueError"), [], None)
    if wrapped:
        assertion = tirx.AttrStmt(0, "lexical_alloc_scope", 1, assertion)
    before = before.with_body(tirx.SeqStmt([*before.body.seq, assertion]))
    after = _apply(before, pass_name)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.AssertStmt)[0].condition, condition)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_wrapped_assert_preserves_data_conditions_and_value_boundaries(pass_name):
    condition, i, j = _fixed_boundary_condition("clz")
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    data = tirx.Select(i * j < 16, 7, 9)
    assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("ValueError"), [], None)
    wrapped = tirx.AttrStmt(0, "lexical_alloc_scope", 1, tirx.SeqStmt([tirx.Evaluate(data), assertion]))
    body = tirx.SeqStmt([tirx.DeclBuffer(buffer), wrapped, tirx.Evaluate(tirx.BufferLoad(buffer, [i * j]))])
    func = tirx.PrimFunc([buffer.data, i, j], body, buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.AssertStmt)[0].condition, condition)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.Select)[0], data)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


@pytest.mark.parametrize("wrapper", ["if", "for", "while", "thread_extent", "virtual_thread"])
@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_conditional_assert_does_not_guard_access_after_wrapper(pass_name, wrapper):
    before, i, j = _guarded_access("attr_assert")
    assertion = before.body.seq[1]
    flag = tirx.Var("flag", "bool")
    if wrapper == "if":
        assertion = tirx.IfThenElse(flag, assertion, None)
    elif wrapper == "for":
        assertion = tirx.For(tirx.Var("k", "int32"), 0, 0, tirx.ForKind.SERIAL, assertion)
    elif wrapper == "while":
        assertion = tirx.While(flag, assertion)
    else:
        thread = tirx.IterVar(None, tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
        assertion = tirx.AttrStmt(thread, wrapper, 0, assertion)
    body = tirx.SeqStmt([before.body.seq[0], assertion, before.body.seq[-1]])
    before = tirx.PrimFunc([*before.params, flag], body, buffer_map=before.buffer_map)
    after = _apply(before, pass_name)
    tvm.ir.assert_structural_equal(_nodes(after.body, tirx.AssertStmt)[0].condition, i * j < 16)
    assert _nodes(after.body, tirx.BufferLoad)[0].indices[0].dtype == "int64"


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_wrapped_sequence_assert_combines_local_and_following_accesses(pass_name):
    i, j, k = [tirx.Var(name, "int32") for name in ("i", "j", "k")]
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    condition = tirx.And(i * j < 16, k * j < 16)
    assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("IndexError"), [], None)
    inner = tirx.SeqStmt([assertion, tirx.Evaluate(tirx.BufferLoad(buffer, [i * j]))])
    wrapped = tirx.AttrStmt(0, "lexical_alloc_scope", 1, inner)
    body = tirx.SeqStmt([tirx.DeclBuffer(buffer), wrapped, tirx.Evaluate(tirx.BufferLoad(buffer, [k * j]))])
    before = tirx.PrimFunc([buffer.data, i, j, k], body, buffer_map={buffer.data: buffer})
    after = _apply(before, pass_name)
    multiplications = _nodes(_nodes(after.body, tirx.AssertStmt)[0].condition, tirx.Mul)
    assert len(multiplications) == 2 and all(expr.dtype == "int64" for expr in multiplications)


def test_cpu_lowering_keeps_wrapped_assert_consistent_with_address():
    @T.prim_func
    def main(A: T.Tensor((16,), "int32"), B: T.Tensor((1,), "int32"), i: T.int32, j: T.int32):
        with T.Kernel(1, threads=1):
            with T.attr(0, "lexical_alloc_scope", 1):
                assert i * j < 16, "index exceeds buffer size"
            B[0] = A[i * j]

    target = tvm.target.Target("c")
    with target:
        artifact = tilelang.lower(main, target=target, target_host="c")
    functions = list(artifact.device_mod.functions.values())
    assertions = [stmt for func in functions for stmt in _nodes(func.body, tirx.AssertStmt)]
    assert assertions
    for assertion in assertions:
        multiplications = _nodes(assertion.condition, tirx.Mul)
        assert multiplications and all(expr.dtype == "int64" for expr in multiplications)
    addresses = [load.indices[0] for func in functions for load in _nodes(func.body, tirx.BufferLoad)]
    assert addresses and all(index.dtype == "int64" for index in addresses)


@pytest.mark.parametrize("pass_name", ["ConfigIndexBitwidth", "FlattenBuffer"])
def test_memory_guard_inside_fixed_abi_argument_stays_consistent(pass_name):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    buffer = tirx.decl_buffer((16,), "int32", name="A")
    value = tirx.if_then_else(i * j < 16, tirx.BufferLoad(buffer, [i * j]), 1)
    body = tirx.Evaluate(tirx.call_intrin("int32", "tirx.clz", value))
    func = tirx.PrimFunc([buffer.data, i, j], tirx.SeqStmt([tirx.DeclBuffer(buffer), body]), buffer_map={buffer.data: buffer})
    after = _apply(func, pass_name)
    conditional = next(call for call in _nodes(after.body, tirx.Call) if call.op.same_as(tirx.op.Op.get("tirx.if_then_else")))
    assert all(expr.dtype == "int64" for expr in _nodes(conditional.args[0], tirx.Mul))
    assert conditional.dtype == "int32"
    assert _nodes(after.body, tirx.BufferLoad)[0].dtype == "int32"


@tilelang.testing.requires_llvm
def test_cpu_lowering_keeps_unrelated_data_select_int32():
    import torch

    @T.prim_func
    def main(A: T.Tensor((16,), "int32"), B: T.Tensor((2,), "int32"), i: T.int32, j: T.int32, flag: T.bool):
        with T.Kernel(1, threads=1):
            if flag:
                B[1] = A[i * j]
            B[0] = T.Select(i * j < 16, 7, 9)

    kernel = tilelang.compile(main, target="llvm", execution_backend="tvm_ffi")
    source = torch.arange(16, dtype=torch.int32)
    output = torch.full((2,), -1, dtype=torch.int32)
    kernel(source, output, 1 << 30, 4, False)
    assert output.tolist() == [7, -1]


if __name__ == "__main__":
    tilelang.testing.main()
