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
    elif kind == "assert":
        assertion = tvm.get_global_func("tirx.AssertStmt")(condition, tirx.StringImm("IndexError"), [], None)
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
    if kind == "assert":
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
    ["load", "store", "likely", "assert", "block", "while", "widening_cast", "if_then_else", "select", "load_predicate", "store_predicate"],
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
@pytest.mark.parametrize("kind,guard", [("load", "upper"), ("load", "lower"), ("widening_cast", "upper")])
def test_overflowing_guard_rejects_wide_address(pass_name, kind, guard):
    before, i, j = _guarded_access(kind, guard=guard)
    after = _apply(before, pass_name)
    executable = _compile_predicate(_condition(after, "load"), [i, j])
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


if __name__ == "__main__":
    tilelang.testing.main()
