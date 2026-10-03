import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
import tilelang.language as T


def _collect_tvm_access_ptr_offsets(func: tvm.tirx.PrimFunc):
    offsets = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tirx.tvm_access_ptr":
            offsets.append(node.args[2])

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return offsets


def test_flatten_buffer_promotes_tvm_access_ptr_offset_to_int64():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 30):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 4,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int64"


def test_flatten_buffer_keeps_safe_tvm_access_ptr_offset_int32():
    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 20):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 2,
                    1,
                    1,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    before_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(before_offsets) == 1
    assert str(before_offsets[0].dtype) == "int32"

    mod = tilelang.transform.FlattenBuffer()(mod)
    after_offsets = _collect_tvm_access_ptr_offsets(mod["main"])
    assert len(after_offsets) == 1
    assert str(after_offsets[0].dtype) == "int32"


def _rewrite_index(index, parameters, access_kind, transform):
    tir = tvm.tirx
    buffer = tir.decl_buffer((16,), "int32", name="A")
    if access_kind == "load":
        body = tir.Evaluate(tir.BufferLoad(buffer, [index]))
    elif access_kind == "store":
        body = tir.BufferStore(buffer, tir.const(42, "int32"), [index])
    else:
        body = tir.Evaluate(tir.tvm_access_ptr(tir.type_annotation("int32"), buffer.data, index, 1, 1))
    func = tir.PrimFunc([buffer, *parameters], body)
    mod = transform(tvm.IRModule({"main": func}))
    body = mod["main"].body
    if access_kind == "load":
        return body.value.indices[0]
    if access_kind == "store":
        return body.indices[0]
    return body.value.args[2]


_INDEX_PROMOTION_PATHS = [
    pytest.param("load", tilelang.transform.FlattenBuffer, id="flatten-load"),
    pytest.param("store", tilelang.transform.FlattenBuffer, id="flatten-store"),
    pytest.param("access_ptr", tilelang.transform.FlattenBuffer, id="flatten-access-ptr"),
    pytest.param("load", tilelang.transform.ConfigIndexBitwidth, id="legalize-load"),
    pytest.param("store", tilelang.transform.ConfigIndexBitwidth, id="legalize-store"),
]


@pytest.mark.parametrize("access_kind,make_transform", _INDEX_PROMOTION_PATHS)
@pytest.mark.parametrize("addend", [0, 1])
def test_index_promotion_preserves_explicit_narrowing_cast(access_kind, make_transform, addend):
    # At x=2**32 the original index is addend, not 2**32 + addend.
    # The cast-only case also triggers promotion at the INT32_MAX bound.
    tir = tvm.tirx
    x = tir.Var("x", "int64")
    narrowed = tir.Cast("int32", x)
    index = narrowed if addend == 0 else narrowed + addend
    after = _rewrite_index(index, [x], access_kind, make_transform())
    expected = tir.Cast("int64", narrowed)
    if addend:
        expected += tir.IntImm("int64", addend)
    tvm.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize("access_kind,make_transform", _INDEX_PROMOTION_PATHS)
def test_index_promotion_preserves_nested_signed_conversion(access_kind, make_transform):
    # Dropping the int32 conversion changes values in [2**31, 2**32).
    tir = tvm.tirx
    x = tir.Var("x", "int64")
    converted = tir.Cast("int32", tir.Cast("uint32", x))
    after = _rewrite_index(converted + 1, [x], access_kind, make_transform())
    expected = tir.Cast("int64", converted) + tir.IntImm("int64", 1)
    tvm.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize("access_kind,make_transform", _INDEX_PROMOTION_PATHS)
def test_index_promotion_keeps_bounded_cast_int32(access_kind, make_transform):
    tir = tvm.tirx
    x = tir.Var("x", "int64")
    index = tir.Cast("int32", x % 8) + 1
    after = _rewrite_index(index, [x], access_kind, make_transform())
    assert str(after.dtype) == "int32"
    tvm.ir.assert_structural_equal(after, index)


@pytest.mark.parametrize("access_kind,make_transform", _INDEX_PROMOTION_PATHS)
def test_index_promotion_still_widens_large_arithmetic(access_kind, make_transform):
    tir = tvm.tirx
    x = tir.Var("x", "int32")
    after = _rewrite_index(x * 4, [x], access_kind, make_transform())
    expected = tir.Cast("int64", x) * tir.IntImm("int64", 4)
    tvm.ir.assert_structural_equal(after, expected)


@pytest.mark.parametrize("access_kind,make_transform", _INDEX_PROMOTION_PATHS)
def test_index_promotion_handles_widening_cast(access_kind, make_transform):
    tir = tvm.tirx
    x = tir.Var("x", "int16")
    index = tir.Cast("int32", x) * (1 << 20)
    after = _rewrite_index(index, [x], access_kind, make_transform())
    assert str(after.dtype) == "int64"
    analyzer = tvm.arith.Analyzer()
    for value in [-32768, 0, 32767]:
        substituted = tir.stmt_functor.substitute(after, {x: tir.IntImm("int16", value)})
        assert int(analyzer.simplify(substituted).value) == value * (1 << 20)


def test_flatten_buffer_preserves_vector_narrowing_cast_lanes():
    tir = tvm.tirx
    x = tir.Var("x", "int64x4")
    narrowed = tir.Cast("int32x4", x)
    after = _rewrite_index(narrowed, [x], "load", tilelang.transform.FlattenBuffer())
    assert str(after.dtype) == "int64x4"
    casts = []

    def collect(node):
        if isinstance(node, tir.Cast) and str(node.dtype) == "int64x4":
            casts.append(node)

    tir.stmt_functor.post_order_visit(after, collect)
    assert len(casts) == 1
    tvm.ir.assert_structural_equal(casts[0], tir.Cast("int64x4", narrowed))


if __name__ == "__main__":
    tilelang.testing.main()
