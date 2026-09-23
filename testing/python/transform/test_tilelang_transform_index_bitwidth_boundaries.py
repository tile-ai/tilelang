import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm import tirx


INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _make_access(index, access):
    # These are pass-only tests: no buffer is allocated or accessed at runtime.
    buffer = tirx.decl_buffer((1,), "uint8", name="A")
    if access == "load":
        body = tirx.Evaluate(tirx.BufferLoad(buffer, [index]))
    elif access == "store":
        body = tirx.BufferStore(buffer, tirx.const(1, "uint8"), [index])
    else:
        body = tirx.Evaluate(tirx.call_intrin("handle", "tirx.tvm_access_ptr", tirx.type_annotation("uint8"), buffer.data, index, 1, 1))
    return buffer, body


def _get_index(func, access):
    indices = []

    def visit(node):
        if (access == "load" and isinstance(node, tirx.BufferLoad)) or (access == "store" and isinstance(node, tirx.BufferStore)):
            indices.append(node.indices[0])
        elif access == "access_ptr" and isinstance(node, tirx.Call) and node.op.same_as(tvm.ir.Op.get("tirx.tvm_access_ptr")):
            indices.append(node.args[2])

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(indices) == 1
    return indices[0]


ACCESS_PASSES = [
    pytest.param("FlattenBuffer", "load", id="flatten-load"),
    pytest.param("FlattenBuffer", "store", id="flatten-store"),
    pytest.param("FlattenBuffer", "access_ptr", id="flatten-access-ptr"),
    pytest.param("ConfigIndexBitwidth", "load", id="legalize-load"),
    pytest.param("ConfigIndexBitwidth", "store", id="legalize-store"),
]


@pytest.mark.parametrize("pass_name,access", ACCESS_PASSES)
@pytest.mark.parametrize("value", [INT32_MIN, INT32_MIN + 1, INT32_MAX - 1, INT32_MAX])
def test_index_bitwidth_representable_boundary(pass_name, access, value):
    index = tirx.const(value, "int32")
    buffer, body = _make_access(index, access)
    before = tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer}).with_attr("global_symbol", "main")
    after = getattr(tilelang.transform, pass_name)()(tvm.IRModule.from_expr(before))["main"]

    actual = _get_index(after, access)
    assert actual.dtype == "int32"
    assert tvm.arith.Analyzer().simplify(actual).value == value


@pytest.mark.parametrize("pass_name,access", ACCESS_PASSES)
@pytest.mark.parametrize(
    "base,step,expected_dtype",
    [
        pytest.param(INT32_MIN, 1, "int32", id="at-lower-bound"),
        pytest.param(INT32_MAX - 2, 1, "int32", id="below-upper-bound"),
        pytest.param(INT32_MAX - 1, 1, "int32", id="at-upper-bound"),
        pytest.param(INT32_MAX, 1, "int64", id="above-upper-bound"),
        pytest.param(INT32_MIN, -1, "int64", id="below-lower-bound"),
    ],
)
def test_index_bitwidth_arithmetic_bounds(pass_name, access, base, step, expected_dtype):
    i = tirx.Var("i", "int32")
    index = tirx.const(base, "int32") + i * step
    buffer, access_stmt = _make_access(index, access)
    body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, access_stmt)
    before = tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer}).with_attr("global_symbol", "main")
    assert _get_index(before, access).dtype == "int32"
    after = getattr(tilelang.transform, pass_name)()(tvm.IRModule.from_expr(before))["main"]

    actual = _get_index(after, access)
    assert actual.dtype == expected_dtype
    for value in (0, 1):
        substituted = tirx.stmt_functor.substitute(actual, {i: tirx.const(value, "int32")})
        assert tvm.arith.Analyzer().simplify(substituted).value == base + value * step


@pytest.mark.parametrize("pass_name,access", ACCESS_PASSES)
@pytest.mark.parametrize("value", [INT32_MIN - 1, INT32_MAX, INT32_MAX + 1])
def test_index_bitwidth_preserves_existing_int64(pass_name, access, value):
    buffer, body = _make_access(tirx.const(value, "int64"), access)
    before = tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer}).with_attr("global_symbol", "main")
    after = getattr(tilelang.transform, pass_name)()(tvm.IRModule.from_expr(before))["main"]

    actual = _get_index(after, access)
    assert actual.dtype == "int64"
    assert tvm.arith.Analyzer().simplify(actual).value == value


if __name__ == "__main__":
    tilelang.testing.main()
