"""Flattening must preserve rounding in reverse, split iterator indices."""

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm import tirx


# (base, scale, inner divisor, outer divisor). Include forward and exactly
# divisible controls so the fix does not disable normal iterator simplification.
CASES = [
    (7, -1, 2, 2),
    (15, -1, 2, 4),
    (15, -2, 2, 4),
    (16, -2, 2, 2),
    (0, 1, 2, 2),
    (0, 2, 2, 4),
]


def _index(x, base, scale, inner, outer, modulo):
    numerator = base + scale * (x // inner)
    return numerator % outer if modulo else numerator // outer


def test_positive_split_still_simplifies():
    x = tirx.Var("x", "int32")
    result = tvm.arith.iter_map_simplify([(x // 2) // 2], {x: tvm.ir.Range(0, 16)}, check_level="surjective")[0]
    tvm.ir.assert_structural_equal(result, x // 4)


@pytest.mark.parametrize("modulo", [False, True])
def test_direct_negative_divisor_is_not_simplified(modulo):
    x = tirx.Var("x", "int32")
    expr = (x + 1) % -2 if modulo else (x + 1) // -2
    result = tvm.arith.iter_map_simplify([expr], {x: tvm.ir.Range(0, 16)}, check_level="surjective")[0]
    tvm.ir.assert_structural_equal(result, expr)


def test_reverse_split_removes_the_outer_division():
    x = tirx.Var("x", "int32")
    # (16 - 2 * (x // 2)) // 2 * 16 == (8 - x // 2) * 16. The outer floor
    # division must be folded into the iterator instead of surviving as a
    # negative-scale division.
    expr = ((16 - 2 * (x // 2)) // 2) * 16
    result = tvm.arith.iter_map_simplify([expr], {x: tvm.ir.Range(0, 16)}, check_level="surjective")[0]
    tvm.ir.assert_structural_equal(result, x // 2 * -16 + 128)


def test_symbolic_divisor_is_only_rejected_when_provably_non_positive():
    x = tirx.Var("x", "int32")
    n = tirx.Var("n", "int32")
    expr = (x + 1) // n * 16
    result = tvm.arith.iter_map_simplify([expr], {x: tvm.ir.Range(0, 16)}, check_level="surjective")[0]

    # An unknown-sign divisor is not provably non-positive, so the expression
    # is left alone rather than rejected outright.
    tvm.ir.assert_structural_equal(result, expr)

    # Substituting a negative divisor afterwards must still produce the right
    # floor-division result; this guards against assuming the divisor is
    # positive when the sign cannot be proven.
    for divisor in (-4, -2, -1, 1, 2, 4):
        for value in range(16):
            actual = tvm.arith.Analyzer().simplify(
                tirx.stmt_functor.substitute(
                    result,
                    {x: tirx.IntImm("int32", value), n: tirx.IntImm("int32", divisor)},
                )
            )
            assert int(actual) == ((value + 1) // divisor) * 16, (divisor, value, result)


@pytest.mark.parametrize("base,scale,inner,outer", CASES)
@pytest.mark.parametrize("modulo", [False, True])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_iter_map_preserves_reverse_split(base, scale, inner, outer, modulo, dtype):
    x = tirx.Var("x", dtype)
    expr = _index(x, base, scale, inner, outer, modulo) * 16
    result = tvm.arith.iter_map_simplify([expr], {x: tvm.ir.Range(0, 16)}, check_level="surjective")[0]
    for value in range(16):
        actual = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(result, {x: tirx.IntImm(dtype, value)}))
        expected = _index(value, base, scale, inner, outer, modulo) * 16
        assert int(actual) == expected, (value, expr, result)


@pytest.mark.parametrize("base,scale,inner,outer", CASES)
@pytest.mark.parametrize("modulo", [False, True])
@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("thread_bound", [False, True])
def test_flatten_reverse_split_index(base, scale, inner, outer, modulo, store, thread_bound):
    x = tirx.Var("x", "int32")
    a = tirx.decl_buffer((64, 16), "int32", name="A")
    b = tirx.decl_buffer((16,), "int32", name="B")
    index = _index(x, base, scale, inner, outer, modulo)
    body = tirx.BufferStore(a, b[x], [index, 0]) if store else tirx.BufferStore(b, a[index, 0], [x])
    if thread_bound:
        axis = tirx.IterVar(tvm.ir.Range(0, 16), x, tirx.IterVar.ThreadIndex, "blockIdx.x")
        body = tirx.AttrStmt(axis, "thread_extent", 16, body)
    else:
        body = tirx.For(x, 0, 16, tirx.ForKind.SERIAL, body)
    func = tirx.PrimFunc([a.data, b.data], body, buffer_map={a.data: a, b.data: b})
    lowered = tilelang.transform.FlattenBuffer()(tvm.IRModule({"main": func}))["main"]
    accesses = []

    def visit(node):
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)) and node.buffer.data.same_as(a.data):
            accesses.append(node)

    tirx.stmt_functor.post_order_visit(lowered.body, visit)
    assert len(accesses) == 1 and len(accesses[0].indices) == 1
    flattened = accesses[0].indices[0]
    for value in range(16):
        actual = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(flattened, {x: value}))
        expected = _index(value, base, scale, inner, outer, modulo) * 16
        assert int(actual) == expected, (value, index, flattened)


if __name__ == "__main__":
    tilelang.testing.main()
