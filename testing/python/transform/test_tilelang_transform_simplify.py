# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

from tvm import te


def bind_then(var, value, body):
    return tvm.tirx.SeqStmt([tvm.tirx.Bind(var, value), body])


def buffer_pair(shape=(16,)):
    return (
        tvm.tirx.decl_buffer(shape, "float32", name="A"),
        tvm.tirx.decl_buffer(shape, "float32", name="C"),
    )


def simplify_and_compare(before, expected, config=None):
    """Helper function to run simplify pass and compare results."""
    if config is None:
        config = {}

    full_config = {PassConfigKey.TL_SIMPLIFY.value: config}

    with tvm.transform.PassContext(config=full_config):
        after = tl.transform.Simplify()(before)

    # Compare bodies only, ignoring function name differences
    # Use map_free_vars=True to allow mapping of free variables (function parameters)
    after_func = after["main"]
    expected_func = expected["main"]
    tvm.ir.assert_structural_equal(after_func.body, expected_func.body, map_free_vars=True)


def test_stmt_simplify():
    A, C = buffer_pair()
    n = te.size_var("n")
    i = tvm.tirx.Var("i", "int32")

    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [i]), [i])
    loop = tvm.tirx.For(i, 0, n, tvm.tirx.ForKind.SERIAL, tvm.tirx.IfThenElse(i < 12, store, None))
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], bind_then(n, 10, loop)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body, tvm.tirx.BufferStore)


def test_thread_extent_simplify():
    A, C = buffer_pair()
    n = te.size_var("n")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")

    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [tx.var + ty.var]), [tx.var])
    stmt = tvm.tirx.IfThenElse(tx.var + ty.var < 12, store, None)
    stmt = tvm.tirx.AttrStmt(ty, "thread_extent", 1, stmt)
    stmt = tvm.tirx.AttrStmt(tx, "thread_extent", n, stmt)
    stmt = tvm.tirx.AttrStmt(tx, "thread_extent", n, stmt)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], bind_then(n, 10, stmt)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body.body.body, tvm.tirx.BufferStore)


def test_context_singleton_floordiv_index():
    A, C = buffer_pair((128,))
    tx = te.thread_axis("threadIdx.x")
    i = tvm.tirx.Var("i", "int32")
    vec = tvm.tirx.Var("vec", "int32")

    before_index = tx.var // 128 * 64 + i * 4 + vec - 64
    expected_index = i * 4 + vec
    before_accum_index = (tx.var - 128) // 128 * 64 + i
    expected_accum_index = i
    before_store = tvm.tirx.BufferStore(A, tvm.tirx.const(0, "float32"), [before_index])
    expected_store = tvm.tirx.BufferStore(A, tvm.tirx.const(0, "float32"), [expected_index])
    before_accum_store = tvm.tirx.BufferStore(C, tvm.tirx.const(0, "float32"), [before_accum_index])
    expected_accum_store = tvm.tirx.BufferStore(C, tvm.tirx.const(0, "float32"), [expected_accum_index])

    before_body = tvm.tirx.For(
        i,
        0,
        32,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.For(
            vec,
            0,
            4,
            tvm.tirx.ForKind.SERIAL,
            tvm.tirx.SeqStmt([before_store, before_accum_store]),
        ),
    )
    expected_body = tvm.tirx.For(
        i,
        0,
        32,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.For(
            vec,
            0,
            4,
            tvm.tirx.ForKind.SERIAL,
            tvm.tirx.SeqStmt([expected_store, expected_accum_store]),
        ),
    )
    before_stmt = tvm.tirx.AttrStmt(
        tx,
        "thread_extent",
        256,
        tvm.tirx.IfThenElse(tx.var < 128, tvm.tirx.Evaluate(0), before_body),
    )
    expected_stmt = tvm.tirx.AttrStmt(
        tx,
        "thread_extent",
        256,
        tvm.tirx.IfThenElse(tx.var < 128, tvm.tirx.Evaluate(0), expected_body),
    )

    mod_before = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], before_stmt))
    mod_expected = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], expected_stmt))
    simplify_and_compare(mod_before, mod_expected)


def test_if_likely():
    A, C = buffer_pair((1024,))
    n = te.size_var("n")
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")

    idx = tx.var * 32 + ty.var
    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [idx]), [tx.var])
    cond = tvm.tirx.likely(idx < n)
    stmt = tvm.tirx.IfThenElse(cond, tvm.tirx.IfThenElse(cond, store, None), None)
    stmt = tvm.tirx.AttrStmt(ty, "thread_extent", 32, stmt)
    stmt = tvm.tirx.AttrStmt(tx, "thread_extent", 32, stmt)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C, n], stmt))
    body = tl.transform.Simplify()(mod)["main"].body
    assert isinstance(body.body.body, tvm.tirx.IfThenElse)
    assert not isinstance(body.body.body.then_case, tvm.tirx.IfThenElse)


def test_load_store_noop():
    """Store of a value that was just read from the same location is a no-op."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0]

    @T.prim_func
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_load_store_noop_after_simplify():
    """As test_load_store_noop, but requiring simplification to identify."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0] + (5.0 - 5.0)

    @T.prim_func
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_nested_condition():
    """Nested IfThenElse with the same condition can be simplified."""

    @T.prim_func
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i == 5:
                    A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_nested_provable_condition():
    """Simplify inner conditional using constraint from outer."""

    @T.prim_func
    def before(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                if i < 7:
                    A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((16,), "float32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 0.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_nested_var_condition():
    """Simplify inner conditional using constraint from outer."""

    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i == n:
                    A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 0.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_altered_buffer_contents():
    """No simplification of data-dependent conditionals."""

    @T.prim_func
    def before(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0

    mod_before = tvm.IRModule({"main": before})
    # Expected is the same as before
    simplify_and_compare(mod_before, mod_before)


def test_negation_of_condition():
    """Use negation of outer condition to simplify inner."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                if i != 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i == 5:
                A[i] = 1

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_negation_of_not_equal():
    """Test negation with != outer condition."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                if i == 5:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            if i != 5:
                A[i] = 1

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_negation_of_var_condition():
    """Test negation with dynamic condition."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                if i != n:
                    A[i] = 0
                else:
                    A[i] = 1

    @T.prim_func
    def expected(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 1

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_literal_constraint_split_boolean_and():
    """Split a boolean AND into independent constraints."""

    @T.prim_func
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                if i == n:
                    A[i, j] = 0

    @T.prim_func
    def expected(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n and j == n:
                A[i, j] = 0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_literal_constraint_split_boolean_or():
    """Split a boolean OR into independent constraints."""

    @T.prim_func
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2

    @T.prim_func
    def expected(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                A[i, j] = 2

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_if_then_else_expr():
    @T.prim_func
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0, dtype="float32")

    @T.prim_func
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_ceil_log2_int():
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = T.cast(T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32")

    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 4

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_left_shift_lower_bound():
    """Integer bounds are propagated through left shift."""

    @T.prim_func
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(1, i, dtype="int32") >= 1:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_left_shift_upper_bound():
    """Integer bounds are propagated through left shift."""

    @T.prim_func
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(31, i, dtype="int32") <= 1015808:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_conditional_floor_mod():
    """A regression test for negative floormod denominator."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(i, -2) == 0:
            A[0] = True

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_simplify_rhs_of_boolean_and_using_lhs():
    """Boolean expressions can introduce contexts."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 and n < 10

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_lhs_of_boolean_and_using_rhs():
    """Boolean expressions can introduce contexts for their arguments."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 and n < 5

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_rhs_of_boolean_or_using_lhs():
    """Boolean expressions can introduce contexts."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 or n < 5

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_lhs_of_boolean_or_using_rhs():
    """Boolean expressions can introduce contexts for their arguments."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 or n < 10

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_conditional_using_buffer_value():
    """Simplify a conditional using the known value in the buffer."""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        if A[0] == 0:
            A[0] = 42

    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = 42

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_PROVE_CONDITIONAL.value: True})


def test_simplify_non_conditional():
    """Propagate a known value to later expressions."""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = A[0] + 1

    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = 1

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS.value: True})


def test_suppress_simplify_non_conditional():
    """Propagate a known value to later expressions - disabled."""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = A[0] + 1

    mod_before = tvm.IRModule({"main": before})
    simplify_and_compare(mod_before, mod_before, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS.value: False})


def test_simplify_buffer_store():
    """Simplification using prior known."""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 5
        A[0] = A[0] + 7

    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 5
        A[0] = 12

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS.value: True})


def test_rewrite_as_and_of_ors():
    """If enabled, rewrite boolean expressions into AND of OR."""

    @T.prim_func
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    @T.prim_func
    def expected(A: T.Buffer(3, "bool")):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS.value: True})


def test_suppress_rewrite_as_and_of_ors():
    """Only rewrite into AND of OR when allowed."""

    @T.prim_func
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    mod_before = tvm.IRModule({"main": before})
    simplify_and_compare(mod_before, mod_before, {PassConfigKey.TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS.value: False})


def test_buffer_shape_constraint():
    @T.prim_func
    def before(a: T.handle):
        n = T.int64()
        A = T.match_buffer(a, (n * 32,), "float32")
        A[T.min(T.int64(0), n)] = T.float32(0)

    @T.prim_func
    def expected(a: T.handle):
        n = T.int64()
        A = T.match_buffer(a, (n * 32,), "float32")
        A[T.int64(0)] = T.float32(0)

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_tilelang_enable_simplify_let_inline_true():
    """Test that let statements are inlined when tilelang_enable_simplify_let_inline=True (default)."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            x = i + 1
            A[i] = x

    @T.prim_func
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            A[i] = i + 1

    mod_before = tvm.IRModule({"main": before})
    mod_expected = tvm.IRModule({"main": expected})
    # Default behavior: let statements are inlined
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE.value: True})


def test_tilelang_enable_simplify_let_inline_false():
    """Test that let statements are NOT inlined when tilelang_enable_simplify_let_inline=False."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            x = i + 1
            A[i] = x

    mod_before = tvm.IRModule({"main": before})
    # When disabled, let statements should be preserved (before == after)
    simplify_and_compare(mod_before, mod_before, {PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE.value: False})


if __name__ == "__main__":
    tilelang.testing.main()
