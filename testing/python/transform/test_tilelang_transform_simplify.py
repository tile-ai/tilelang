# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

from tilelang._compat import tirx
from tilelang._compat.tensor import TensorView
from tilelang._compat.stmt_functor import ir_transform, post_order_visit
from tvm_tirx import ir
from tvm_tirx.tirx import op as tirx_op
import pytest


def launch_thread(variable, extent, body):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(variable.name), extent], [variable], None, body)


def nodes(root, kind):
    result = []
    post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


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

    def unscoped(body):
        def visit(node):
            if (
                isinstance(node, ir.RegionStmt)
                and node.op.name == "tl.scope"
                and not len(node.attrs)
                and not node.body_params
                and not node.result_vars
                and not nodes(node.body, ir.Bind)
            ):
                return node.body
            return None

        return ir_transform(body, None, visit)

    tvm.ir.assert_structural_equal(unscoped(after_func.body), unscoped(expected_func.body), map_free_vars=True)
    return after


def test_stmt_simplify():
    A, C = buffer_pair()
    n = tirx.Var("n", "int32")
    i = tvm.tirx.Var("i", "int32")

    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [i]), [i])
    loop = tvm.tirx.For(i, 0, n, 1, tvm.tirx.ForKind.DEFAULT, tvm.tirx.IfThenElse(i < 12, store, None))
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], bind_then(n, 10, loop)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(nodes(body, ir.TensorStore)) == 1
    assert not nodes(body, ir.If)


def test_thread_extent_simplify():
    A, C = buffer_pair()
    n = tirx.Var("n", "int32")
    tx = tirx.Var("threadIdx.x", "int32")
    ty = tirx.Var("threadIdx.y", "int32")

    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [tx + ty]), [tx])
    stmt = tvm.tirx.IfThenElse(tx + ty < 12, store, None)
    stmt = launch_thread(ty, 1, stmt)
    stmt = launch_thread(tx, n, stmt)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], bind_then(n, 10, stmt)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(nodes(body, ir.TensorStore)) == 1
    assert not nodes(body, ir.If)


def test_dynamic_reduction_tail_guard():
    batch = tvm.tirx.Var("batch", "int32")
    i = tvm.tirx.Var("i", "int32")
    tx = tirx.Var("threadIdx.x", "int32")
    source = tvm.tirx.decl_buffer((batch, 192), "float32", name="A")
    output = tvm.tirx.decl_buffer((128,), "float32", name="C")
    offset = i * 128 + tx
    store = tvm.tirx.BufferStore(output, tvm.tirx.BufferLoad(source, [offset // 192, offset % 192]), [tx])
    guard = tvm.tirx.IfThenElse(offset < batch * 192, store, None)
    loop = tvm.tirx.For(i, 0, (batch * 192 - 1) // 128 + 1, 1, tvm.tirx.ForKind.DEFAULT, guard)
    body = launch_thread(tx, 128, loop)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([source, output], body))

    # For batch=1, i=1 and tx=64, the offset is 192 and the guard is false.
    # Incorrect modular bounds used to prove it true without a batch > 0 assume.
    simplify_and_compare(mod, mod)


def test_context_singleton_floordiv_index():
    A, C = buffer_pair((128,))
    tx = tirx.Var("threadIdx.x", "int32")
    i = tvm.tirx.Var("i", "int32")
    vec = tvm.tirx.Var("vec", "int32")

    before_index = tx // 128 * 64 + i * 4 + vec - 64
    expected_index = i * 4 + vec
    before_accum_index = (tx - 128) // 128 * 64 + i
    expected_accum_index = i
    before_store = tvm.tirx.BufferStore(A, tvm.tirx.const(0, "float32"), [before_index])
    expected_store = tvm.tirx.BufferStore(A, tvm.tirx.const(0, "float32"), [expected_index])
    before_accum_store = tvm.tirx.BufferStore(C, tvm.tirx.const(0, "float32"), [before_accum_index])
    expected_accum_store = tvm.tirx.BufferStore(C, tvm.tirx.const(0, "float32"), [expected_accum_index])

    before_body = tvm.tirx.For(
        i,
        0,
        32,
        1,
        tvm.tirx.ForKind.DEFAULT,
        tvm.tirx.For(
            vec,
            0,
            4,
            1,
            tvm.tirx.ForKind.DEFAULT,
            tvm.tirx.SeqStmt([before_store, before_accum_store]),
        ),
    )
    expected_body = tvm.tirx.For(
        i,
        0,
        32,
        1,
        tvm.tirx.ForKind.DEFAULT,
        tvm.tirx.For(
            vec,
            0,
            4,
            1,
            tvm.tirx.ForKind.DEFAULT,
            tvm.tirx.SeqStmt([expected_store, expected_accum_store]),
        ),
    )
    before_stmt = launch_thread(
        tx,
        256,
        tvm.tirx.IfThenElse(tx < 128, tvm.tirx.Evaluate(0), before_body),
    )
    expected_stmt = launch_thread(
        tx,
        256,
        tvm.tirx.IfThenElse(tx < 128, tvm.tirx.Evaluate(0), expected_body),
    )

    mod_before = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], before_stmt))
    mod_expected = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C], expected_stmt))
    simplify_and_compare(mod_before, mod_expected)


def test_if_likely():
    A, C = buffer_pair((1024,))
    n = tirx.Var("n", "int32")
    tx = tirx.Var("threadIdx.x", "int32")
    ty = tirx.Var("threadIdx.y", "int32")

    idx = tx * 32 + ty
    store = tvm.tirx.BufferStore(A, tvm.tirx.BufferLoad(C, [idx]), [tx])
    cond = tvm.tirx.likely(idx < n)
    stmt = tvm.tirx.IfThenElse(cond, tvm.tirx.IfThenElse(cond, store, None), None)
    stmt = launch_thread(ty, 32, stmt)
    stmt = launch_thread(tx, 32, stmt)
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([A, C, n], stmt))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(nodes(body, ir.If)) == 1


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
                A[i] = T.if_then_else(i < 12, T.float32(1.0), T.float32(2.0))

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
        A[0] = T.Cast("int32", T.ceil(T.log2(T.Cast("float64", 14))))

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
            if T.shift_left(1, i) >= 1:
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
            if T.shift_left(31, i) <= 1015808:
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
    extent = T.dynamic("extent", "int64")

    @T.prim_func
    def before(A: T.Tensor((extent,), "float32")):
        n = extent
        A[T.min(T.int64(0), n)] = T.float32(0)

    @T.prim_func
    def expected(A: T.Tensor((extent,), "float32")):
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


def test_simplify_removes_unused_macro_argument(capfd):
    """An ignored macro argument must not leave an uninitialized buffer read."""

    @T.macro
    def ignore_argument(value):
        return 7.0

    @T.prim_func
    def before(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((1,), "float32")
            B[0] = ignore_argument(x[0])

    @T.prim_func
    def expected(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((1,), "float32")
            B[0] = 7.0

    mod_before = tvm.IRModule({"main": before})
    tl.transform.VerifyBufferInit()(mod_before)
    assert "Buffer read before initialization" in capfd.readouterr().err

    after = simplify_and_compare(mod_before, tvm.IRModule({"main": expected}))
    tl.transform.VerifyBufferInit()(after)
    assert "Buffer read before initialization" not in capfd.readouterr().err


def test_simplify_removes_unused_macro_bind_chain():
    """Removing the last unused binding must also remove its producers."""

    @T.macro
    def ignore_rebound_argument(value):
        value = value + 1.0
        value = value + 2.0
        return 7.0

    @T.prim_func
    def before(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((1,), "float32")
            B[0] = ignore_rebound_argument(x[0])

    @T.prim_func
    def expected(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((1,), "float32")
            B[0] = 7.0

    simplify_and_compare(tvm.IRModule({"main": before}), tvm.IRModule({"main": expected}))


def test_simplify_keeps_used_macro_argument(capfd):
    """A macro that uses its argument must still report the uninitialized read."""

    @T.macro
    def increment(value):
        value = value + 1.0
        return value

    @T.prim_func
    def before(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((1,), "float32")
            B[0] = increment(x[0])

    mod_before = tvm.IRModule({"main": before})
    after = tl.transform.Simplify()(mod_before)
    loads = nodes(after["main"].body, ir.TensorLoad)
    assert len(loads) == 1 and loads[0].source.name == "x"
    tl.transform.VerifyBufferInit()(after)
    assert "Buffer read before initialization" in capfd.readouterr().err


def test_simplify_keeps_bind_read_before_write():
    """The stored value is the old A[0]; the read must stay before the write."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32"), C: T.Buffer((1,), "float32")):
        value = A[0]
        A[0] = 2.0
        C[0] = value

    mod_before = tvm.IRModule({"main": before})
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(lambda A: tirx.call_extern("float32", "update_state", A.data), id="extern_call"),
        pytest.param(lambda A: T.atomic_add(A[0], 1.0, return_prev=True), id="atomic_add"),
        pytest.param(lambda A: tirx_op.gpu_storage_sync("shared"), id="storage_sync"),
        pytest.param(
            lambda A: tirx_op.masked_load(A, 0, tirx.call_extern("bool", "predicate_effect"), ty="float32"),
            id="buffer_load_predicate",
        ),
    ],
)
def test_simplify_keeps_bind_with_side_effects(make_value):
    """Delete the unused consumer, but keep the operation with side effects."""
    A, _ = buffer_pair()
    value = make_value(A)
    if value.ty == ir.PrimType("void"):
        effectful_bind = ir.Evaluate(value)
        unused_consumer = ir.Bind(ir.Var("unused", "int32"), ir.const(1))
    else:
        result = tirx.Var("result", value.ty)
        effectful_bind = tirx.Bind(result, value)
        unused_consumer = tirx.Bind(tirx.Var("unused", value.ty), result + 1)

    before_body = tirx.SeqStmt([effectful_bind, unused_consumer])
    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([A], before_body))
    mod_expected = tvm.IRModule.from_expr(tirx.PrimFunc([A], effectful_bind))
    simplify_and_compare(mod_before, mod_expected)


@pytest.mark.parametrize("annotation", ["volatile_scope", "tirx.volatile"])
def test_simplify_keeps_bind_reading_volatile_buffer(annotation):
    """A read through an alias still observes the original buffer's volatility."""
    source = TensorView.create("source", [1], "float32", scope="local")
    alias = TensorView.create("alias", [1], "float32", scope="local")
    value = tirx.Var("value", "float32")

    def wrap_volatile(body):
        allocation_attrs = {"tirx.volatile": True} if annotation == "tirx.volatile" else {}
        allocation = ir.Bind(source.tensor, ir.Call("tirx.alloc_tensor", [], attrs=allocation_attrs, ty_args=[source.tensor.ty]))
        body = ir.SeqStmt([allocation, alias.declare(source.data), body])
        if annotation == "volatile_scope":
            body = ir.RegionStmt("tl.scope", [], [], ir.DictAttrs({"volatile_scope": source.tensor}), body)
        return tvm.IRModule.from_expr(tirx.PrimFunc([], body))

    volatile_read = tirx.Bind(value, alias.load(0))
    unused_consumer = tirx.Bind(tirx.Var("unused", "float32"), value + 1.0)
    mod_before = wrap_volatile(tirx.SeqStmt([volatile_read, unused_consumer]))
    mod_expected = wrap_volatile(volatile_read)
    simplify_and_compare(mod_before, mod_expected)


def test_simplify_keeps_binds_used_in_buffer_metadata():
    """Buffer shape, strides, offset and data pointer all count as variable uses."""
    source = tirx.decl_buffer((3,), "int32")
    size = tirx.Var("size", "int32")
    stride = tirx.Var("stride", "int32")
    offset = tirx.Var("offset", "int32")
    data = tirx.Var("data", ir.PtrType(tvm.ir.PrimType("float32")))
    view = TensorView.create("view", [size], "float32", strides=[stride], elem_offset=offset)

    body = tirx.SeqStmt(
        [
            tirx.Bind(size, source[0]),
            tirx.Bind(stride, source[1]),
            tirx.Bind(offset, source[2]),
            tirx.Bind(data, ir.Call("tirx.call_pure_extern", ["get_pointer"], ty=data.ty)),
            view.declare(data),
            view.store(0, 1.0),
        ]
    )
    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([source], body))
    simplify_and_compare(mod_before, mod_before)


def test_simplify_keeps_bind_used_in_nested_annotation():
    """A variable referenced only through annotation maps/arrays is still live."""
    source = tirx.decl_buffer((1,), "int32")
    output = tirx.decl_buffer((2,), "float32")
    tag = tirx.Var("tag", "int32")
    i = tirx.Var("i", "int32")
    loop = tirx.For(
        i,
        0,
        2,
        1,
        tirx.ForKind.DEFAULT,
        tirx.BufferStore(output, 1.0, [i]),
        annotations={"test.metadata": {"nested": [tag]}},
    )
    body = bind_then(tag, source[0], loop)
    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([source, output], body))
    simplify_and_compare(mod_before, mod_before)


def test_simplify_inlines_constant_including_annotation_uses():
    """Inline ordinary uses, but keep the definition still used by an annotation."""
    output = tirx.decl_buffer((2,), "int32")
    tag = tirx.Var("tag", "int32")
    i = tirx.Var("i", "int32")
    annotations = {"test.metadata": {"nested": [tag]}}

    # tag = 4; for (..., annotation=[tag]): output[i] = tag + 1
    before_loop = tirx.For(i, 0, 2, 1, tirx.ForKind.DEFAULT, tirx.BufferStore(output, tag + 1, [i]), annotations=annotations)
    # The store simplifies to 5, while the annotation still requires tag = 4.
    expected_loop = tirx.For(
        i, 0, 2, 1, tirx.ForKind.DEFAULT, tirx.BufferStore(output, 5, [i]), annotations={"test.metadata": {"nested": [ir.const(4)]}}
    )
    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([output], bind_then(tag, 4, before_loop)))
    mod_expected = tvm.IRModule.from_expr(tirx.PrimFunc([output], expected_loop))
    simplify_and_compare(mod_before, mod_expected)


@pytest.mark.parametrize("site", ["function", "region", "allocation", "call", "cast"])
def test_simplify_keeps_bind_used_in_metadata_objects(site):
    """References outside ordinary expression children must keep bindings live."""
    source = tirx.decl_buffer((2,), "int32")
    output = tirx.decl_buffer((1,), "float32")
    tag = tirx.Var("tag", "int32")
    metadata = {"test.metadata": {"nested": [tag]}}
    body = tirx.BufferStore(output, 1.0, [0])

    if site == "region":
        body = ir.RegionStmt("tl.scope", [], [], ir.DictAttrs(metadata), body)
    elif site == "allocation":
        scratch = tirx.decl_buffer((1,), "float32", scope="local")
        body = tirx.SeqStmt([ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], attrs=metadata, ty_args=[scratch.ty])), body])
    elif site == "call":
        call = tirx.call_extern("float32", "observe")
        call = ir.Call(call.op, call.args, attrs=metadata, ty=call.ty)
        body = tirx.BufferStore(output, call, [0])
    elif site == "cast":
        value = tirx.Cast("float32", ir.Call("tirx.reinterpret", [source[1]], attrs=metadata, ty="int32"))
        body = tirx.BufferStore(output, value, [0])

    func = tirx.PrimFunc([source, output], bind_then(tag, source[0], body))
    if site == "function":
        func = func.with_attr("test.metadata", metadata["test.metadata"])
    mod_before = tvm.IRModule.from_expr(func)
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize("access", ["load", "store"])
def test_simplify_keeps_bind_used_only_in_buffer_predicate(access):
    """Both load and store predicates are uses even when the index is constant."""
    source = tirx.decl_buffer((1,), "int32")
    A, C = buffer_pair()
    predicate = tirx.Var("predicate", "bool")
    if access == "load":
        body = tirx.BufferStore(C, tirx_op.masked_load(A, 0, predicate, ty="float32"), [0])
    else:
        body = ir.Evaluate(tirx_op.masked_store(C, ir.const(1.0, "float32"), 0, predicate))
    body = bind_then(predicate, tirx.LT(0, source[0]), body)
    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([source, A, C], body))
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize(
    "store_uses_shared_value",
    [pytest.param(False, id="all_uses_dead"), pytest.param(True, id="live_store")],
)
def test_simplify_dead_bind_with_shared_nodes(store_uses_shared_value):
    """Shared nodes must neither keep dead producers alive nor hide live uses."""
    A, C = buffer_pair()
    cond = tirx.Var("cond", "bool")
    value = tirx.Var("value", "float32")
    shared = value + 1.0
    dead = tirx.Bind(tirx.Var("dead", "float32"), shared)
    store = tirx.BufferStore(C, shared if store_uses_shared_value else 0.0, [0])

    # Reuse the exact same Bind in both branches, and the same expression in the live store.
    before_body = bind_then(value, A[0], tirx.SeqStmt([tirx.IfThenElse(cond, dead, dead), store]))
    # Bind cleanup currently leaves the empty branches in place.
    expected_body = tirx.SeqStmt([tirx.IfThenElse(cond, tirx.Evaluate(0), tirx.Evaluate(0)), store])
    if store_uses_shared_value:
        expected_body = bind_then(value, A[0], expected_body)

    mod_before = tvm.IRModule.from_expr(tirx.PrimFunc([A, C, cond], before_body))
    mod_expected = tvm.IRModule.from_expr(tirx.PrimFunc([A, C, cond], expected_body))
    simplify_and_compare(mod_before, mod_expected)


if __name__ == "__main__":
    tilelang.testing.main()
