# ruff: noqa
import tvm_tirx
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.tirx.layout import Iter
from tilelang.language import _native
from tilelang.language._native import encode_annotations
from tilelang.testing.ir import collect_nodes
import pytest


def bind_then(var, value, body):
    return ir.SeqStmt([ir.Bind(var, _native._expr(value)), body])


def tensor(name, shape, dtype="float32", scope="global", **kwargs):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout="default", **kwargs))


def thread_scope(variable, tag, extent, body):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), _native._expr(extent)], [variable], None, body)


def attribute(node, key, value, body):
    schema, payload = encode_annotations({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def annotation(values):
    schema, payload = encode_annotations(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def buffer_pair(shape=(16,)):
    return (
        tensor("A", shape),
        tensor("C", shape),
    )


def simplify_and_compare(before, expected, config=None):
    """Helper function to run simplify pass and compare results."""
    if config is None:
        config = {}

    full_config = {PassConfigKey.TL_SIMPLIFY.value: config}

    with tvm_tirx.transform.PassContext(config=full_config):
        after = tl.transform.Simplify()(before)

    # Compare bodies only, ignoring function name differences
    # Use map_free_vars=True to allow mapping of free variables (function parameters)
    after_func = after["main"]
    expected_func = expected["main"]
    tvm_tirx.ir.assert_structural_equal(after_func.body, expected_func.body, map_free_vars=True)
    return after


def test_stmt_simplify():
    A, C = buffer_pair()
    n = ir.Var("n", "int32")
    i = ir.Var("i", "int32")

    store = ir.TensorStore(A, [i], C[i])
    loop = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, ir.If(i < 12, store, None))
    mod = tvm_tirx.IRModule.from_expr(tirx.Function([A, C], bind_then(n, 10, loop)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(collect_nodes(body, ir.If)) == 0
    assert len(collect_nodes(body, ir.For)) == 1
    assert len(collect_nodes(body, ir.TensorStore)) == 1


def test_thread_extent_simplify():
    A, C = buffer_pair()
    n = ir.Var("n", "int32")
    tx_outer = ir.Var("tx_outer", "int32")
    tx = ir.Var("tx", "int32")
    ty = ir.Var("ty", "int32")

    store = ir.TensorStore(A, [tx], C[tx + ty])
    stmt = ir.If(tx + ty < 12, store, None)
    stmt = thread_scope(ty, "threadIdx.y", 1, stmt)
    stmt = thread_scope(tx, "threadIdx.x", n, stmt)
    stmt = thread_scope(tx_outer, "threadIdx.x", n, stmt)
    mod = tvm_tirx.IRModule.from_expr(tirx.Function([A, C], bind_then(n, 10, stmt)))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(collect_nodes(body, ir.If)) == 0
    assert len(collect_nodes(body, ir.RegionStmt)) == 3
    assert len(collect_nodes(body, ir.TensorStore)) == 1


def test_dynamic_reduction_tail_guard():
    batch = ir.Var("batch", "int32")
    i = ir.Var("i", "int32")
    tx = ir.Var("tx", "int32")
    source = tensor("A", (batch, 192))
    output = tensor("C", (128,))
    offset = i * 128 + tx
    store = ir.TensorStore(output, [tx], source[offset // 192, offset % 192])
    guard = ir.If(offset < batch * 192, store, None)
    loop = ir.For(i, 0, (batch * 192 - 1) // 128 + 1, 1, ir.ForKind.DEFAULT, guard)
    body = thread_scope(tx, "threadIdx.x", 128, loop)
    mod = tvm_tirx.IRModule.from_expr(tirx.Function([source, output], body))

    # For batch=1, i=1 and tx=64, the offset is 192 and the guard is false.
    # Incorrect modular bounds used to prove it true without a batch > 0 assume.
    simplify_and_compare(mod, mod)


def test_context_singleton_floordiv_index():
    A, C = buffer_pair((128,))
    tx = ir.Var("tx", "int32")
    i = ir.Var("i", "int32")
    vec = ir.Var("vec", "int32")

    before_index = tx // 128 * 64 + i * 4 + vec - 64
    expected_index = i * 4 + vec
    before_accum_index = (tx - 128) // 128 * 64 + i
    expected_accum_index = i
    before_store = ir.TensorStore(A, [before_index], ir.const(0, "float32"))
    expected_store = ir.TensorStore(A, [expected_index], ir.const(0, "float32"))
    before_accum_store = ir.TensorStore(C, [before_accum_index], ir.const(0, "float32"))
    expected_accum_store = ir.TensorStore(C, [expected_accum_index], ir.const(0, "float32"))

    before_body = ir.For(
        i,
        0,
        32,
        1,
        ir.ForKind.DEFAULT,
        ir.For(
            vec,
            0,
            4,
            1,
            ir.ForKind.DEFAULT,
            ir.SeqStmt([before_store, before_accum_store]),
        ),
    )
    expected_body = ir.For(
        i,
        0,
        32,
        1,
        ir.ForKind.DEFAULT,
        ir.For(
            vec,
            0,
            4,
            1,
            ir.ForKind.DEFAULT,
            ir.SeqStmt([expected_store, expected_accum_store]),
        ),
    )
    before_stmt = thread_scope(
        tx,
        "threadIdx.x",
        256,
        ir.If(tx < 128, ir.Evaluate(0), before_body),
    )
    expected_stmt = thread_scope(
        tx,
        "threadIdx.x",
        256,
        ir.If(tx < 128, ir.Evaluate(0), expected_body),
    )

    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([A, C], before_stmt))
    mod_expected = tvm_tirx.IRModule.from_expr(tirx.Function([A, C], expected_stmt))
    simplify_and_compare(mod_before, mod_expected)


def test_if_likely():
    A, C = buffer_pair((1024,))
    n = ir.Var("n", "int32")
    tx = ir.Var("tx", "int32")
    ty = ir.Var("ty", "int32")

    idx = tx * 32 + ty
    store = ir.TensorStore(A, [tx], C[idx])
    cond = prim.likely(idx < n)
    stmt = ir.If(cond, ir.If(cond, store, None), None)
    stmt = thread_scope(ty, "threadIdx.y", 32, stmt)
    stmt = thread_scope(tx, "threadIdx.x", 32, stmt)
    mod = tvm_tirx.IRModule.from_expr(tirx.Function([A, C, n], stmt))
    body = tl.transform.Simplify()(mod)["main"].body
    assert len(collect_nodes(body, ir.If)) == 1
    assert len(collect_nodes(body, ir.TensorStore)) == 1


def test_load_store_noop():
    """Store of a value that was just read from the same location is a no-op."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0]

    @T.prim_func
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_load_store_noop_after_simplify():
    """As test_load_store_noop, but requiring simplification to identify."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0] + (5.0 - 5.0)

    @T.prim_func
    def expected(A: T.Buffer((1,), "float32")):
        T.evaluate(0)

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_altered_buffer_contents():
    """No simplification of data-dependent conditionals."""

    @T.prim_func
    def before(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0

    mod_before = tvm_tirx.IRModule({"main": before})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_ceil_log2_int():
    """Simplify expressions resulting from topi.math.ceil_log2"""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = T.cast(T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32")

    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 4

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected)


def test_simplify_rhs_of_boolean_and_using_lhs():
    """Boolean expressions can introduce contexts."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 and n < 10

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_lhs_of_boolean_and_using_rhs():
    """Boolean expressions can introduce contexts for their arguments."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 and n < 5

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_rhs_of_boolean_or_using_lhs():
    """Boolean expressions can introduce contexts."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10 or n < 5

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES.value: True})


def test_simplify_lhs_of_boolean_or_using_rhs():
    """Boolean expressions can introduce contexts for their arguments."""

    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 or n < 10

    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 10

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS.value: True})


def test_suppress_simplify_non_conditional():
    """Propagate a known value to later expressions - disabled."""

    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = A[0] + 1

    mod_before = tvm_tirx.IRModule({"main": before})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS.value: True})


def test_rewrite_as_and_of_ors():
    """If enabled, rewrite boolean expressions into AND of OR."""

    @T.prim_func
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    @T.prim_func
    def expected(A: T.Buffer(3, "bool")):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS.value: True})


def test_suppress_rewrite_as_and_of_ors():
    """Only rewrite into AND of OR when allowed."""

    @T.prim_func
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))

    mod_before = tvm_tirx.IRModule({"main": before})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
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

    mod_before = tvm_tirx.IRModule({"main": before})
    mod_expected = tvm_tirx.IRModule({"main": expected})
    # Default behavior: let statements are inlined
    simplify_and_compare(mod_before, mod_expected, {PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE.value: True})


def test_tilelang_enable_simplify_let_inline_false():
    """Test that let statements are NOT inlined when tilelang_enable_simplify_let_inline=False."""

    @T.prim_func
    def before(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            x = i + 1
            A[i] = x

    mod_before = tvm_tirx.IRModule({"main": before})
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
            x = T.alloc_var("float32")
            B[0] = ignore_argument(x)

    @T.prim_func
    def expected(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_var("float32")
            B[0] = 7.0

    mod_before = tvm_tirx.IRModule({"main": before})
    tl.transform.VerifyBufferInit()(mod_before)
    assert "Buffer read before initialization" in capfd.readouterr().err

    after = simplify_and_compare(mod_before, tvm_tirx.IRModule({"main": expected}))
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
            x = T.alloc_var("float32")
            B[0] = ignore_rebound_argument(x)

    @T.prim_func
    def expected(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_var("float32")
            B[0] = 7.0

    simplify_and_compare(tvm_tirx.IRModule({"main": before}), tvm_tirx.IRModule({"main": expected}))


def test_simplify_keeps_used_macro_argument(capfd):
    """A macro that uses its argument must still report the uninitialized read."""

    @T.macro
    def increment(value):
        value = value + 1.0
        return value

    @T.prim_func
    def before(B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_var("float32")
            B[0] = increment(x)

    mod_before = tvm_tirx.IRModule({"main": before})
    after = tl.transform.Simplify()(mod_before)
    # Native Bind simplification can inline the final pure arithmetic use,
    # but the argument's actual uninitialized load must remain observable.
    reads = collect_nodes(after["main"].body, ir.TensorLoad)
    assert len(reads) == 1 and reads[0].source.name == "x"
    tl.transform.VerifyBufferInit()(after)
    assert "Buffer read before initialization" in capfd.readouterr().err


def test_simplify_keeps_bind_read_before_write():
    """The stored value is the old A[0]; the read must stay before the write."""

    @T.prim_func
    def before(A: T.Buffer((1,), "float32"), C: T.Buffer((1,), "float32")):
        value = A[0]
        A[0] = 2.0
        C[0] = value

    mod_before = tvm_tirx.IRModule({"main": before})
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(lambda A: tirx.call_extern("update_state", A.data, ty="float32"), id="extern_call"),
        pytest.param(lambda A: T.atomic_add(A[0], 1.0, return_prev=True), id="atomic_add"),
        pytest.param(lambda A: T.tvm_storage_sync("shared"), id="storage_sync"),
        pytest.param(
            lambda A: tirx.op.masked_load(A, ir.const(0), tirx.call_extern("predicate_effect", ty="bool")),
            id="buffer_load_predicate",
        ),
    ],
)
def test_simplify_keeps_bind_with_side_effects(make_value):
    """Delete the unused consumer, but keep the operation with side effects."""
    A, _ = buffer_pair()
    value = make_value(A)
    if value.ty == ir.PrimType("void"):
        # Native synchronization returns void; its effect is a statement,
        # while an independent unused scalar result is still dead.
        effectful_bind = ir.Evaluate(value)
        unused_consumer = ir.Bind(ir.Var("unused", "int32"), ir.const(1))
    else:
        result = ir.Var("result", value.ty)
        effectful_bind = ir.Bind(result, value)
        unused_consumer = ir.Bind(ir.Var("unused", value.ty), result + 1)

    before_body = ir.SeqStmt([effectful_bind, unused_consumer])
    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([A], before_body))
    mod_expected = tvm_tirx.IRModule.from_expr(tirx.Function([A], effectful_bind))
    simplify_and_compare(mod_before, mod_expected)


@pytest.mark.parametrize("annotation", ["volatile_scope", "tirx.volatile"])
def test_simplify_keeps_bind_reading_volatile_buffer(annotation):
    """A read through an alias still observes the original buffer's volatility."""
    A = tensor("A", (1,), scope="local")
    alias_type = tirx.TensorType((1,), "float32", scope="local", layout="default", data=A.data)
    alias = ir.Var("alias", alias_type)
    value = ir.Var("value", "float32")

    def wrap_volatile(body):
        allocation = ir.Call(
            "tirx.alloc_tensor", [], ty_args=[A.ty], attrs={"tirx.volatile": True} if annotation == "tirx.volatile" else None
        )
        alias_binding = ir.Bind(alias, ir.Call("tirx.decl_tensor", [A.data], ty_args=[alias.ty]))
        body = ir.SeqStmt([alias_binding, body])
        if annotation == "volatile_scope":
            body = attribute(A.data, "volatile_scope", 1, body)
        body = ir.SeqStmt([ir.Bind(A, allocation), body])
        return tvm_tirx.IRModule.from_expr(tirx.Function([], body))

    volatile_read = ir.Bind(value, alias[0])
    unused_consumer = ir.Bind(ir.Var("unused", "float32"), value + 1.0)
    mod_before = wrap_volatile(ir.SeqStmt([volatile_read, unused_consumer]))
    mod_expected = wrap_volatile(volatile_read)
    simplify_and_compare(mod_before, mod_expected)


def test_simplify_keeps_binds_used_in_buffer_metadata():
    """Buffer shape, strides, offset and data pointer all count as variable uses."""
    source = tensor("source", (3,), "int32")
    size = ir.Var("size", "int32")
    stride = ir.Var("stride", "int32")
    offset = ir.Var("offset", "int32")
    data = ir.Var("data", ir.PtrType(ir.PrimType("float32")))
    layout = tirx.TileLayout.from_iters([Iter(size, stride, "m")], offset={"m": offset})
    view = ir.Var("view", tirx.TensorType((size,), "float32", data=data, layout=layout))

    body = ir.SeqStmt(
        [
            ir.Bind(size, source[0]),
            ir.Bind(stride, source[1]),
            ir.Bind(offset, source[2]),
            ir.Bind(data, tirx.call_pure_extern("get_pointer", ty=data.ty)),
            ir.Bind(view, ir.Call("tirx.decl_tensor", [data], ty_args=[view.ty])),
            ir.TensorStore(view, [0], 1.0),
        ]
    )
    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([source], body))
    simplify_and_compare(mod_before, mod_before)


def test_simplify_keeps_bind_used_in_nested_annotation():
    """A variable referenced only through annotation maps/arrays is still live."""
    source = tensor("source", (1,), "int32")
    output = tensor("output", (2,), "float32")
    tag = ir.Var("tag", "int32")
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        0,
        2,
        1,
        ir.ForKind.DEFAULT,
        ir.TensorStore(output, [i], 1.0),
        annotations={"test.metadata": {"nested": [tag]}},
    )
    body = bind_then(tag, source[0], loop)
    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([source, output], body))
    simplify_and_compare(mod_before, mod_before)


def test_simplify_inlines_constant_but_keeps_annotation_binding():
    """Inline ordinary uses, but keep the definition still used by an annotation."""
    output = tensor("output", (2,), "int32")
    tag = ir.Var("tag", "int32")
    i = ir.Var("i", "int32")
    annotations = {"test.metadata": {"nested": [tag]}}

    # tag = 4; for (..., annotation=[tag]): output[i] = tag + 1
    before_loop = ir.For(i, 0, 2, 1, ir.ForKind.DEFAULT, ir.TensorStore(output, [i], tag + 1), annotations=annotations)
    # The store simplifies to 5, while the annotation still requires tag = 4.
    expected_loop = ir.For(i, 0, 2, 1, ir.ForKind.DEFAULT, ir.TensorStore(output, [i], 5), annotations=annotations)
    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([output], bind_then(tag, 4, before_loop)))
    mod_expected = tvm_tirx.IRModule.from_expr(tirx.Function([output], bind_then(tag, 4, expected_loop)))
    simplify_and_compare(mod_before, mod_expected)


@pytest.mark.parametrize("site", ["function", "block", "allocation", "call", "cast"])
def test_simplify_keeps_bind_used_in_metadata_objects(site):
    """References outside ordinary expression children must keep bindings live."""
    source = tensor("source", (2,), "int32")
    output = tensor("output", (1,), "float32")
    tag = ir.Var("tag", "int32")
    metadata = {"test.metadata": {"nested": [tag]}}
    body = ir.TensorStore(output, [0], 1.0)

    if site == "block":
        body = attribute(None, "tl.sblock", {"name": "metadata", "no_realize": False, "annotations": metadata}, body)
    elif site == "allocation":
        scratch = tensor("scratch", (1,), "float32", scope="local")
        body = ir.SeqStmt([ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty], attrs=metadata)), body])
    elif site == "call":
        call = tirx.call_extern("observe", ty="float32")
        call = ir.Call(call.op, call.args, attrs=metadata, ty=call.ty)
        body = ir.TensorStore(output, [0], call)
    elif site == "cast":
        # Primitive Cast has no annotation field. Metadata at this source
        # position lives in the explicit native annotation statement; all
        # referenced Vars remain ordinary operands of that statement.
        value = prim.Cast("float32", source[1])
        body = ir.SeqStmt([annotation(metadata), ir.TensorStore(output, [0], value)])

    func = tirx.Function([source, output], bind_then(tag, source[0], body))
    if site == "function":
        func = func.with_attr("test.metadata", metadata["test.metadata"])
    mod_before = tvm_tirx.IRModule.from_expr(func)
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize("access", ["load", "store"])
def test_simplify_keeps_bind_used_only_in_buffer_predicate(access):
    """Both load and store predicates are uses even when the index is constant."""
    source = tensor("source", (1,), "int32")
    A, C = buffer_pair()
    predicate = ir.Var("predicate", "bool")
    if access == "load":
        load = tirx.op.masked_load(A, ir.const(0), predicate)
        body = ir.TensorStore(C, [0], load)
    else:
        body = ir.If(predicate, ir.TensorStore(C, [0], 1.0), None)
    body = bind_then(predicate, prim.LT(0, source[0]), body)
    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([source, A, C], body))
    simplify_and_compare(mod_before, mod_before)


@pytest.mark.parametrize(
    "store_uses_shared_value",
    [pytest.param(False, id="all_uses_dead"), pytest.param(True, id="live_store")],
)
def test_simplify_dead_bind_with_shared_nodes(store_uses_shared_value):
    """Shared nodes must neither keep dead producers alive nor hide live uses."""
    A, C = buffer_pair()
    cond = ir.Var("cond", "bool")
    value = ir.Var("value", "float32")
    shared = value + 1.0
    dead = ir.Bind(ir.Var("dead", "float32"), shared)
    store = ir.TensorStore(C, [0], shared if store_uses_shared_value else 0.0)

    # Reuse the exact same Bind in both branches, and the same expression in the live store.
    before_body = bind_then(value, A[0], ir.SeqStmt([ir.If(cond, dead, dead), store]))
    # Bind cleanup currently leaves the empty branches in place.
    expected_body = ir.SeqStmt([ir.If(cond, ir.Evaluate(0), ir.Evaluate(0)), store])
    if store_uses_shared_value:
        expected_body = bind_then(value, A[0], expected_body)

    mod_before = tvm_tirx.IRModule.from_expr(tirx.Function([A, C, cond], before_body))
    mod_expected = tvm_tirx.IRModule.from_expr(tirx.Function([A, C, cond], expected_body))
    simplify_and_compare(mod_before, mod_expected)


if __name__ == "__main__":
    tilelang.testing.main()
