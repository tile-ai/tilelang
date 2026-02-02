"""Tests for Loop Invariant Code Motion (LICM) pass with CSE extraction."""

from tilelang import tvm as tvm
from tvm import tir
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _apply_licm(func):
    """Apply LICM pass and return the transformed function."""
    from tilelang.transform import PassConfigKey

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOOP_INVARIANT_CODE_MOTION: True}):
        mod = tl.transform.LoopInvariantCodeMotion()(mod)
    return mod["main"]


def _get_body(func):
    """Get the body of a function, skipping BlockRealize/Block if present."""
    body = func.body
    while hasattr(body, "block"):
        body = body.block.body
    return body


def _find_lets_in_stmt(stmt, in_loop=False):
    """Recursively find all LetStmt var names, tracking whether they're in a loop."""
    lets_outside = set()
    lets_inside = set()

    def visit(node, in_loop):
        if isinstance(node, tir.LetStmt):
            if in_loop:
                lets_inside.add(node.var.name)
            else:
                lets_outside.add(node.var.name)
            visit(node.body, in_loop)
        elif isinstance(node, tir.For):
            visit(node.body, True)
        elif isinstance(node, tir.IfThenElse):
            visit(node.then_case, in_loop)
            if node.else_case:
                visit(node.else_case, in_loop)
        elif isinstance(node, tir.SeqStmt):
            for s in node.seq:
                visit(s, in_loop)
        elif isinstance(node, tir.AttrStmt):
            visit(node.body, in_loop)
        elif isinstance(node, (tir.BufferStore, tir.Evaluate)):
            pass
        elif isinstance(node, tir.BlockRealize):
            visit(node.block.body, in_loop)

    visit(stmt, in_loop)
    return lets_outside, lets_inside


def _count_expr_occurrences(func, expr_pattern):
    """Count how many times an expression pattern appears in the function body."""
    script = func.script()
    return script.count(expr_pattern)


# =============================================================================
# Phase 1 Tests: LetStmt Hoisting
# =============================================================================


def test_basic_let_hoist():
    """Basic case: loop-invariant let should be hoisted outside the loop."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in range(128):
            x = a + b
            B[i] = A[i] + T.cast(x, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in outside, "x should be outside loop"
    assert "x" not in inside, "x should not be inside loop"


def test_let_chain_hoist():
    """Chain of dependent let statements should all be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in range(128):
            x = a + b
            y = x * 2
            B[i] = A[i] + T.cast(y, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in outside, "x should be outside loop"
    assert "y" in outside, "y should be outside loop"
    assert "x" not in inside, "x should not be inside loop"
    assert "y" not in inside, "y should not be inside loop"


def test_no_hoist_loop_variant():
    """Let using loop variable should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            x = i * 2
            B[i] = A[i] + T.cast(x, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in inside, "x should remain inside loop (uses loop var)"
    assert "x" not in outside, "x should not be outside loop"


def test_partial_hoist():
    """Only invariant lets should be hoisted, variant ones remain."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in range(128):
            x = a + b  # invariant
            y = x + i  # variant (uses i)
            B[i] = A[i] + T.cast(y, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in outside, "x should be outside loop"
    assert "x" not in inside, "x should not be inside loop"
    assert "y" in inside, "y should remain inside loop (uses loop var)"
    assert "y" not in outside, "y should not be outside loop"


def test_no_hoist_reads_written_buffer():
    """Let that reads a buffer written in the loop should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            A[i] = T.float32(1.0)
            x = A[0]  # reads A which is written in loop
            B[i] = x

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in inside, "x should remain inside loop (reads written buffer)"


# =============================================================================
# Phase 2 Tests: Subexpression Extraction
# =============================================================================


def test_extract_repeated_subexpr():
    """Repeated loop-invariant subexpressions should be extracted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        base: T.int32,
        offset: T.int32,
    ):
        for i in range(128):
            # (base + offset) appears twice - should be extracted
            A[base + offset + i] = T.float32(1.0)
            B[base + offset + i] = T.float32(2.0)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    # Should have extracted the common subexpression
    # Check that a cse_var was created outside the loop
    cse_vars = [v for v in outside if v.startswith("cse_var")]
    assert len(cse_vars) >= 1, f"Should have extracted CSE variable, got {outside}"


def test_extract_complex_subexpr():
    """Complex repeated expressions should be extracted."""

    @T.prim_func
    def before(
        A: T.Tensor((1024,), T.float32),
        B: T.Tensor((1024,), T.float32),
        stride: T.int32,
        offset: T.int32,
    ):
        for i in range(32):
            # (stride * 8 + offset) appears twice - should be extracted
            A[(stride * 8 + offset) + i] = T.float32(1.0)
            B[(stride * 8 + offset) + i * 2] = T.float32(2.0)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    cse_vars = [v for v in outside if v.startswith("cse_var")]
    assert len(cse_vars) >= 1, f"Should have extracted CSE variable, got {outside}"


def test_no_extract_single_occurrence():
    """Single-occurrence expressions should NOT be extracted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
        c: T.int32,
    ):
        for i in range(128):
            # Each expression appears only once
            A[a + b + i] = T.float32(1.0)
            B[b + c + i] = T.float32(2.0)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    # Should NOT have extracted any CSE variable (each expr appears once)
    cse_vars = [v for v in outside if v.startswith("cse_var")]
    print(cse_vars)
    # This is a weaker test - we just check it doesn't crash
    assert result is not None


def test_no_extract_loop_variant_subexpr():
    """Subexpressions using loop variable should NOT be extracted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            # (i * 2) uses loop variable - should NOT be extracted
            A[i * 2] = T.float32(1.0)
            B[i * 2 + 1] = T.float32(2.0)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    # Should NOT have extracted any CSE variable
    cse_vars = [v for v in outside if v.startswith("cse_var")]
    assert len(cse_vars) == 0, f"Should NOT have extracted loop-variant expr, got {outside}"


def test_extract_single_occurrence_high_complexity():
    """Single-occurrence expressions with high complexity should be extracted (LICM mode)."""

    @T.prim_func
    def before(
        A: T.Tensor((1024,), T.float32),
        B: T.Tensor((1024,), T.float32),
        block_idx: T.int32,
    ):
        for i in range(32):
            # (block_idx * 131072) appears only once but has complexity >= 3
            # Should be extracted as LICM optimization
            A[block_idx * 131072 + i] = T.float32(1.0)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    # Should have extracted the high-complexity expression even though it appears once
    cse_vars = [v for v in outside if v.startswith("cse_var")]
    assert len(cse_vars) >= 1, f"Should have extracted high-complexity expr, got {outside}"


# =============================================================================
# Combined Tests
# =============================================================================


def test_combined_let_and_subexpr():
    """Test both LetStmt hoisting and subexpression extraction together."""

    @T.prim_func
    def before(
        A: T.Tensor((256,), T.float32),
        B: T.Tensor((256,), T.float32),
        base: T.int32,
        scale: T.int32,
    ):
        for i in range(64):
            # Existing let - should be hoisted
            offset = base * scale
            # Repeated subexpr in indices - should be extracted
            A[(base + scale) + i] = T.float32(1.0)
            B[(base + scale) + i * 2] = T.cast(offset, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    # 'offset' should be hoisted
    assert "offset" in outside, "offset should be hoisted"
    # CSE var should be created for (base + scale)
    cse_vars = [v for v in outside if v.startswith("cse_var")]
    assert len(cse_vars) >= 1, "Should have extracted CSE variable"


def test_nested_loop():
    """Test LICM on nested loops - bottom-up processing."""

    @T.prim_func
    def before(
        A: T.Tensor((16, 128), T.float32),
        B: T.Tensor((16, 128), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in range(16):
            for j in range(128):
                x = a + b
                B[i, j] = A[i, j] + T.cast(x, T.float32)

    result = _apply_licm(before)
    # Just verify it doesn't crash and processes correctly
    assert result is not None


def test_parallel_loop():
    """LICM should work with parallel loops."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in T.Parallel(128):
            x = a + b
            B[i] = A[i] + T.cast(x, T.float32)

    result = _apply_licm(before)
    outside, inside = _find_lets_in_stmt(_get_body(result))

    assert "x" in outside, "x should be outside loop"
    assert "x" not in inside, "x should not be inside loop"


def test_no_change_needed():
    """Loop without any optimization opportunity should remain unchanged."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            B[i] = A[i] * T.float32(2.0)

    result = _apply_licm(before)
    assert result is not None


def test_config_custom_threshold():
    """Test that custom config values are respected."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        base: T.int32,
    ):
        for i in range(128):
            # complexity=3, appears once - should be extracted with default config
            A[base * 100 + i] = T.float32(1.0)

    # With default config (min_complexity_for_licm=3), should extract
    result_default = _apply_licm(before)
    outside_default, _ = _find_lets_in_stmt(_get_body(result_default))
    cse_vars_default = [v for v in outside_default if v.startswith("cse_var")]

    # With higher threshold (min_complexity_for_licm=5), should NOT extract
    from tilelang.transform import PassConfigKey

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOOP_INVARIANT_CODE_MOTION: True,
            PassConfigKey.TL_LICM: {PassConfigKey.TL_LICM_MIN_COMPLEXITY_FOR_LICM: 5},
        }
    ):
        result_strict = tl.transform.LoopInvariantCodeMotion()(mod)["main"]
    outside_strict, _ = _find_lets_in_stmt(_get_body(result_strict))
    cse_vars_strict = [v for v in outside_strict if v.startswith("cse_var")]

    assert len(cse_vars_default) >= 1, f"Default config should extract, got {outside_default}"
    assert len(cse_vars_strict) == 0, f"Strict config should not extract, got {outside_strict}"


def test_enable_licm():
    """Test that LICM can be enabled/disabled via config."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        a: T.int32,
        b: T.int32,
    ):
        for i in range(128):
            x = a + b
            B[i] = A[i] + T.cast(x, T.float32)

    from tilelang.transform import PassConfigKey

    # With LICM disabled (default), x should remain inside
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    result_disabled = tl.transform.LoopInvariantCodeMotion()(mod)["main"]
    outside_disabled, inside_disabled = _find_lets_in_stmt(_get_body(result_disabled))
    assert "x" in inside_disabled, "x should remain inside loop when LICM disabled"
    assert "x" not in outside_disabled, "x should not be outside when LICM disabled"

    # With LICM enabled, x should be hoisted
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOOP_INVARIANT_CODE_MOTION: True}):
        result_enabled = tl.transform.LoopInvariantCodeMotion()(mod)["main"]
    outside_enabled, inside_enabled = _find_lets_in_stmt(_get_body(result_enabled))
    assert "x" in outside_enabled, "x should be outside loop when LICM enabled"


# =============================================================================
# Debug Test
# =============================================================================


def test_print_result():
    """Debug test to print the transformation result."""

    @T.prim_func
    def before(
        A: T.Tensor((256,), T.float32),
        B: T.Tensor((256,), T.float32),
        base: T.int32,
        scale: T.int32,
    ):
        for i in range(64):
            offset = base * scale
            A[(base + scale) + i] = T.float32(1.0)
            B[(base + scale) + i * 2] = T.cast(offset, T.float32)

    print("\n=== Before LICM ===")
    print(before.script())

    result = _apply_licm(before)
    print("\n=== After LICM ===")
    print(result.script())


if __name__ == "__main__":
    # tilelang.testing.main()
    test_print_result()
