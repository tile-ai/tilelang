from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LoopUnswitching()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), map_free_vars=True)


def _check_with_config(original, transformed, config):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config=config):
        mod = tl.transform.LoopUnswitching()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), map_free_vars=True)


def test_basic_hoist():
    """Basic case: loop-invariant if should be hoisted outside the loop."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                B[i] = A[i]
        else:
            for _i in range(128):
                T.evaluate(0)

    _check(before, expected)


def test_hoist_with_else():
    """Conservative: if with non-trivial else should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = A[i] * T.float32(2.0)

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # Should remain unchanged
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = A[i] * T.float32(2.0)

    _check(before, expected)


def test_no_hoist_loop_variant():
    """If condition depends on loop variable, should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            if i < 64:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        # Should remain unchanged
        for i in range(128):
            if i < 64:
                B[i] = A[i]

    _check(before, expected)


def test_no_hoist_reads_written_buffer():
    """If condition reads a buffer written in the loop, should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            A[i] = T.float32(1.0)
            if A[0] > 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        # Should remain unchanged
        for i in range(128):
            A[i] = T.float32(1.0)
            if A[0] > 0:
                B[i] = A[i]

    _check(before, expected)


def test_hoist_with_other_stmts():
    """Conservative: if with other side-effecting statements should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            C[i] = A[i]
            if cond[0] > 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # Should remain unchanged
        for i in range(128):
            C[i] = A[i]
            if cond[0] > 0:
                B[i] = A[i]

    _check(before, expected)


def test_nested_loop_inner_invariant():
    """Loop-invariant if should be hoisted to outermost possible level."""

    @T.prim_func
    def before(
        A: T.Tensor((16, 128), T.float32),
        B: T.Tensor((16, 128), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(16):
            for j in range(128):
                if cond[0] > 0:
                    B[i, j] = A[i, j]

    @T.prim_func
    def expected(
        A: T.Tensor((16, 128), T.float32),
        B: T.Tensor((16, 128), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # if is hoisted outside both loops since cond[0] is invariant to both
        if cond[0] > 0:
            for i in range(16):
                for j in range(128):
                    B[i, j] = A[i, j]
        else:
            for _i in range(16):
                for _j in range(128):
                    T.evaluate(0)

    _check(before, expected)


def test_parallel_loop():
    """Loop-invariant if in parallel loop."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in T.Parallel(128):
            if cond[0] > 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in T.Parallel(128):
                B[i] = A[i]
        else:
            for _i in T.Parallel(128):
                T.evaluate(0)

    _check(before, expected)


def test_hoist_let_bound_variable():
    """If condition uses a Let-bound variable, both should be hoisted together."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((1,), T.float32),
    ):
        for i in range(128):
            pos = C[0]
            if pos >= T.float32(0):
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((1,), T.float32),
    ):
        # Let binding is hoisted before the if, redundant inner LetStmt is removed
        pos = C[0]
        if pos >= T.float32(0):
            for i in range(128):
                B[i] = A[i]
        else:
            for _i in range(128):
                T.evaluate(0)

    _check(before, expected)


def test_hoist_multiple_let_bound_variables():
    """If condition uses multiple Let-bound variables, all should be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((2,), T.float32),
    ):
        for i in range(128):
            x = C[0]
            y = C[1]
            if x + y >= T.float32(0):
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((2,), T.float32),
    ):
        # Let bindings are hoisted before the if, redundant inner LetStmts are removed
        x = C[0]
        y = C[1]
        if x + y >= T.float32(0):
            for i in range(128):
                B[i] = A[i]
        else:
            for _i in range(128):
                T.evaluate(0)

    _check(before, expected)


def test_multiple_identical_conditions():
    """Multiple if statements with the same condition should all be replaced."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            if cond[0] > 0:
                C[i] = A[i] * T.float32(2.0)

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                B[i] = A[i]
                C[i] = A[i] * T.float32(2.0)
        else:
            for _i in range(128):
                T.evaluate(0)
                T.evaluate(0)

    _check(before, expected)


def test_multiple_identical_conditions_with_else():
    """Conservative: multiple if-else statements should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = T.float32(0)
            if cond[0] > 0:
                C[i] = A[i] * T.float32(2.0)
            else:
                C[i] = T.float32(1)

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # Should remain unchanged
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = T.float32(0)
            if cond[0] > 0:
                C[i] = A[i] * T.float32(2.0)
            else:
                C[i] = T.float32(1)

    _check(before, expected)


def test_no_hoist_let_bound_loop_variant():
    """Let-bound variable depends on loop var, condition should NOT be hoisted."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            idx = i % 2
            if idx == 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        # Should remain unchanged since idx depends on loop variable i
        for i in range(128):
            idx = i % 2
            if idx == 0:
                B[i] = A[i]

    _check(before, expected)


def test_no_hoist_multiple_let():
    @tilelang.jit()
    def get_fused_mapping_kernel(topk_idx: T.Tensor[(1,), T.int32]):
        with T.Kernel():
            _tmp1 = T.alloc_shared((1,), "int")
            for i in T.serial(0, 4, 2):
                _tmp2 = topk_idx[i]
                T.assume(0 <= _tmp2 < 1)
                if _tmp2 != -1:
                    T.atomic_add(_tmp1[_tmp2], 1)

    get_fused_mapping_kernel.compile()


def test_no_hoist_thread_idx_predicate():
    """Do not unswitch predicates that depend on threadIdx.

    These predicates are loop-invariant, but hoisting them can split execution
    across threads and break later synchronization insertion passes.
    """

    @T.prim_func
    def before(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), dtype=T.int32)
        B = T.match_buffer(B_ptr, (256,), dtype=T.int32)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                for i in T.unroll(0, 2):
                    B[tx] = A[tx]
                    if tx == 0:
                        B[i] = T.int32(1)

    _check(before, before)


def test_hoist_with_else_when_enabled():
    """Allow hoisting if-else when explicitly enabled."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = A[i] * T.float32(2.0)

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                B[i] = A[i]
        else:
            for i in range(128):
                B[i] = A[i] * T.float32(2.0)

    _check_with_config(
        before,
        expected,
        config={PassConfigKey.TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE: True},
    )


def test_hoist_with_other_stmts_when_enabled():
    """Allow hoisting when loop contains other side effects if enabled."""

    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            C[i] = A[i]
            if cond[0] > 0:
                B[i] = A[i]

    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                C[i] = A[i]
                B[i] = A[i]
        else:
            for i in range(128):
                C[i] = A[i]
                T.evaluate(0)

    _check_with_config(
        before,
        expected,
        config={PassConfigKey.TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE: True},
    )


def test_no_hoist_thread_idx_predicate_even_when_enabled():
    """The aggressive option must not unswitch per-thread predicates."""

    @T.prim_func
    def before(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), dtype=T.int32)
        B = T.match_buffer(B_ptr, (256,), dtype=T.int32)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                for i in T.unroll(0, 2):
                    B[tx] = A[tx]
                    if tx == 0:
                        B[i] = T.int32(1)

    _check_with_config(
        before,
        before,
        config={PassConfigKey.TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE: True},
    )


if __name__ == "__main__":
    tilelang.testing.main()
