from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
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
    """If with else branch should be hoisted with both branches."""

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
    """If with other statements in loop body."""

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


if __name__ == "__main__":
    tilelang.testing.main()
