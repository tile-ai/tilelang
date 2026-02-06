from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, expected):
    """Helper function to verify structural equality after transformations"""
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LegalizeNegativeIndex()(mod)
    expected = tvm.IRModule.from_expr(expected.with_attr("global_symbol", "main"))
    tvm.ir.assert_structural_equal(mod["main"], expected["main"], True)


def test_buffer_load_negative_index_legalized():
    """
    Test that negative indices are legalized by adding buffer extent.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        value = A[-1]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        value = A[1023]  # A[-1] becomes A[1023]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_mixed_negative_positive_indices():
    """
    Test mixed negative and positive indices - only negative ones are legalized.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        value = A[-1, 10]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        value = A[1023, 10]  # A[-1, 10] becomes A[1023, 10]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_multiple_negative_indices():
    """
    Test multiple negative indices in different dimensions.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[-1, -2, -3]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[1023, 510, 253]  # -1+1024=1023, -2+512=510, -3+256=253
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_negative_index_in_expression():
    """
    Test negative index as part of a larger expression.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[-i]
            B[-i] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[1024 - i]
            B[1024 - i] = value

    _check(before, after)


def test_buffer_load_non_negative_index_unchanged():
    """
    Test that non-negative indices remain unchanged.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # No changes expected for non-negative indices
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_unknown_sign_index_warning():
    """
    Test that indices with unknown sign trigger warnings but are processed.
    This test mainly checks that the pass doesn't crash on unknown signs.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        i = T.Var("i", T.int32)
        value = A[i]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        i = T.Var("i", T.int32)
        # Unknown sign indices should remain unchanged
        value = A[i]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value

    _check(before, after)


def test_buffer_load_vector_index_negative_broadcast():
    """
    Test negative indices in vectorized operations (broadcast case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        value = A[T.Broadcast(1023, 4)]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    _check(before, after)


def test_buffer_load_vector_index_negative_ramp():
    """
    Test negative indices in vectorized operations (ramp case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Ramp(-4, 1, 4)  # noqa: F841
        value = A[T.Ramp(1020, 1, 4)]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value

    _check(before, after)


def test_buffer_load_nested_buffer_loads():
    """
    Test legalization with nested buffer load expressions.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        inner_val = A[-1, 10]
        outer_val = A[inner_val.astype(T.int32), -2]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = outer_val

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        inner_val = A[1023, 10]
        outer_val = A[inner_val.astype(T.int32), 510]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = outer_val

    _check(before, after)


def test_buffer_store_negative_index():
    """
    Test negative indices in buffer store operations are legalized.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        A[-1] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        A[1023] = 42.0

    _check(before, after)


def test_buffer_store_mixed_negative_positive_indices():
    """
    Test mixed negative and positive indices in buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        A[-1, 10] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        A[1023, 10] = 42.0

    _check(before, after)


def test_buffer_store_multiple_negative_indices():
    """
    Test multiple negative indices in different dimensions for buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        A[-1, -2, -3] = 42.0

    @T.prim_func
    def after(A: T.Tensor((1024, 512, 256), T.float32)):
        A[1023, 510, 253] = 42.0  # -1+1024=1023, -2+512=510, -3+256=253

    _check(before, after)


def test_buffer_store_negative_index_in_expression():
    """
    Test negative index as part of a larger expression in buffer store.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[-i] = i * 2.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[1024 - i] = i * 2.0

    _check(before, after)


def test_buffer_store_vector_index_negative_broadcast():
    """
    Test negative indices in vectorized store operations (broadcast case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        values = T.Broadcast(42.0, 4)
        A[vec] = values

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        values = T.Broadcast(42.0, 4)
        A[T.Broadcast(1023, 4)] = values

    _check(before, after)


def test_buffer_store_vector_index_negative_ramp():
    """
    Test negative indices in vectorized store operations (ramp case).
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        values = T.Ramp(0.0, 1.0, 4)  # values: [0.0, 1.0, 2.0, 3.0]
        A[vec] = values

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Ramp(-4, 1, 4)  # noqa: F841
        values = T.Ramp(0.0, 1.0, 4)
        A[T.Ramp(1020, 1, 4)] = values

    _check(before, after)


def test_buffer_store_nested_in_condition():
    """
    Test negative index buffer store within conditional statements.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[-1] = 42.0
        else:
            A[-2] = 24.0

    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[1023] = 42.0
        else:
            A[1022] = 24.0

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
