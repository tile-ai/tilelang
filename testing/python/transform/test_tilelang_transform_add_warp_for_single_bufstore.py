from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, expected):
    """Helper function to verify structural equality after transformations"""
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.AddWrapperForSingleBufStore()(mod)
    expected = tvm.IRModule.from_expr(expected.with_attr("global_symbol", "main"))
    tvm.ir.assert_structural_equal(mod["main"], expected["main"], True)


def test_fragment_store_wrapped_with_parallel():
    """
    Test that fragment buffer stores at depth 0 are wrapped with parallel loops.

    Conditions met:
    - Fragment buffer with index 0 access
    - Outside tile operations (tile_operation_depth == 0)
    - No thread bindings
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        # Fragment buffer allocation
        frag_buf = T.alloc_fragment((1,), "float32")

        # This store should be wrapped - meets all conditions
        frag_buf[0] = A[0]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # Wrapped with parallel loop
        for _ in T.parallel(0, 1):
            frag_buf[0] = A[0]

    _check(before, after)


def test_non_fragment_store_not_wrapped():
    """
    Test that non-fragment buffer stores are not wrapped.

    Conditions NOT met:
    - Buffer is not fragment scope (shared memory)
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        shared_buf = T.alloc_shared((1,), "float32")

        # This store should NOT be wrapped - not a fragment buffer
        shared_buf[0] = A[0]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        shared_buf = T.alloc_shared((1,), "float32")

        # No parallel wrapper added
        shared_buf[0] = A[0]

    _check(before, after)


def test_fragment_store_inside_tile_not_wrapped():
    """
    Test that fragment stores inside tile operations are not wrapped.

    Conditions NOT met:
    - Inside tile operation (tile_operation_depth > 0)
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # Inside parallel loop - should NOT be wrapped
        for i in T.parallel(0, 10):
            frag_buf[0] = A[i]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # No additional parallel wrapper inside existing parallel loop
        for i in T.parallel(0, 10):
            frag_buf[0] = A[i]

    _check(before, after)


def test_fragment_store_with_thread_binding_not_wrapped():
    """
    Test that fragment stores with thread bindings are not wrapped.

    Conditions NOT met:
    - Has thread binding variables
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            frag_buf = T.alloc_fragment((1,), "float32")

            # Inside thread binding - should NOT be wrapped
            frag_buf[0] = A[bx * 16 + by]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            frag_buf = T.alloc_fragment((1,), "float32")

            # No parallel wrapper added due to thread binding
            frag_buf[0] = A[bx * 16 + by]

    _check(before, after)


def test_fragment_non_zero_index_throws_error():
    """
    Test that fragment buffer access with non-zero indices throws error.

    Expected: ValueError for unsupported non-zero fragment access
    """

    @T.prim_func
    def invalid_func(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((4,), "float32")

        # This should throw error - non-zero index access
        frag_buf[2] = A[0]

    # This test should catch the ValueError
    import pytest

    pass_instance = tl.transform.AddWrapperForSingleBufStore()
    mod = tvm.IRModule.from_expr(invalid_func.with_attr("global_symbol", "main"))

    with pytest.raises(Exception) as exc_info:
        pass_instance(mod)

    error_msg = str(exc_info.value)
    assert any(msg in error_msg for msg in [
        "non-zero index",
        "fragment[0]",
        "Fragment buffer access",
        "not supported",
    ]), f"Unexpected error message: {error_msg}"


def test_mixed_fragment_and_shared_stores():
    """
    Test mixed buffer stores - only fragment stores should be wrapped.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")
        shared_buf = T.alloc_shared((1,), "float32")

        # Only this one should be wrapped
        frag_buf[0] = A[0]
        shared_buf[0] = A[1]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")
        shared_buf = T.alloc_shared((1,), "float32")

        # Fragment store wrapped, shared store not wrapped
        for _ in T.parallel(0, 1):
            frag_buf[0] = A[0]
        shared_buf[0] = A[1]

    _check(before, after)


def test_nested_control_flow_tracking():
    """
    Test that tile operation depth is correctly tracked through nested structures.
    """

    @T.prim_func
    def before(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # Depth 0 - should be wrapped
        frag_buf[0] = A[0]

        if A[0] > 0:
            # Depth 0 - should be wrapped
            frag_buf[0] = A[1]

            for i in T.parallel(0, 5):
                # Depth 1 - should NOT be wrapped
                frag_buf[0] = A[i]

        # Back to depth 0 - should be wrapped
        frag_buf[0] = A[2]

    @T.prim_func
    def after(A: T.Tensor((1024,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # All depth 0 stores wrapped
        for _ in T.parallel(0, 1):
            frag_buf[0] = A[0]

        if A[0] > 0:
            for _ in T.parallel(0, 1):
                frag_buf[0] = A[1]

            for i in T.parallel(0, 5):
                # Depth 1 - no wrapper
                frag_buf[0] = A[i]

        for _ in T.parallel(0, 1):
            frag_buf[0] = A[2]

    _check(before, after)


def test_pipelined_loop_depth_tracking():
    """
    Test that pipelined loops are considered tile operations.
    """

    @T.prim_func
    def before(A: T.Tensor((32,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # Inside pipelined loop - should NOT be wrapped
        for ko in T.Pipelined(32, num_stages=3):
            frag_buf[0] = A[ko]

    @T.prim_func
    def after(A: T.Tensor((32,), "float32")):
        frag_buf = T.alloc_fragment((1,), "float32")

        # No wrapper inside pipelined loop (tile operation)
        for ko in T.Pipelined(32, num_stages=3):
            frag_buf[0] = A[ko]

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
