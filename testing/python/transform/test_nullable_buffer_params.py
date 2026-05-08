import torch
import tilelang
import tilelang.testing
from tilelang import language as T


def test_nullable_shared_shape():
    """Test that buffers sharing a shape variable can be nullable."""

    @tilelang.jit
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def test_kernel(
            a: T.Tensor[(m,), T.int32],
            b: T.Tensor[(m,), T.int32],
            c: T.Tensor[(m,), T.int32],
        ):
            with T.Kernel(1, threads=64):
                tx = T.get_thread_binding()
                if tx == 0:
                    T.print(m)

        return test_kernel

    m = 200
    kernel = get_kernel()

    # Create test tensors
    tensor_a = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    tensor_b = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    tensor_c = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)

    print("Test 1: All tensors provided")
    kernel(tensor_a, tensor_b, tensor_c)
    print("✓ PASS: All tensors provided")

    print("\nTest 2: Only first tensor provided")
    kernel(tensor_a, None, None)
    print("✓ PASS: Only first tensor provided")

    print("\nTest 3: Only middle tensor provided")
    kernel(None, tensor_b, None)
    print("✓ PASS: Only middle tensor provided")

    print("\nTest 4: Only last tensor provided")
    kernel(None, None, tensor_c)
    print("✓ PASS: Only last tensor provided")

    print("\nTest 5: First and last tensors provided")
    kernel(tensor_a, None, tensor_c)
    print("✓ PASS: First and last tensors provided")

    print("\nTest 6: All tensors are None (should fail)")
    try:
        kernel(None, None, None)
        print("✗ FAIL: Should have raised an error")
        return False
    except RuntimeError as e:
        if "at least one non-null buffer" in str(e):
            print(f"✓ PASS: Correctly rejected with error: {e}")
        else:
            print(f"✗ FAIL: Wrong error message: {e}")
            return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    return True


def test_nullable_single_source_shape():
    """Test that a single buffer with a symbolic shape var must be non-null.

    This guards against the previous segfault when binding m from x.shape[0]
    with x == None.
    """

    @tilelang.jit
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def sample_kernel(x: T.Tensor[(m,), T.int32]):
            with T.Kernel(1, threads=1):
                tx = T.get_thread_binding()
                if tx == 0:
                    T.print(m)

        return sample_kernel

    m = 16
    kernel = get_kernel()

    # Provide a valid tensor: should run
    x = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    kernel(x)

    # Passing None should not segfault; m binds to 0 and kernel is a no-op
    kernel(None)


def test_nullable_shared_shape_with_no_source_buffers_but_other_tensor_present():
    """Test that unused buffers sharing a symbolic shape var can both be None.

    Repro for:
      - Two (or more) unused buffers have shape (m,)
      - All buffers that mention `m` are passed as None
      - Another (non-null) tensor argument exists, but does not mention `m`

    TVM requires at least one non-null buffer to bind `m` when it appears in multiple
    buffers. TileLang should handle this gracefully for truly-unused nullable buffers.
    """

    @tilelang.jit(execution_backend="tvm_ffi")
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def test_kernel(
            a: T.Tensor[(m,), T.float16],
            b: T.Tensor[(m,), T.float16],
            out: T.Tensor[(1,), T.float16],
        ):
            with T.Kernel(1, threads=32):
                fragment = T.alloc_fragment((1,), T.float32)
                T.copy(out[0], fragment)
                T.copy(fragment, out[0])

        return test_kernel

    kernel = get_kernel()

    out = torch.randn((1,), device="cuda", dtype=torch.float16)
    out_ref = out.clone()

    # Both `a` and `b` are None; they also share the symbolic shape var `m`.
    # This should run because `a`/`b` are unused by the kernel body.
    kernel(None, None, out)
    torch.testing.assert_close(out, out_ref)


if __name__ == "__main__":
    tilelang.testing.main()
