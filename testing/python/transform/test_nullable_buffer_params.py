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


if __name__ == "__main__":
    tilelang.testing.main()
