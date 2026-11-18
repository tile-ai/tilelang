import torch
import pytest
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


def _get_nullable_shared_shape_kernel():
    """Build the TVM-FFI kernel used by nullable-first device-ID tests."""

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
                for i in T.Parallel(1):
                    fragment[i] = fragment[i] + 1.0
                T.copy(fragment, out[0])

        return test_kernel

    return get_kernel()


def test_nullable_shared_shape_with_no_source_buffers_but_other_tensor_present():
    """Test nullable unused buffers with a non-null tensor on the default device."""

    kernel = _get_nullable_shared_shape_kernel()

    out = torch.randn((1,), device="cuda", dtype=torch.float16)
    out_ref = out + 1.0

    # Both `a` and `b` are None; they also share the symbolic shape var `m`.
    # This should run because `a`/`b` are unused by the kernel body.
    kernel(None, None, out)
    torch.testing.assert_close(out, out_ref)


def test_nullable_shared_shape_with_no_source_buffers_but_other_tensor_present_on_cuda1():
    """Test device-ID binding when cuda:0 is current but the tensor is on cuda:1."""

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    cuda0_capability = torch.cuda.get_device_capability(0)
    cuda1_capability = torch.cuda.get_device_capability(1)
    if cuda0_capability != cuda1_capability:
        pytest.skip(
            "reuses one JIT kernel compiled for cuda:0, so both devices must have "
            "the same compute capability; "
            f"cuda:0={cuda0_capability}, cuda:1={cuda1_capability}"
        )

    kernel = _get_nullable_shared_shape_kernel()
    original_device = torch.cuda.current_device()

    # Compile the kernel on cuda:0, then invoke it on a cuda:1 tensor while
    # cuda:0 remains the current device. This exercises the FFI device switch.
    with torch.cuda.device(0):
        out_cuda0 = torch.full((1,), 1.0, device="cuda:0", dtype=torch.float16)
        kernel(None, None, out_cuda0)
        torch.testing.assert_close(out_cuda0, torch.full_like(out_cuda0, 2.0))

        out_cuda1 = torch.full((1,), 3.0, device="cuda:1", dtype=torch.float16)
        expected_cuda1 = torch.full_like(out_cuda1, 4.0)
        assert torch.cuda.current_device() == 0

        # The tensor's device must drive the packed FFI device binding, even
        # though cuda:0 is current. The output change proves the kernel ran.
        kernel(None, None, out_cuda1)
        torch.testing.assert_close(out_cuda1, expected_cuda1)
        assert torch.cuda.current_device() == 0

    # The context manager must restore the device that was current on entry.
    assert torch.cuda.current_device() == original_device


if __name__ == "__main__":
    tilelang.testing.main()
