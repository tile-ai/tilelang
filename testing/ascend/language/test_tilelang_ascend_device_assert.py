"""Test T.device_assert() on Ascend target."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def test_device_assert():
    """T.device_assert(cond, no_stack_info=True) → device_assert via template."""

    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(32) as bx:
            T.device_assert(bx >= 0)

    kernel = tilelang.compile(program)
    source = kernel.get_kernel_source()
    assert "device_assert" in source
    print("PASS: test_device_assert")


def test_device_assert_in_vf():
    """T.device_assert() inside T.SimtVF → device_assert via template."""

    @T.prim_func
    def program(Q: T.Tensor((4,), T.float32)):
        with T.Kernel(1) as bx:
            shared_buf = T.alloc_shared([16, 16], T.float32)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(16, 16):
                    shared_buf[i, j] = bx
                tid = T.get_thread_binding()
                T.device_assert(tid >= 0)

    kernel = tilelang.compile(program)
    source = kernel.get_kernel_source()
    assert "device_assert" in source
    print("PASS: test_device_assert_in_vf")


if __name__ == "__main__":
    tilelang.testing.main()
