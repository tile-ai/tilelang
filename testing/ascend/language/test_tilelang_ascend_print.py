"""Test T.print() on Ascend target."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def test_prim_expr_with_msg():
    """T.print(expr, msg=...) → debug_print_var via template."""

    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(32) as bx:
            T.print(bx + 42, msg="block_val")

    kernel = tilelang.compile(program, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "debug_print_var" in source
    assert "block_val" in source
    print("PASS: test_prim_expr_with_msg")


def test_shared_print():
    """T.print(shared buffer) → debug_print_buffer_value via template."""

    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(64) as bx:
            shared_buf = T.alloc_shared([16, 16], T.float32)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(16, 16):
                    shared_buf[i, j] = bx
            T.print(shared_buf)

    kernel = tilelang.compile(program, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "debug_print_buffer_value" in source
    assert "AscendC::DumpTensor" not in source
    assert "AscendC::printf" not in source
    print("PASS: test_shared_print")


def test_global_print():
    """T.print(global buffer) → debug_print_buffer_value via template."""

    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(16) as bx:
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    Q[bx, i] = bx
            if bx == 0:
                T.print(Q)

    kernel = tilelang.compile(program, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "debug_print_buffer_value" in source
    assert "AscendC::DumpTensor" not in source
    assert "AscendC::printf" not in source
    print("PASS: test_global_print")


def test_simtvf_inside():
    """T.print() inside SimtVF → debug_print_var via template."""

    @T.prim_func
    def program(Q: T.Tensor((4,), T.float32)):
        with T.Kernel(1) as bx:
            shared_buf = T.alloc_shared([16, 16], T.float32)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(16, 16):
                    shared_buf[i, j] = bx
                tid = T.get_thread_binding()
                T.print(tid, msg="tid")

    kernel = tilelang.compile(program, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "debug_print_var" in source
    print("PASS: test_simtvf_inside")


def test_float_var():
    """T.print(float_expr) → debug_print_var via template."""

    @T.prim_func
    def program(Q: T.Tensor((16, 16), T.float32)):
        with T.Kernel(32) as bx:
            val = bx * 3.14
            T.print(val, msg="float_val")

    kernel = tilelang.compile(program, target="ascend", pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
    source = kernel.get_kernel_source()
    assert "debug_print_var" in source
    print("PASS: test_float_var")


if __name__ == "__main__":
    tilelang.testing.main()
