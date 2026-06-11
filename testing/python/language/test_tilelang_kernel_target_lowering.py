import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_default_thread_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(8) as bx:
            A[bx] = T.float32(1)

    return main.with_attr("global_symbol", "main")


def _make_explicit_thread_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(8, threads=4) as bx:
            tx = T.get_thread_binding()
            A[bx] = T.cast(tx, "float32")

    return main.with_attr("global_symbol", "main")


def _make_unit_thread_kernel():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32")):
        with T.Kernel(8, threads=1) as bx:
            tx = T.get_thread_binding()
            A[bx] = T.cast(tx, "float32")

    return main.with_attr("global_symbol", "main")


def _lower_kernel_launch(func, target, lower_pass):
    target = tvm.target.Target(target)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    return lower_pass()(mod)


def test_kernel_ast_is_target_neutral():
    script = _make_default_thread_kernel().script()

    assert "threadIdx" not in script
    assert "blockIdx" not in script
    assert "tilelang.kernel_dim_kind" in script


def test_kernel_lower_to_cuda_thread_bindings():
    mod = _lower_kernel_launch(
        _make_default_thread_kernel(),
        {"kind": "cuda", "arch": "sm_80"},
        tilelang.transform.LowerKernelLaunchToThreadBinding,
    )
    script = mod["main"].script()

    assert "blockIdx.x" in script
    assert "threadIdx.x" in script


def test_kernel_lower_to_cpu_serial_grid_without_default_threads():
    mod = _lower_kernel_launch(
        _make_default_thread_kernel(),
        "c",
        tilelang.transform.LowerKernelLaunchToSerial,
    )
    script = mod["main"].script()

    assert "blockIdx" not in script
    assert "threadIdx" not in script
    assert "for tx" not in script


def test_kernel_lower_to_cpu_serial_ignores_explicit_threads(capfd):
    mod = _lower_kernel_launch(
        _make_explicit_thread_kernel(),
        "c",
        tilelang.transform.LowerKernelLaunchToSerial,
    )
    script = mod["main"].script()
    captured = capfd.readouterr()

    assert "threadIdx" not in script
    assert "for tx" not in script
    assert "for ty" not in script
    assert "for tz" not in script
    assert 'A[bx] = T.Cast("float32", 0)' in script
    assert "thread extent `4` is ignored by serial backends" in captured.out + captured.err


def test_kernel_lower_to_cpu_serial_unit_thread_without_warning(capfd):
    mod = _lower_kernel_launch(
        _make_unit_thread_kernel(),
        "c",
        tilelang.transform.LowerKernelLaunchToSerial,
    )
    script = mod["main"].script()
    captured = capfd.readouterr()

    assert "threadIdx" not in script
    assert "for tx" not in script
    assert 'A[bx] = T.Cast("float32", 0)' in script
    assert "ignored by serial backends" not in captured.out + captured.err


if __name__ == "__main__":
    tilelang.testing.main()
