from tilelang import tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _run_inject_assumes(func: tvm.tirx.PrimFunc):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    return tl.transform.InjectAssumes()(mod)


def test_inject_assumes_warns_on_local_var_assume(capfd):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), l: T.int32):
        with T.Kernel(1, threads=32):
            x = T.alloc_var(T.int32)
            x = l
            T.assume(x >= 0 and x < 128)
            A[x] = T.float32(0.0)

    _run_inject_assumes(main)

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "T.assume condition reads from T.alloc_var/local.var buffer `x`" in output
    assert "local.var is mutable" in output


def test_inject_assumes_does_not_warn_on_scalar_assume(capfd):
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), x: T.int32):
        with T.Kernel(1, threads=32):
            T.assume(x >= 0 and x < 128)
            A[x] = T.float32(0.0)

    _run_inject_assumes(main)

    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "T.assume condition reads from T.alloc_var/local.var buffer" not in output


if __name__ == "__main__":
    tilelang.testing.main()
