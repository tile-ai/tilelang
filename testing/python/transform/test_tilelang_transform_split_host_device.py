from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def split_host_device_with_user_assume():
    n = T.dynamic("n")

    @T.prim_func
    def main(a: T.Tensor[(n,), T.int32]):
        T.assume(n >= 233 and n <= 1000)
        with T.Kernel(1, threads=128):
            for i in T.serial(T.ceildiv(n - 233, 123)):
                a[i] = 1

    return main


def test_split_host_device_with_user_assume(with_B=False, with_bias=False):
    tester = split_host_device_with_user_assume()
    mod = tvm.IRModule({tester.attrs["global_symbol"]: tester})
    # There are some dependent pass that need to be run before SplitHostDevice
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("cuda", "c"))(mod)
    mod = tl.transform.InjectAssumes()(mod)
    tilelang.analysis.ASTPrinter()(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    tilelang.analysis.ASTPrinter()(mod)

    mod2 = tl.transform.SplitHostDevice()(mod)

    assert len(mod2.functions) == 2


if __name__ == "__main__":
    tilelang.testing.main()
