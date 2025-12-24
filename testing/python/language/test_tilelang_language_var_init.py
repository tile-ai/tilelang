import tilelang
import tilelang.language as T
import tilelang.testing


def test_var_assign() -> None:
    @tilelang.jit(out_idx=-1)
    def jit_kernel():
        @T.prim_func
        def test_var_assign(A: T.Tensor((2,), T.int32)):
            with T.Kernel(1) as _:
                a = T.alloc_var(T.int32, init=1)
                b = T.alloc_var(T.int32, init=a)  # b gets value of a
                a = 2
                d = T.alloc_var(T.int32, init=a)  # c gets new value of a
                A[0] = b
                A[1] = d

        print(test_var_assign)
        return test_var_assign

    kernel = jit_kernel()
    print(kernel.get_kernel_source())
    res = kernel()
    assert res[0] == 1
    assert res[1] == 2


if __name__ == "__main__":
    tilelang.testing.main()
