import tilelang
import tilelang.language as T
import tilelang.testing


def test_alloc_var_preserves_float64_initializer_dtype() -> None:
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float64)):
        with T.Kernel(1):
            value = T.alloc_var(T.float64, init=1e300)
            A[0] = value

    initializers = []

    def collect_float64_initializer(node):
        if isinstance(node, tilelang.tvm.tirx.FloatImm) and node.dtype == T.float64:
            initializers.append(node.value)

    tilelang.tvm.tirx.stmt_functor.post_order_visit(kernel.body, collect_float64_initializer)
    assert 1e300 in initializers


# TODO: var init is not supported on hip.
@tilelang.testing.requires_cuda
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
