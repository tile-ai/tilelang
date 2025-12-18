import tilelang.testing
from tilelang import language as T


def alloc_var(
    N,
    block_N,
    dtype,
):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_var(dtype)
            tmp = 1  # noqa: F841
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_var(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = alloc_var(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source()
    assert "tmp =" in code


def test_alloc_var():
    run_alloc_var(1024, 128, T.float16)


def alloc_var_add(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            tmp = T.alloc_var(dtype)
            tmp = 1  # noqa: F841
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = A_shared[i] + tmp
            T.copy(A_shared, B[bx * block_N])

    return main


def run_alloc_var_add(
    N,
    block_N,
    dtype,
):
    program = alloc_var_add(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source()
    assert "tmp =" in code


def test_alloc_var_add():
    run_alloc_var_add(1024, 128, T.float16)


def alloc_var_with_initializer(
    N,
    block_N,
    dtype,
    init_value,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            tmp = T.alloc_var(dtype, init_value)
            T.copy(A[bx * block_N], B[bx * block_N])
            for i in T.Parallel(block_N):
                B[bx * block_N + i] = tmp

    return main


def run_alloc_var_with_initializer(
    N,
    block_N,
    dtype,
    init_value,
):
    program = alloc_var_with_initializer(N, block_N, dtype, init_value)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source()
    assert f"= {init_value};" in code


def test_alloc_var_with_initializer():
    run_alloc_var_with_initializer(256, 64, T.int32, 5)


def alloc_multi_vars_with_initializer(
    N,
    block_N,
    dtype,
):
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            tmp0 = T.alloc_var(dtype, 1)
            tmp1 = T.alloc_var(dtype, 2)
            T.copy(A[bx * block_N], B[bx * block_N])
            for i in T.Parallel(block_N):
                B[bx * block_N + i] = tmp0 + tmp1

    return main


def run_alloc_multi_vars_with_initializer(
    N,
    block_N,
    dtype,
):
    program = alloc_multi_vars_with_initializer(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    code = kernel.get_kernel_source(kernel_only=True)
    assert code.count("= 1;") == 1
    assert code.count("= 2;") == 1


def test_alloc_multi_vars_with_initializer():
    run_alloc_multi_vars_with_initializer(256, 64, T.int32)


if __name__ == "__main__":
    tilelang.testing.main()
