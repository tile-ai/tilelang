import tilelang.testing
from tilelang import language as T
import torch


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
    assert "tmp =" in code or "tmp[0] =" in code


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
    assert "tmp =" in code or "tmp[0] =" in code


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


# TODO(Gong): ROCm is not supported yet, disable for now
@tilelang.testing.requires_cuda
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


# TODO(Gong): ROCm is not supported yet, disable for now
@tilelang.testing.requires_cuda
def test_alloc_multi_vars_with_initializer():
    run_alloc_multi_vars_with_initializer(256, 64, T.int32)


def alloc_global(
    N,
    block_N,
    dtype,
):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        C = T.alloc_global((N,), dtype)
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            T.copy(A[bx * block_N : (bx + 1) * block_N], C[bx * block_N : (bx + 1) * block_N])
            T.copy(C[bx * block_N : (bx + 1) * block_N], B[bx * block_N : (bx + 1) * block_N])

    return main


def run_alloc_global(
    N,
    block_N,
    dtype,
):
    program = alloc_global(N, block_N, dtype)

    kernel = tilelang.compile(program, out_idx=[1])
    # print(kernel.get_host_source())
    # code = kernel.get_kernel_source()
    # print(code)
    A = torch.randn(N, device="cuda", dtype=getattr(torch, dtype))
    B = kernel(A)
    torch.testing.assert_close(B, A, rtol=1e-2, atol=1e-2)


@tilelang.jit
def alloc_global_eagerjit(A, block_N, dtype):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    B = T.empty(
        [
            N,
        ],
        dtype=dtype,
    )
    C = T.alloc_global(
        [
            N,
        ],
        dtype,
    )

    with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
        T.copy(A[bx * block_N : (bx + 1) * block_N], C[bx * block_N : (bx + 1) * block_N])
        T.copy(C[bx * block_N : (bx + 1) * block_N], B[bx * block_N : (bx + 1) * block_N])

    return B


def run_alloc_global_eagerjit(
    N,
    block_N,
    dtype,
):
    A = torch.randn(N, device="cuda", dtype=getattr(torch, dtype))
    B = alloc_global_eagerjit(A, block_N, dtype)
    torch.testing.assert_close(B, A, rtol=1e-2, atol=1e-2)


def test_alloc_global():
    run_alloc_global(1024, 128, T.float16)
    run_alloc_global_eagerjit(1024, 128, T.float16)


if __name__ == "__main__":
    tilelang.testing.main()
