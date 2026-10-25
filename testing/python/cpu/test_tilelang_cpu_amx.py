"""Intel AMX selection, code generation, and hardware correctness tests."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
import tilelang.testing
from tilelang.cpu import amx_target, host_has_amx


def _matmul(M, N, K, a_dtype, accum_dtype, trans_A=False, trans_B=False, b_dtype=None):
    b_dtype = a_dtype if b_dtype is None else b_dtype
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, a_dtype),
        B: T.Tensor(b_shape, b_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_local(a_shape, a_dtype)
            B_local = T.alloc_local(b_shape, b_dtype)
            C_local = T.alloc_local((M, N), accum_dtype)
            T.copy(A, A_local)
            T.copy(B, B_local)
            T.clear(C_local)
            T.gemm(A_local, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C)

    return main


def _compile(func, *, parallel=False):
    target = amx_target()
    return tilelang.compile(
        func,
        out_idx=[2],
        target=target,
        target_host=target,
        execution_backend="cython",
        pass_configs={"tl.cpu_parallel": parallel},
    )


def _full_range_int8(shape, dtype):
    info = torch.iinfo(dtype)
    values = torch.randint(info.min, info.max + 1, shape, dtype=dtype)
    values.view(-1)[:2] = torch.tensor((info.min, info.max), dtype=dtype)
    return values


@T.prim_func
def _blocked_bf16_matmul(
    A: T.Tensor((16, 64), T.bfloat16),
    B: T.Tensor((64, 16), T.bfloat16),
    C: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1) as _:
        A_local = T.alloc_local((16, 32), T.bfloat16)
        B_local = T.alloc_local((32, 16), T.bfloat16)
        C_local = T.alloc_local((16, 16), T.float32)
        T.clear(C_local)
        for ko in T.serial(2):
            T.copy(A[0, ko * 32], A_local)
            T.copy(B[ko * 32, 0], B_local)
            T.gemm(A_local, B_local, C_local)
        T.copy(C_local, C)


@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_codegen_and_tail_correctness():
    kernel = _compile(_matmul(17, 19, 67, T.bfloat16, T.float32))
    assert "tl_amx_bf16_gemm" in kernel.get_kernel_source()
    a = torch.randn((17, 67), dtype=torch.bfloat16)
    b = torch.randn((67, 19), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = a.float() @ b.float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_parallel_codegen_and_correctness():
    kernel = _compile(_matmul(64, 128, 96, T.bfloat16, T.float32), parallel=True)
    assert "tl_amx_bf16_gemm_parallel" in kernel.get_kernel_source()
    a = torch.randn((64, 96), dtype=torch.bfloat16)
    b = torch.randn((96, 128), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = a.float() @ b.float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_accumulates_across_k_blocks():
    kernel = _compile(_blocked_bf16_matmul)
    a = torch.randn((16, 64), dtype=torch.bfloat16)
    b = torch.randn((64, 16), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = a.float() @ b.float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("trans_A,trans_B", [(True, False), (False, True), (True, True)])
@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_transpose(trans_A, trans_B):
    kernel = _compile(_matmul(16, 16, 32, T.bfloat16, T.float32, trans_A, trans_B))
    a = torch.randn((32, 16) if trans_A else (16, 32), dtype=torch.bfloat16)
    b = torch.randn((16, 32) if trans_B else (32, 16), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = (a.T if trans_A else a).float() @ (b.T if trans_B else b).float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (T.int8, T.int8),
        (T.uint8, T.uint8),
        (T.int8, T.uint8),
        (T.uint8, T.int8),
    ],
)
@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_int8_codegen_and_tail_correctness(a_dtype, b_dtype):
    kernel = _compile(_matmul(17, 19, 67, a_dtype, T.int32, b_dtype=b_dtype))
    assert "tl_amx_int8_gemm" in kernel.get_kernel_source()
    a_torch_dtype = torch.int8 if a_dtype == T.int8 else torch.uint8
    b_torch_dtype = torch.int8 if b_dtype == T.int8 else torch.uint8
    a = _full_range_int8((17, 67), a_torch_dtype)
    b = _full_range_int8((67, 19), b_torch_dtype)
    actual = kernel(a, b)
    expected = a.to(torch.int32) @ b.to(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (T.int8, T.int8),
        (T.uint8, T.uint8),
        (T.int8, T.uint8),
        (T.uint8, T.int8),
    ],
)
@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_int8_parallel_codegen_and_correctness(a_dtype, b_dtype):
    kernel = _compile(_matmul(64, 128, 128, a_dtype, T.int32, b_dtype=b_dtype), parallel=True)
    assert "_parallel" in kernel.get_kernel_source()
    a_torch_dtype = torch.int8 if a_dtype == T.int8 else torch.uint8
    b_torch_dtype = torch.int8 if b_dtype == T.int8 else torch.uint8
    a = _full_range_int8((64, 128), a_torch_dtype)
    b = _full_range_int8((128, 128), b_torch_dtype)
    actual = kernel(a, b)
    expected = a.to(torch.int32) @ b.to(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_sliced_regions_use_physical_strides():
    @T.prim_func
    def main(
        A: T.Tensor((17, 67), T.bfloat16),
        B: T.Tensor((67, 19), T.bfloat16),
        C: T.Tensor((17, 19), T.float32),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_local((19, 71), T.bfloat16)
            B_local = T.alloc_local((69, 23), T.bfloat16)
            C_local = T.alloc_local((19, 23), T.float32)
            T.copy(A, A_local[1:18, 2:69])
            T.copy(B, B_local[1:68, 2:21])
            T.clear(C_local)
            T.gemm(A_local[1:18, 2:69], B_local[1:68, 2:21], C_local[1:18, 2:21])
            T.copy(C_local[1:18, 2:21], C)

    kernel = _compile(main)
    a = torch.randn((17, 67), dtype=torch.bfloat16)
    b = torch.randn((67, 19), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = a.float() @ b.float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not host_has_amx(), reason="requires Intel AMX BF16 and INT8")
def test_amx_bf16_rank3_singleton_regions():
    @T.prim_func
    def main(
        A: T.Tensor((17, 67), T.bfloat16),
        B: T.Tensor((67, 19), T.bfloat16),
        C: T.Tensor((17, 19), T.float32),
    ):
        with T.Kernel(1) as _:
            A_local = T.alloc_local((1, 19, 71), T.bfloat16)
            B_local = T.alloc_local((1, 69, 23), T.bfloat16)
            C_local = T.alloc_local((1, 19, 23), T.float32)
            T.copy(A, A_local[0, 1:18, 2:69])
            T.copy(B, B_local[0, 1:68, 2:21])
            T.clear(C_local)
            T.gemm(A_local[0:1, 1:18, 2:69], B_local[0:1, 1:68, 2:21], C_local[0:1, 1:18, 2:21])
            T.copy(C_local[0, 1:18, 2:21], C)

    kernel = _compile(main)
    a = torch.randn((17, 67), dtype=torch.bfloat16)
    b = torch.randn((67, 19), dtype=torch.bfloat16)
    actual = kernel(a, b)
    expected = a.float() @ b.float()
    tilelang.testing.torch_assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not host_has_amx(), reason="requires an AMX-capable compiler and host")
def test_amx_target_falls_back_for_float32_gemm():
    kernel = _compile(_matmul(8, 8, 8, T.float32, T.float32))
    assert "tl_amx_bf16_gemm(" not in kernel.get_kernel_source()
    assert "tl_amx_int8_gemm_" not in kernel.get_kernel_source()
    a = torch.randn((8, 8), dtype=torch.float32)
    b = torch.randn((8, 8), dtype=torch.float32)
    tilelang.testing.torch_assert_close(kernel(a, b), a @ b, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
