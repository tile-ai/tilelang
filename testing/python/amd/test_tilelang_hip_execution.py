"""Numerical build/load/launch coverage for both HIP execution adapters."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.rocm import language as T


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_hip_vector_add_on_torch_stream(execution_backend, dtype):
    @T.prim_func
    def add_one(A: T.Tensor((1024,), dtype), B: T.Tensor((1024,), dtype)):
        with T.Kernel(8, threads=128) as block:
            for i in T.Parallel(128):
                B[block * 128 + i] = A[block * 128 + i] + 1.0

    kernel = tilelang.compile(add_one, target="hip", target_host="c", execution_backend=execution_backend, out_idx=[1])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        data = (torch.arange(1024, device="cuda") % 16).to(getattr(torch, dtype))
        result = kernel(data)
        expected = data + 1
    stream.synchronize()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython"])
def test_hip_shared_memory_reduction(execution_backend):
    @T.prim_func
    def row_sum(A: T.Tensor((4, 256), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(4, threads=128) as row:
            values = T.alloc_fragment((256,), "float32")
            total = T.alloc_fragment((1,), "float32")
            T.copy(A[row, :], values)
            T.reduce_sum(values, total, dim=0)
            B[row] = total[0]

    kernel = tilelang.compile(row_sum, target="hip", target_host="c", execution_backend=execution_backend, out_idx=[1])
    data = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(4, 256)
    result = kernel(data)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, data.sum(dim=1), rtol=0, atol=0)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython"])
def test_hip_tensor_gemm(execution_backend):
    @T.prim_func
    def matmul(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            left = T.alloc_shared((64, 32), "float16")
            right = T.alloc_shared((32, 64), "float16")
            accum = T.alloc_fragment((64, 64), "float32")
            T.copy(A, left)
            T.copy(B, right)
            T.clear(accum)
            T.gemm(left, right, accum)
            T.copy(accum, C)

    kernel = tilelang.compile(matmul, target="hip", target_host="c", execution_backend=execution_backend, out_idx=[2])
    left = torch.randn(64, 32, dtype=torch.float16)
    right = torch.randn(32, 64, dtype=torch.float16)
    result = kernel(left.cuda(), right.cuda())
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), left.float() @ right.float(), rtol=1e-4, atol=1e-4)
