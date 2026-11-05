"""Public strided copy kernels keep their parameter metadata after address lowering."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _copy_kernel(dynamic_shape):
    rows = T.dynamic("rows") if dynamic_shape else 1
    stride = T.dynamic("stride", "int64")

    @T.prim_func
    def main(
        src: T.StridedTensor((rows, 32), (stride, 1), "float32"),
        dst: T.StridedTensor((rows, 32), (stride, 1), "float32"),
    ):
        T.assume(stride % 2 == 0)
        with T.Kernel(1, threads=128):
            for row, column in T.Parallel(rows, 32):
                dst[row, column] = src[row, column] + 3

    return main


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
@pytest.mark.parametrize("dynamic_shape", [False, True])
@tilelang.testing.requires_cuda
def test_public_copy_keeps_dynamic_strides_and_padding(execution_backend, dynamic_shape):
    kernel = tilelang.compile(_copy_kernel(dynamic_shape), target="cuda", execution_backend=execution_backend)
    for rows in [1, 3] if dynamic_shape else [1]:
        src_storage = torch.full((rows * 40,), -313, dtype=torch.float32, device="cuda")
        dst_storage = torch.full((rows * 40,), -991, dtype=torch.float32, device="cuda")
        src = torch.as_strided(src_storage, (rows, 32), (40, 1))
        dst = torch.as_strided(dst_storage, (rows, 32), (40, 1))
        src.copy_(torch.arange(rows * 32, dtype=torch.float32, device="cuda").reshape(rows, 32))
        kernel(src, dst)
        torch.cuda.synchronize()
        torch.testing.assert_close(dst, src + 3, rtol=0, atol=0)
        padding = dst_storage.reshape(rows, 40)[:, 32:]
        torch.testing.assert_close(padding, torch.full_like(padding, -991), rtol=0, atol=0)
