"""L0C accumulator dtype validation."""

import pytest

import tilelang.ascend.language as T


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_alloc_l0c_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="float32 or int32"):

        @T.prim_func
        def kernel(
            X: T.Tensor((256, 256), "bfloat16"),
            W: T.Tensor((256, 256), "bfloat16"),
            C: T.Tensor((256, 256), dtype),
        ):
            with T.Kernel(1):
                res = T.alloc_l0c((256, 256), dtype)
                x_l1 = T.alloc_l1((256, 256), "bfloat16")
                w_l1 = T.alloc_l1((256, 256), "bfloat16")
                T.copy(X, x_l1)
                T.copy(W, w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=True)
                T.copy(res, C)
