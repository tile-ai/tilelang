"""T.gemm must reject non-zero K origins on L1 operand regions."""

import pytest

import tilelang.ascend.language as T


def test_gemm_rejects_nonzero_k_origin():
    with pytest.raises(ValueError, match="non-zero K origin"):

        @T.prim_func
        def kernel(
            X: T.Tensor((256, 512), "bfloat16"),
            W: T.Tensor((256, 512), "bfloat16"),
            C: T.Tensor((256, 256), "float32"),
        ):
            with T.Kernel(1):
                res = T.alloc_l0c((256, 256), "float32")
                x_l1 = T.alloc_l1((256, 512), "bfloat16")
                w_l1 = T.alloc_l1((256, 512), "bfloat16")
                T.copy(X, x_l1)
                T.copy(W, w_l1)
                T.gemm(x_l1[:, 256:512], w_l1[:, 256:512], res, transpose_B=True, clear_accum=True)
                T.copy(res, C)


def test_gemm_rejects_nonzero_k_origin_with_transpose_a():
    with pytest.raises(ValueError, match="non-zero K origin"):

        @T.prim_func
        def kernel(
            X: T.Tensor((512, 256), "bfloat16"),
            W: T.Tensor((256, 512), "bfloat16"),
            C: T.Tensor((256, 256), "float32"),
        ):
            with T.Kernel(1):
                res = T.alloc_l0c((256, 256), "float32")
                x_l1 = T.alloc_l1((512, 256), "bfloat16")
                w_l1 = T.alloc_l1((256, 512), "bfloat16")
                T.copy(X, x_l1)
                T.copy(W, w_l1)
                T.gemm(x_l1[256:512, :], w_l1, res, transpose_A=True, transpose_B=True, clear_accum=True)
                T.copy(res, C)
