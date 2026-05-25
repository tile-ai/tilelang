import pytest
import tvm

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_im2col_kernel(use_deprecated_alias=False):
    N, C, H, W, F, K = 1, 32, 8, 8, 32, 3
    S, D, P = 1, 1, 1
    block_M, block_N, block_K = 16, 32, 32
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    @T.prim_func
    def conv(
        data: T.Tensor((N, H, W, C), T.float16),
        weight: T.Tensor((KH, KW, C, F), T.float16),
        out: T.Tensor((N, OH, OW, F), T.float16),
    ):
        with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=128) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), T.float16)
            weight_shared = T.alloc_shared((block_K, block_N), T.float16)
            out_local = T.alloc_fragment((block_M, block_N), T.float32)
            out_shared = T.alloc_shared((block_M, block_N), T.float16)

            weight_flat = T.Tensor((KH * KW * C, F), T.float16, weight.data)
            out_flat = T.Tensor((N * OH * OW, F), T.float16, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=3):
                if use_deprecated_alias:
                    T.c2d_im2col(data, data_shared, by, k_iter, KH, S, D, P)
                else:
                    T.im2col(data, data_shared, by, k_iter, KH, S, D, P)
                T.copy(weight_flat[k_iter * block_K, bx * block_N], weight_shared)
                T.gemm(data_shared, weight_shared, out_local)

            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return conv


def _lower_to_cuda_source(func, arch):
    target = {"kind": "cuda", "arch": arch}
    with tvm.transform.PassContext(), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_im2col_uses_simt_fallback_before_hopper():
    src = _lower_to_cuda_source(_make_im2col_kernel(), "sm_80")
    assert "tma_load_im2col" not in src
    assert "pipeline_mbar_mem" not in src


@tilelang.testing.requires_cuda
def test_im2col_uses_tma_on_hopper():
    src = _lower_to_cuda_source(_make_im2col_kernel(), "sm_90")
    assert "tma_load_im2col" in src


@tilelang.testing.requires_cuda
def test_c2d_im2col_alias_warns_and_uses_new_tileop():
    with pytest.warns(DeprecationWarning, match="T.c2d_im2col is deprecated"):
        func = _make_im2col_kernel(use_deprecated_alias=True)
    src = _lower_to_cuda_source(func, "sm_80")
    assert "tma_load_im2col" not in src


if __name__ == "__main__":
    test_im2col_uses_simt_fallback_before_hopper()
    test_im2col_uses_tma_on_hopper()
    test_c2d_im2col_alias_warns_and_uses_new_tileop()
