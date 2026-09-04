import pytest
import tvm

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_im2col_kernel(use_deprecated_alias=False, channels=32, block_K=32, block_M=16, hw=8):
    N, C, H, W, F, K = 1, channels, hw, hw, 32, 3
    S, D, P = 1, 1, 1
    block_N = 32
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
def test_im2col_wide_channel_block_uses_multiple_boxes():
    """128 fp16 channels exceed the 64-element full-bank box, so the copy
    issues two tma_load_im2col instructions stepping the channel coordinate."""
    import re

    src = _lower_to_cuda_source(_make_im2col_kernel(channels=128, block_K=128), "sm_90")
    assert re.search(r"for \(int \w+ = 0; \w+ < 2; \+\+\w+\) \{\n\s*tl::tma_load_im2col\(", src)


@tilelang.testing.requires_cuda
def test_im2col_tall_pixel_block_uses_single_box():
    """An im2col TensorMap accesses up to 1024 pixels per column (unlike the
    256-element tiled box cap), so a 320-pixel block is one instruction."""
    import re

    src = _lower_to_cuda_source(_make_im2col_kernel(block_M=320, hw=20), "sm_90")
    assert "tma_load_im2col" in src
    assert not re.search(r"for [^\n]*\{\n\s*tl::tma_load_im2col\(", src)


@tilelang.testing.requires_cuda
def test_im2col_single_channel_rejects_pixel_inner_pairing():
    """With channels == 1 the pixel mode can pair as TMA mode 0; the channel
    dim then sits at a one-element global stride, which the non-innermost
    16-byte stride rule rejects before any descriptor is built — never a
    silently swapped channel/pixel encoding."""
    N, C, H, W, K = 1, 1, 8, 8, 3
    block_M, block_K = 16, 1

    @T.prim_func
    def kern(
        data: T.Tensor((N, H, W, C), T.float16),
        out: T.Tensor((block_M, block_K), T.float16),
    ):
        with T.Kernel(1, threads=128):
            sh = T.alloc_shared((block_M, block_K), T.float16)
            T.im2col(data, sh, 0, 0, K, 1, 1, 1)
            T.copy(sh, out)

    with pytest.raises(Exception, match="im2col cannot lower.*16-byte multiple"):
        _lower_to_cuda_source(kern, "sm_90")


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
