"""CuTeDSL fragment-to-shared copies through stmatrix, mirroring the CUDA
coverage in test_tilelang_language_copy_ldmatrix.py."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_gemm_fragment_8x8, make_gemm_fragment_8x8_transposed, make_swizzled_layout


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _shared_region_store(row, col, transposed, swizzled, dtype, lower_dim_first=True):
    fragment = make_gemm_fragment_8x8_transposed() if transposed else make_gemm_fragment_8x8()
    fragment = fragment.repeat([2, 2], repeat_on_thread=False, lower_dim_first=lower_dim_first)

    @T.prim_func
    def main(A: T.Tensor((16, 16), dtype), C: T.Tensor((32, 32), dtype)):
        with T.Kernel(1, threads=32):
            Af = T.alloc_fragment((16, 16), dtype)
            As = T.alloc_shared((32, 32), dtype)
            T.annotate_layout({Af: fragment})
            if swizzled:
                T.annotate_layout({As: make_swizzled_layout(As)})
            T.copy(A, Af)
            T.fill(As, -1)
            T.copy(Af, As[row : row + 16, col : col + 16])
            T.copy(As, C)

    return main


def _compile(func):
    _require_cutedsl()
    return tilelang.compile(func, out_idx=[1], target="cutedsl")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("row, col, transposed", [(0, 0, False), (16, 16, True), (0, 1, False)])
def test_stmatrix_16bit(row, col, transposed):
    kernel = _compile(_shared_region_store(row, col, transposed, True, T.float16))
    source = kernel.get_kernel_source()
    assert ("tl.ptx_stmatrix_x4" in source) == (col % 8 == 0)
    a = torch.arange(16 * 16, device="cuda", dtype=torch.float16).reshape(16, 16)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.float16)
    expected[row : row + 16, col : col + 16] = a
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("col, lower_dim_first", [(16, True), (8, True), (16, False)])
def test_stmatrix_int8(col, lower_dim_first):
    kernel = _compile(_shared_region_store(16, col, True, True, T.int8, lower_dim_first=lower_dim_first))
    # The 8-bit variant needs 16-byte row alignment and a matching register order.
    assert ("tl.ptx_stmatrix_m16n8_x2_trans" in kernel.get_kernel_source()) == (col % 16 == 0 and lower_dim_first)
    a = torch.arange(-128, 128, device="cuda", dtype=torch.int32).to(torch.int8).reshape(16, 16)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.int8)
    expected[16:32, col : col + 16] = a
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_stmatrix_float8():
    kernel = _compile(_shared_region_store(16, 16, True, True, T.float8_e4m3fn))
    source = kernel.get_kernel_source()
    assert "tl.ptx_stmatrix_m16n8_x2_trans" in source
    assert "tl.pack_b8x4(" in source
    bits = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    bits[bits & 0x7F == 0x7F] = 0
    a = bits.view(torch.float8_e4m3fn).reshape(16, 16)
    out = kernel(a)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    expected[16:32, 16:32] = a
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


if __name__ == "__main__":
    tilelang.testing.main()
