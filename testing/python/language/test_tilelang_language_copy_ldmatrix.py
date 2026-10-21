import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_gemm_fragment_8x8, make_gemm_fragment_8x8_transposed, make_swizzled_layout


def _shared_region_copy(row, col, transposed, swizzled, fragment_rows=16):
    fragment = make_gemm_fragment_8x8_transposed() if transposed else make_gemm_fragment_8x8()
    fragment = fragment.repeat([fragment_rows // 8, 2], repeat_on_thread=False)

    @T.prim_func
    def main(A: T.Tensor((32, 32), T.float16), C: T.Tensor((16, 16), T.float16)):
        with T.Kernel(1, threads=32):
            As = T.alloc_shared((32, 32), T.float16)
            Af = T.alloc_fragment((fragment_rows, 16), T.float16)
            T.annotate_layout({Af: fragment})
            if swizzled:
                T.annotate_layout({As: make_swizzled_layout(As)})
            T.copy(A, As)
            T.copy(As[row : row + 16, col : col + 16], Af[:16, :])
            T.copy(Af[:16, :], C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 5)
@pytest.mark.parametrize("row, col", [(0, 0), (16, 0), (0, 16), (16, 16), (0, 1)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("swizzled", [False, True])
def test_ldmatrix_shared_region(row, col, transposed, swizzled):
    kernel = tilelang.compile(_shared_region_copy(row, col, transposed, swizzled), out_idx=[1])
    # An unaligned column origin must keep using the ordinary copy fallback.
    source = kernel.get_kernel_source()
    assert ("tl::ptx_ldmatrix" in source) == (col % 8 == 0)
    if col % 8 == 0:
        assert ("tl::ptx_ldmatrix_x4_trans" in source) == transposed
    a = torch.arange(32 * 32, device="cuda", dtype=torch.float16).reshape(32, 32)
    torch.testing.assert_close(kernel(a), a[row : row + 16, col : col + 16], rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 5)
def test_ldmatrix_partial_fragment_falls_back():
    kernel = tilelang.compile(_shared_region_copy(16, 16, False, False, fragment_rows=32), out_idx=[1])
    assert "tl::ptx_ldmatrix" not in kernel.get_kernel_source()
    a = torch.arange(32 * 32, device="cuda", dtype=torch.float16).reshape(32, 32)
    torch.testing.assert_close(kernel(a), a[16:32, 16:32], rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 5)
@pytest.mark.parametrize("row, col", [(0, 0), (16, 0), (0, 16), (16, 16)])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_ldmatrix_shared_region_feeds_gemm(row, col, transpose_a):
    @T.prim_func
    def main(A: T.Tensor((32, 32), T.float16), B: T.Tensor((16, 16), T.float16), C: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            As = T.alloc_shared((32, 32), T.float16)
            Bs = T.alloc_shared((16, 16), T.float16)
            Af = T.alloc_fragment((16, 16), T.float16)
            Cf = T.alloc_fragment((16, 16), T.float32)
            T.copy(A, As)
            T.copy(B, Bs)
            T.copy(As[row : row + 16, col : col + 16], Af)
            T.clear(Cf)
            T.gemm(Af, Bs, Cf, transpose_A=transpose_a)
            T.copy(Cf, C)

    kernel = tilelang.compile(main, out_idx=[2])
    a = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    region = a[row : row + 16, col : col + 16].float()
    if transpose_a:
        region = region.T
    torch.testing.assert_close(kernel(a, b), region @ b.float(), rtol=1e-4, atol=1e-4)


def _shared_region_store(row, col, transposed, swizzled, fragment_row=0, copy_rows=16, dtype=T.float16, lower_dim_first=True):
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
            T.copy(Af[fragment_row : fragment_row + copy_rows, :], As[row : row + copy_rows, col : col + 16])
            T.copy(As, C)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize(
    "row, col, transposed, swizzled",
    [
        (0, 0, False, False),
        (16, 0, False, False),
        (0, 16, True, False),
        (16, 16, True, True),
        (1, 8, False, True),
        (0, 1, False, False),
    ],
)
def test_stmatrix_shared_region(row, col, transposed, swizzled):
    kernel = tilelang.compile(_shared_region_store(row, col, transposed, swizzled), out_idx=[1])
    source = kernel.get_kernel_source()
    assert ("tl::ptx_stmatrix" in source) == (col % 8 == 0)
    if col % 8 == 0:
        assert ("tl::ptx_stmatrix_m8n8_x4_trans" in source) == transposed
    a = torch.arange(16 * 16, device="cuda", dtype=torch.float16).reshape(16, 16)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.float16)
    expected[row : row + 16, col : col + 16] = a
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("fragment_row", [0, 8])
def test_stmatrix_partial_fragment_falls_back(fragment_row):
    kernel = tilelang.compile(_shared_region_store(16, 16, False, False, fragment_row=fragment_row, copy_rows=8), out_idx=[1])
    assert "tl::ptx_stmatrix" not in kernel.get_kernel_source()
    a = torch.arange(16 * 16, device="cuda", dtype=torch.float16).reshape(16, 16)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.float16)
    expected[16:24, 16:32] = a[fragment_row : fragment_row + 8, :]
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("col, lower_dim_first", [(16, True), (8, True), (16, False)])
def test_stmatrix_int8_shared_region(col, lower_dim_first):
    kernel = tilelang.compile(_shared_region_store(16, col, True, True, dtype=T.int8, lower_dim_first=lower_dim_first), out_idx=[1])
    # The 8-bit variant needs 16-byte row alignment and a matching register order.
    assert ("tl::ptx_stmatrix_m16n8_x2_trans" in kernel.get_kernel_source()) == (col % 16 == 0 and lower_dim_first)
    a = torch.arange(-128, 128, device="cuda", dtype=torch.int32).to(torch.int8).reshape(16, 16)
    expected = torch.full((32, 32), -1, device="cuda", dtype=torch.int8)
    expected[16:32, col : col + 16] = a
    torch.testing.assert_close(kernel(a), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
