"""A partial transposed region must read only its logical matrix elements."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


def _l0_explicit_transpose_kernel(M, K, N, dtype="bfloat16", mn_region=None, k_region=None, dst_k=None):
    """Explicit L1->L0 transpose, optionally restricting the source region."""
    tile_m = M if mn_region is None else mn_region
    tile_k = K if dst_k is None else dst_k
    copy_k = K if k_region is None else k_region

    @T.prim_func
    def main(
        A: T.Tensor((K, M), dtype),
        B: T.Tensor((N, tile_k), dtype),
        C: T.Tensor((tile_m, N), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((K, M), dtype)
            b_l1 = T.alloc_l1((N, tile_k), dtype)
            a_l0 = T.alloc_l0a((tile_m, tile_k), dtype)
            b_l0 = T.alloc_l0b((N, tile_k), dtype)
            acc = T.alloc_l0c((tile_m, N), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1[:copy_k, :tile_m], a_l0, transpose=True)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return main


@pytest.mark.parametrize("dtype,mn_region", [("bfloat16", 24), ("float32", 25)])
def test_transpose_partial_fractal_correctness(dtype, mn_region):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    kernel = tilelang.compile(_l0_explicit_transpose_kernel(32, 32, 64, dtype, mn_region=mn_region), out_idx=-1, target="ascend")
    generator = torch.Generator().manual_seed(0)
    a = torch.randint(-4, 5, (32, 32), generator=generator).to(getattr(torch, dtype)) / 8
    b = torch.randint(-4, 5, (64, 32), generator=generator).to(getattr(torch, dtype)) / 8
    expected = a[:, :mn_region].float().T @ b.float().T
    actual = kernel(a.npu(), b.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
