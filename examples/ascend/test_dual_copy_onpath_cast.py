"""NPU accuracy tests for L0C fp32 -> UB f16/bf16 dual_copy on-path cast.

Covers a compact-N column slice (64x32 GEMM in 64x64 L0C), a dynamic
over-half tail (192 in a 256 tile), plus a small full-tile smoke check.
"""

import pytest
import torch

import tilelang
from example_dual_copy_onpath_cast import gemm, ref_program, run_dynamic_tail, run_subregion

DST_CASES = (("float16", 2e-2), ("bfloat16", 5e-2))


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype,thresh", DST_CASES)
def test_onpath_compact_subregion(split, dst_dtype, thresh):
    """64x64 L0C alloc, compact gemm+dual_copy on acc[:, :32]."""
    run_subregion(dtype="float16", dst_dtype=dst_dtype, split=split, thresh=thresh)


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype,thresh", DST_CASES)
def test_onpath_dynamic_tail(split, dst_dtype, thresh):
    """256x256 GEMM; dual_copy a runtime 192 prefix into a half-tile UB."""
    run_dynamic_tail(dtype="float16", dst_dtype=dst_dtype, split=split, thresh=thresh)


@pytest.mark.parametrize("split", ["M", "N"])
def test_onpath_full_tile_small(split):
    """Single-block full-tile smoke check (not the 8192 regression size)."""
    m = n = k = 256
    dst_dtype, thresh = "float16", 2e-2
    device = torch.device("npu")
    kernel = tilelang.compile(
        gemm(m, k, n, dtype="float16", dst_dtype=dst_dtype, split=split),
        out_idx=-1,
    )
    x = torch.randn(m, k, dtype=torch.float16, device=device)
    w = torch.randn(n, k, dtype=torch.float16, device=device)
    c = kernel(x, w)
    torch.npu.synchronize()
    expected = ref_program(x, w, dst_dtype)
    rel = (c.float() - expected.float()).abs().max() / expected.float().abs().max()
    assert rel < thresh, f"rel_max={rel:.2e}"


if __name__ == "__main__":
    for dst_dtype, thresh in DST_CASES:
        for split in ["M", "N"]:
            test_onpath_compact_subregion(split, dst_dtype, thresh)
        for split in ["M", "N"]:
            test_onpath_dynamic_tail(split, dst_dtype, thresh)
    for split in ["M", "N"]:
        test_onpath_full_tile_small(split)
    print("PASS: onpath column-slice / dynamic-tail / small full-tile")
