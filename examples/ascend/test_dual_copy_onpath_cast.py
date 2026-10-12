"""NPU accuracy tests for L0C fp32 -> UB f16/bf16 dual_copy on-path cast.

Covers a compact-N column slice (64x32 GEMM in 64x64 L0C), a dynamic
over-half tail (192 in a 256 tile), plus a small full-tile smoke check.
"""

import pytest
import torch

import tilelang
from example_dual_copy_onpath_cast import (
    gemm,
    ref_program,
    run_compact_full_n_copy_and_dual,
    run_compact_n_split,
    run_destination_window,
    run_dynamic_tail,
    run_region_prefix,
    run_subregion,
    run_unit_flag_reuse,
)

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


def test_onpath_compact_mad_copy_and_dual():
    """Compact 32x64 MAD in a 64x64 L0C: plain fp32 copy + fp32/fp16 dual_copy."""
    run_compact_full_n_copy_and_dual()


@pytest.mark.parametrize("dst_dtype,thresh", DST_CASES)
def test_onpath_compact_n_split(dst_dtype, thresh):
    """Compact 32x64 MAD N-split into two 32x32 UB halves."""
    run_compact_n_split(dst_dtype, thresh)


def test_onpath_unit_flag_reuse():
    """Two GEMMs reuse one L0C with unit_flag_ctrl=3, then dual_copy."""
    run_unit_flag_reuse()


@pytest.mark.parametrize("row_start,row_end", [(0, 32), (16, 48)])
def test_onpath_destination_window(row_start, row_end):
    """dual_copy into a window smaller than the UB allocation."""
    run_destination_window(row_start, row_end)


def test_onpath_region_prefix():
    """128x128 prefix of a 256x256 GEMM keeps the producer M pitch."""
    run_region_prefix()


if __name__ == "__main__":
    for dst_dtype, thresh in DST_CASES:
        for split in ["M", "N"]:
            test_onpath_compact_subregion(split, dst_dtype, thresh)
        for split in ["M", "N"]:
            test_onpath_dynamic_tail(split, dst_dtype, thresh)
    for split in ["M", "N"]:
        test_onpath_full_tile_small(split)
    test_onpath_compact_mad_copy_and_dual()
    for dst_dtype, thresh in DST_CASES:
        test_onpath_compact_n_split(dst_dtype, thresh)
    test_onpath_unit_flag_reuse()
    for row_start, row_end in [(0, 32), (16, 48)]:
        test_onpath_destination_window(row_start, row_end)
    test_onpath_region_prefix()
    print("PASS: onpath column-slice / dynamic-tail / small full-tile / review cases")
