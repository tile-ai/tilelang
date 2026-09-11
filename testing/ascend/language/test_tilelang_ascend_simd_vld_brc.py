"""A ``BRC_*`` dist broadcasts one element of the width its suffix names.

``vld`` derives the vector type -- and therefore the emitted C API and its load
granule -- from the source buffer's element type. A broadcast naming a wider
element must widen the result instead of silently degrading to the buffer width,
which previously dropped the upper byte of a packed pair.
"""

import pytest
import torch

import tilelang
import tilelang.language as T


def broadcast_kernel(src_dtype, src_len, dist):
    """Reinterpret a broadcast load into int32 lanes so its granule is visible."""

    @T.prim_func
    def kernel(
        Src: T.Tensor((src_len,), src_dtype),
        Out: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1):
            src_ub = T.alloc_shared((src_len,), src_dtype)
            out_ub = T.alloc_shared((64,), "int32")
            T.copy(Src, src_ub)
            with T.SimdVF():
                full = T.simd.pset(32)
                broadcast = T.simd.vld(src_ub[0], dist=dist)
                lane = T.simd.alloc_local((1,), "int32")
                lane[0] = T.reinterpret(broadcast, "int32x64")
                T.simd.vsts(out_ub[0], lane[0], full)
            T.copy(out_ub, Out)

    return kernel


def test_ascend_simd_vld_brc_widens_past_buffer_width():
    """BRC_B16 over a uint8 buffer reads a 16-bit element, not a single byte."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    kernel = tilelang.compile(broadcast_kernel("uint8", 32, "BRC_B16"), target="ascend")
    src = torch.tensor([0x11, 0x22] * 16, dtype=torch.uint8)
    out = torch.empty(64, dtype=torch.int32, device="npu")
    kernel(src.to("npu"), out)

    # Each 32-bit lane repeats the 16-bit element; the degraded byte-granule
    # load produced the low byte alone.
    assert set(out.cpu().tolist()) == {0x22112211}


def test_ascend_simd_vld_brc_rejects_width_narrower_than_element():
    """A byte broadcast out of a 16-bit buffer has no meaning."""
    with pytest.raises(ValueError, match="does not fit"):
        broadcast_kernel("uint16", 16, "BRC_B8")


def test_ascend_simd_vld_brc_rejects_unknown_width():
    with pytest.raises(ValueError, match="must be one of"):
        broadcast_kernel("uint8", 32, "BRC_B64")
