"""Frontend validation for the sparse-GEMM E-metadata layout helpers.

`make_cutlass_metadata_layout` dispatches by architecture. Both arms pair the
MMA operand dtype with the metadata buffer width, and both must reject an
operand dtype they do not support rather than falling through to the
width-derived placement.
"""

import pytest
import torch
import tilelang
import tilelang.testing
from torch.sparse._semi_structured_conversions import _calculate_meta_reordering_scatter_offsets
from tvm import tirx

from tilelang.layout import make_cutlass_metadata_layout


def _metadata_buffer(dtype, shape=(64, 8)):
    return tirx.decl_buffer(shape, dtype)


@pytest.mark.parametrize("meta_dtype", ["uint16", "int16"])
@pytest.mark.parametrize("mma_dtype", ["float16", "bfloat16", "float32"])
def test_sm8x_metadata_layout_accepts_16_bit_metadata(mma_dtype, meta_dtype):
    """fp16/bf16/fp32 (TF32) use CUTLASS's 16-bit metadata layout."""
    m, k = 64, 8
    layout = make_cutlass_metadata_layout(_metadata_buffer(meta_dtype, (m, k)), mma_dtype, arch="8.0")
    expected = _calculate_meta_reordering_scatter_offsets(m, k, torch.int16, "cpu").tolist()
    actual = []
    for i in range(m):
        for j in range(k):
            row, col = layout.map_forward_index([i, j])
            actual.append(int(row) * k + int(col))
    assert actual == expected


@pytest.mark.parametrize("meta_dtype", ["uint32", "int32"])
@pytest.mark.parametrize("mma_dtype", ["float8_e4m3", "float8_e5m2", "int8", "uint8"])
def test_sm8x_metadata_layout_accepts_32_bit_operands(mma_dtype, meta_dtype):
    """fp8/int8 operands pair with a 32-bit metadata buffer."""
    layout = make_cutlass_metadata_layout(_metadata_buffer(meta_dtype), mma_dtype, arch="8.0")
    assert layout is not None


@pytest.mark.parametrize("mma_dtype,meta_dtype", [("float16", "uint32"), ("float32", "uint32"), ("float8_e4m3", "uint16")])
def test_sm8x_metadata_layout_rejects_mismatched_metadata_width(mma_dtype, meta_dtype):
    """A supported operand paired with the wrong metadata width is rejected."""
    with pytest.raises(ValueError) as exc_info:
        make_cutlass_metadata_layout(_metadata_buffer(meta_dtype), mma_dtype, arch="8.0")
    assert "metadata should be" in str(exc_info.value)


@pytest.mark.parametrize(
    "mma_dtype",
    [
        "float8_e4m3fn",  # torch-canonical spelling of a supported fp8 dtype
        "float8_e4m3fnuz",
        "not_a_dtype",
    ],
)
@pytest.mark.parametrize("meta_dtype", ["uint16", "uint32"])
def test_sm8x_metadata_layout_rejects_unlisted_operand_dtype(mma_dtype, meta_dtype):
    """An operand dtype outside the sm8x allowlists is rejected.

    The width check used to be a pair of allowlists with no fallback, so every
    other spelling fell through to the width-derived placement and silently got
    the wrong group/interweave convention.
    """
    with pytest.raises(NotImplementedError) as exc_info:
        make_cutlass_metadata_layout(_metadata_buffer(meta_dtype), mma_dtype, arch="8.0")
    assert mma_dtype in str(exc_info.value)


def test_sm90_metadata_layout_rejects_unlisted_operand_dtype():
    """Control: the sm90 arm already fail-closes on an unrecognized operand dtype."""
    with pytest.raises(NotImplementedError) as exc_info:
        make_cutlass_metadata_layout(_metadata_buffer("uint8"), "not_a_dtype", arch="9.0", block_k=64)
    assert "not_a_dtype" in str(exc_info.value)


if __name__ == "__main__":
    tilelang.testing.main()
