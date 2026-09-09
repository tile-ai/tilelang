# Regression test for https://github.com/tile-ai/tilelang/issues/3018
# interleave_weight used ``torch.int32(0x...)`` to build bit masks on its
# nbits=1 / nbits=2 branches; ``torch.int32`` is a dtype object, not a
# constructor, so those branches raised ``TypeError`` instead of returning.
# Host-side only: the utility is a plain CPU torch tensor transform.
import pytest
import torch

import tilelang.testing
from examples.dequantize_gemm.quantize.utils import interleave_weight

_MASK32 = 0xFFFFFFFF


def _reference_interleave(word: int, nbits: int, target_dtype: str) -> int:
    """Pure-Python reference for interleave_weight on a single uint32 word."""
    bits_stride = 8 if target_dtype == "int8" else 16
    mask = (1 << nbits) - 1
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    new = 0
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new |= ((word >> (nbits * offset)) & mask) << shift
    new &= _MASK32

    if nbits == 1 and target_dtype == "int8":
        out = new & 0xF0F00F0F
        out |= ((new & 0x000000F0) >> 4) << 16
        out |= ((new & 0x0000F000) >> 12) << 24
        out |= ((new & 0x000F0000) >> 16) << 4
        out |= ((new & 0x0F000000) >> 24) << 12
        return out & _MASK32
    if nbits == 2 and target_dtype == "float16":
        out = new & 0xFF0000FF
        out |= ((new & 0x0000FF00) >> 8) << 16
        out |= ((new & 0x00FF0000) >> 16) << 8
        return out & _MASK32
    if nbits == 1 and target_dtype == "float16":
        out = new & 0xF000000F
        out |= ((new & 0x000000F0) >> 4) << 8
        out |= ((new & 0x00000F00) >> 8) << 16
        out |= ((new & 0x0000F000) >> 12) << 24
        out |= ((new & 0x000F0000) >> 16) << 4
        out |= ((new & 0x00F00000) >> 20) << 12
        out |= ((new & 0x0F000000) >> 24) << 20
        return out & _MASK32
    return new


def _as_uint32_words(tensor: torch.Tensor) -> list[int]:
    return [w & _MASK32 for w in tensor.contiguous().view(torch.int32).flatten().tolist()]


@pytest.mark.parametrize(
    "nbits,target_dtype",
    [
        # The three special branches that previously raised TypeError.
        (1, "int8"),
        (2, "float16"),
        (1, "float16"),
        # Generic tail, kept as controls.
        (2, "int8"),
        (4, "float16"),
        (4, "int8"),
    ],
)
def test_interleave_weight_matches_reference(nbits, target_dtype):
    torch.manual_seed(0)
    qweight = torch.randint(-128, 128, (4, 8), dtype=torch.int8)

    out = interleave_weight(qweight.clone(), nbits, target_dtype)

    assert out.dtype == torch.int8
    assert out.shape == qweight.shape
    expected = [_reference_interleave(w, nbits, target_dtype) for w in _as_uint32_words(qweight)]
    assert _as_uint32_words(out) == expected


if __name__ == "__main__":
    tilelang.testing.main()
