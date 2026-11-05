"""Check the FP4 instruction format independently of its storage tag."""

import pytest
from tvm_ffi import dtype as DataType

from tilelang import _ffi_api


def _descriptor(a, b, scaled):
    if scaled:
        return int(_ffi_api.get_tcgen5_blockscaled_instr_desc(128, 128, a, b, False, True, -1, 1, 1, 2))
    return int(_ffi_api.get_tcgen5_instr_desc(128, 128, 32, a, b, DataType("float32"), False, True, -1, 1))


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("unpack_a,unpack_b", [(False, False), (False, True), (True, False), (True, True)])
def test_packed_and_unpacked_fp4_have_the_same_instruction_descriptor(scaled, unpack_a, unpack_b):
    packed = DataType("float4_e2m1fn")
    unpacked = DataType("custom[float4_e2m1_unpacked]8")
    descriptor = _descriptor(unpacked if unpack_a else packed, unpacked if unpack_b else packed, scaled)
    assert descriptor == _descriptor(packed, packed, scaled)
    assert ((descriptor >> 7) & 7, (descriptor >> 10) & 7) == (5, 5)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("position", ["a", "b"])
@pytest.mark.parametrize("dtype", ["custom[tfloat32]8", "custom[tfloat32]32", "custom[float4_e2m1_unpacked]16"])
def test_unrelated_custom_types_and_wrong_unpacked_storage_width_are_rejected(scaled, position, dtype):
    unsupported = DataType(dtype)
    fp8 = DataType("float8_e4m3fn")
    a, b = (unsupported, fp8) if position == "a" else (fp8, unsupported)
    message = "Unsupported dtype for block-scaled mxf8f6f4 descriptor" if scaled else "Unsupported dtype for TCGEN5MMA descriptor"
    with pytest.raises(RuntimeError, match=message):
        _descriptor(a, b, scaled)
