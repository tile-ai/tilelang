"""Tests for the FP4 E2M1 unpacked shared-memory dtype."""

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang import _ffi_api
from tvm import DataType


def test_float4_e2m1_unpacked_dtype_properties():
    dt = T.float4_e2m1_unpacked
    assert dt.bits == 8
    assert int(dt.type_code) == 131
    assert str(dt) == "custom[float4_e2m1_unpacked]8"
    assert dt.lanes == 1


def test_float4_e2m1_unpacked_tir_dtype_tag():
    from tilelang import language as T_lang

    @T_lang.prim_func
    def main():
        # IR-level tag only; runtime numeric conversion is intentionally absent.
        T_lang.evaluate(T_lang.float4_e2m1_unpacked(0.0))

    assert main.body.value.dtype.is_float4_e2m1_unpacked()


def test_float4_e2m1_unpacked_distinct_from_packed():
    packed = T.float4_e2m1fn
    unpacked = T.float4_e2m1_unpacked
    assert packed.bits == 4
    assert unpacked.bits == 8
    assert packed != unpacked


def test_float4_e2m1_unpacked_dtype_helpers():
    from tilelang.language.dtypes import is_float4, is_float4_e2m1fn, is_float4_e2m1_unpacked

    packed = T.float4_e2m1fn
    unpacked = T.float4_e2m1_unpacked
    assert packed.is_float4_e2m1fn()
    assert not packed.is_float4_e2m1_unpacked()
    assert packed.is_float4()
    assert unpacked.is_float4_e2m1_unpacked()
    assert not unpacked.is_float4_e2m1fn()
    assert unpacked.is_float4()
    assert T.float4_e2m1fnx2.is_float4()
    assert not hasattr(T, "float4_e2m1_unpackedx2")
    assert is_float4_e2m1fn(packed)
    assert is_float4_e2m1_unpacked(unpacked)
    assert is_float4("float4_e2m1fn")
    assert is_float4("custom[float4_e2m1_unpacked]8")
    assert not is_float4(T.float16)


@tilelang.testing.requires_cuda
def test_float4_simt_copy_bans_packed_to_unpacked():
    @T.prim_func
    def main(
        A: T.Tensor((64, 128), T.float4_e2m1fn),
    ):
        with T.Kernel(T.ceildiv(128, 64), T.ceildiv(64, 32), threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 64), T.float4_e2m1_unpacked)
            T.copy(A[by * 32, bx * 64], A_shared)

    with pytest.raises(tvm.TVMError, match="SIMT copy from packed global float4_e2m1fn"):
        tilelang.compile(main, target="cuda")


def _decode_tcgen05_formats(desc: int) -> tuple[int, int]:
    return (desc >> 7) & 0x7, (desc >> 10) & 0x7


def test_tcgen05_instr_desc_fp4_unpacked_x_fp8():
    fp4 = DataType("custom[float4_e2m1_unpacked]8")
    fp8 = DataType("float8_e4m3fn")
    f32 = DataType("float32")
    desc = int(_ffi_api.get_tcgen5_instr_desc(128, 128, 32, fp4, fp8, f32, True, True, 1, 1))
    a_format, b_format = _decode_tcgen05_formats(desc)
    assert a_format == 5
    assert b_format == 0


def test_tcgen05_blockscaled_instr_desc_mxfp4_unpacked_x_mxfp8():
    fp4 = DataType("custom[float4_e2m1_unpacked]8")
    fp8 = DataType("float8_e4m3fn")
    desc = int(_ffi_api.get_tcgen5_blockscaled_instr_desc(128, 128, fp4, fp8, True, True, 1, 1, 0, 0))
    a_format, b_format = _decode_tcgen05_formats(desc)
    assert a_format == 5
    assert b_format == 0
    assert (desc >> 23) & 0x1 == 1  # scale_format = E8M0


if __name__ == "__main__":
    test_float4_e2m1_unpacked_dtype_properties()
    test_float4_e2m1_unpacked_tir_dtype_tag()
    test_float4_e2m1_unpacked_distinct_from_packed()
    test_float4_e2m1_unpacked_dtype_helpers()
    print("ok")
