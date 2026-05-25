"""Tests for float4_e2m1_unpacked (CUTLASS float_e2m1_unpacksmem_t)."""

import tilelang.language as T
from tilelang import tvm as tvm


def test_float4_e2m1_unpacked_dtype_properties():
    dt = T.float4_e2m1_unpacked
    assert dt.bits == 8
    assert int(dt.type_code) == 131
    assert str(dt) == "custom[float4_e2m1_unpacked]8"
    assert dt.lanes == 1


def test_float4_e2m1_unpacked_vector_lanes():
    dt2 = T.float4_e2m1_unpackedx2
    assert dt2.bits == 8
    assert int(dt2.type_code) == 131
    assert dt2.lanes == 2


def test_float4_e2m1_unpacked_tir_cast():
    from tilelang import language as T_lang

    @T_lang.prim_func
    def main():
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
    assert T.float4_e2m1_unpackedx4.is_float4()
    assert is_float4_e2m1fn(packed)
    assert is_float4_e2m1_unpacked(unpacked)
    assert is_float4("float4_e2m1fn")
    assert is_float4("custom[float4_e2m1_unpacked]8")
    assert not is_float4(T.float16)


if __name__ == "__main__":
    test_float4_e2m1_unpacked_dtype_properties()
    test_float4_e2m1_unpacked_vector_lanes()
    test_float4_e2m1_unpacked_tir_cast()
    test_float4_e2m1_unpacked_distinct_from_packed()
    test_float4_e2m1_unpacked_dtype_helpers()
    print("ok")
