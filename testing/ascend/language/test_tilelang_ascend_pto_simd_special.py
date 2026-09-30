"""PTO source lowering for SIMD histogram, prefix, and unpack operations."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


@pytest.mark.pto
def test_pto_dhistv2_codegen():
    @T.prim_func
    def kernel(
        source: T.Tensor((256,), "uint8"),
        output: T.Tensor((128,), "uint16"),
    ):
        with T.Kernel(1):
            source_ub = T.alloc_shared((256,), "uint8")
            output_ub = T.alloc_shared((128,), "uint16")
            histogram = S.alloc_local(1, "uint16")
            T.copy(source, source_ub)
            with T.SimdVF():
                mask = S.pset(8)
                source_v = S.vld(source_ub[0])
                S.dhistv2(histogram[0], source_v, mask, bin=0)
                S.vsts(output_ub[0], histogram[0], S.pset(16))
            T.copy(output_ub, output)

    source = tilelang.lower(kernel, target="pto").kernel_source
    assert "pto.dhistv2(" in source
    assert "tl.simd.dhistv2" not in source


@pytest.mark.pto
def test_pto_vusqz_codegen():
    @T.prim_func
    def kernel(mask_input: T.Tensor((64,), "int32"), output: T.Tensor((64,), "int32")):
        with T.Kernel(1):
            mask_ub = T.alloc_shared((64,), "int32")
            output_ub = T.alloc_shared((64,), "int32")
            T.copy(mask_input, mask_ub)
            with T.SimdVF():
                full = S.pset(32)
                values = S.vld(mask_ub[0])
                predicate = S.vcmps(values, T.int32(1), full, "eq")
                S.vsts(output_ub[0], S.vusqz(predicate, "int32"), full)
            T.copy(output_ub, output)

    source = tilelang.lower(kernel, target="pto").kernel_source
    assert "pto.vusqz(" in source
    assert "pto.vdup(pto.const(0, dtype=pto.i32)" in source
    assert "tl.simd.vusqz" not in source


@pytest.mark.pto
def test_pto_vunpack_codegen():
    @T.prim_func
    def kernel(
        source: T.Tensor((128,), "int16"),
        lower: T.Tensor((64,), "int32"),
        higher: T.Tensor((64,), "int32"),
    ):
        with T.Kernel(1):
            source_ub = T.alloc_shared((128,), "int16")
            lower_ub = T.alloc_shared((64,), "int32")
            higher_ub = T.alloc_shared((64,), "int32")
            T.copy(source, source_ub)
            with T.SimdVF():
                mask = S.pset(32)
                values = S.vld(source_ub[0])
                S.vsts(lower_ub[0], S.vunpack(values, "LOWER"), mask)
                S.vsts(higher_ub[0], S.vunpack(values, "HIGHER"), mask)
            T.copy(lower_ub, lower)
            T.copy(higher_ub, higher)

    source = tilelang.lower(kernel, target="pto").kernel_source
    assert source.count("pto.vunpack(") == 2
    assert "pto.vunpack(" in source and ", 0)" in source and ", 1)" in source
    assert "tl.simd.vunpack" not in source
