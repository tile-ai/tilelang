"""Predicate construction is valid in expression position."""

import re

import pytest
import tilelang
import tilelang.ascend.language as T


@pytest.mark.parametrize("inline", [False, True])
def test_pset_can_be_used_as_an_inline_mask(inline):
    @T.prim_func
    def main(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1):
            a = T.alloc_shared((64,), T.float32)
            b = T.alloc_shared((64,), T.float32)
            T.copy(A, a)
            with T.SimdVF():
                x = T.simd.vld(a[0])
                if inline:
                    y = T.simd.vadd(x, x, T.simd.pset(32, "PAT_VL16"))
                else:
                    mask = T.simd.pset(32, "PAT_VL16")
                    y = T.simd.vadd(x, x, mask)
                T.simd.vsts(b[0], y)
            T.copy(b, B)

    source = tilelang.lower(main, target="ascend").kernel_source
    vadd = next(line for line in source.splitlines() if "simd_inst::vadd(" in line)
    if inline:
        mask_operand = "asc_create_mask_b32(PAT_VL16)"
    else:
        declaration = re.search(r"vector_bool\s+(\w+)\s*=\s*asc_create_mask_b32\(PAT_VL16\)", source)
        assert declaration is not None
        mask_operand = declaration.group(1)
    assert f", {mask_operand}, MODE_ZEROING)" in vadd
