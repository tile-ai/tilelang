"""Carry arithmetic uses matching full 32-bit integer registers."""

import pytest
from tilelang.ascend import language as T
from tvm import tirx
from tvm.script.ir_builder import IRBuilder


@pytest.mark.parametrize("name", ["vaddc", "vsubc", "vaddcs", "vsubcs"])
@pytest.mark.parametrize(
    "lhs,rhs",
    [
        ("float32x64", "float32x64"),
        ("int16x128", "int16x128"),
        ("int32x32", "int32x32"),
        ("int32x64", "uint32x64"),
        ("int32x64", "float32x64"),
    ],
)
def test_carry_rejects_invalid_register_types(name, lhs, rhs):
    with IRBuilder(), T.sblock("root"), T.SimdVF():
        args = [tirx.Var("lhs", lhs), tirx.Var("rhs", rhs)]
        if name.endswith("cs"):
            args.append(tirx.Var("carry", "boolx256"))
        with pytest.raises(TypeError, match="requires matching int32x64 or uint32x64"):
            getattr(T.simd, name)(*args)


@pytest.mark.parametrize("name", ["vaddc", "vsubc", "vaddcs", "vsubcs"])
@pytest.mark.parametrize("dtype", ["int32x64", "uint32x64"])
def test_carry_preserves_integer_result_types(name, dtype):
    with IRBuilder(), T.sblock("root"), T.SimdVF():
        args = [tirx.Var("lhs", dtype), tirx.Var("rhs", dtype)]
        if name.endswith("cs"):
            args.append(tirx.Var("carry", "boolx256"))
        carry, value = getattr(T.simd, name)(*args)
        assert str(carry.dtype) == "boolx256"
        assert str(value.dtype) == dtype
