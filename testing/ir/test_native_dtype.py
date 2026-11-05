"""TileLang storage tags must round-trip through the sole FFI dtype registry."""

import pytest
import tilelang  # noqa: F401
import tvm_ffi
from tvm_tirx import ir, tirx


@pytest.mark.parametrize("name,code,bits", [("tfloat32", 130, 32), ("float4_e2m1_unpacked", 131, 8)])
@pytest.mark.parametrize("lanes", [1, 2, 16])
def test_storage_dtype_round_trip(name, code, bits, lanes):
    suffix = f"x{lanes}" if lanes != 1 else ""
    dtype = tvm_ffi.dtype(f"custom[{name}]{bits}{suffix}")
    assert (dtype.type_code, dtype.bits, dtype.lanes) == (code, bits, lanes)
    assert tvm_ffi.dtype(str(dtype)) == dtype
    tensor_type = tirx.TensorType([ir.IntImm("int32", 4)], dtype, scope="shared")
    assert isinstance(tensor_type.dtype, ir.PrimType)
    assert tensor_type.dtype.dtype == dtype
    restored = ir.load_json(ir.save_json(tensor_type))
    assert (restored.dtype.dtype.type_code, restored.dtype.dtype.bits, restored.dtype.dtype.lanes) == (code, bits, lanes)
    assert restored.dtype.dtype == dtype


def test_registry_rejects_codes_below_custom_begin():
    register = tvm_ffi.get_global_func("dtype.register_custom_type")
    for code in (-256, -128, 0, 127, 128, 256, 384):
        with pytest.raises(tilelang.errors.InternalError, match="type code >= DataType::kCustomBegin"):
            register("invalid_code", code)
    assert tvm_ffi.get_global_func("dtype.get_custom_type_name")(130) == "tfloat32"


def test_unknown_custom_dtype_is_not_silently_assigned():
    with pytest.raises(tilelang.errors.InternalError, match="Type name tilelang_nonexistent not registered"):
        tvm_ffi.dtype("custom[tilelang_nonexistent]32")
