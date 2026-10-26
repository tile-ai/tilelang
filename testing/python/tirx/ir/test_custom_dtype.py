import pytest
import tvm_ffi

from tilelang._compat import ir
from tvm_tirx import script


@pytest.fixture(autouse=True)
def custom_types():
    ir.register_custom_dtype("test_custom_float", 240, "float32")
    ir.register_custom_dtype("test_custom_byte", 241, "uint8")


@pytest.mark.parametrize("lanes", [1, 4, 16, -4])
@pytest.mark.parametrize("code,bits,storage", [(240, 32, "float32"), (241, 8, "uint8")])
def test_custom_dtype_identity_storage_and_roundtrip(code, bits, storage, lanes):
    primitive = ir.prim_type_from_dlpack((code, bits, lanes))
    assert ir.custom_dtype_storage(primitive).dtype.type_code == tvm_ffi.dtype(storage).type_code
    assert ir.custom_dtype_storage(primitive).dtype.bits == bits
    assert ir.custom_dtype_storage(primitive).dtype.lanes == primitive.dtype.lanes
    ir.assert_structural_equal(primitive, ir.load_json(ir.save_json(primitive)))
    restored = eval(ir.script(primitive), {"I": script.ir, "T": script.tirx, "tlir": ir})
    ir.assert_structural_equal(primitive, restored)
    if lanes == 1:
        constant = ir.const(1.25, primitive)
        assert isinstance(constant, ir.FloatImm)
        assert constant.ty == primitive


def test_custom_dtype_registration_is_idempotent_not_replaceable():
    ir.register_custom_dtype("test_custom_float", 240, "float32")
    assert tvm_ffi.dtype("custom[test_custom_float]32").type_code == 240
    with pytest.raises(ValueError, match="name is already registered"):
        ir.register_custom_dtype("test_custom_float", 242, "float32")
    with pytest.raises(ValueError, match="name is already registered"):
        ir.register_custom_dtype("test_custom_float", 240, "uint32")
    with pytest.raises(ValueError, match="code is already registered"):
        ir.register_custom_dtype("test_custom_other", 240, "float32")


@pytest.mark.parametrize("code", [0, 128, 256, -1])
def test_custom_dtype_rejects_reserved_codes(code):
    with pytest.raises(ValueError, match="codes must be"):
        ir.register_custom_dtype("test_custom_invalid", code, "float32")


@pytest.mark.parametrize("storage", ["int4", "float32x4", "custom[test_custom_float]"])
def test_custom_dtype_requires_native_scalar_storage(storage):
    with pytest.raises(ValueError, match="byte-addressable native scalar"):
        ir.register_custom_dtype("test_custom_invalid", 242, storage)


def test_custom_dtype_validates_physical_bit_width():
    with pytest.raises(ValueError, match="requires 8 storage bits"):
        ir.custom_dtype_storage("custom[test_custom_byte]32")
    assert ir.custom_dtype_storage("float32") == ir.PrimType("float32")
