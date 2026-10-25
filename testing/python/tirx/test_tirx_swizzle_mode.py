import pytest
import tvm_ffi
from tvm_tirx import ir

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize(
    "ordinal,width,wgmma,tcgen,alignment",
    [
        (0, 1, 0, 0, 128),
        (1, 32, 3, 6, 256),
        (2, 64, 2, 4, 512),
        (3, 128, 1, 2, 1024),
    ],
)
def test_native_enum_descriptor_encodings(ordinal, width, wgmma, tcgen, alignment):
    mode = tvm_ffi.get_global_func("tl.swizzle_mode_from_ordinal")(ordinal)
    assert mode.same_as(tvm_ffi.get_global_func("tl.swizzle_mode_from_ordinal")(ordinal))
    assert tvm_ffi.get_global_func("tl.swizzle_mode_byte_width")(mode) == width
    assert tvm_ffi.get_global_func("tl.swizzle_mode_wgmma_layout_type")(mode) == wgmma
    assert tvm_ffi.get_global_func("tl.swizzle_mode_tcgen05_layout_type")(mode) == tcgen
    assert tvm_ffi.get_global_func("tl.swizzle_mode_smem_alignment")(mode) == alignment
    assert ir.load_json(ir.save_json(mode)).same_as(mode)


@pytest.mark.parametrize("name,width", [("NONE", 1), ("SWIZZLE_32B", 32), ("SWIZZLE_64B", 64), ("SWIZZLE_128B", 128)])
def test_native_enum_requires_a_typed_value(name, width):
    with pytest.raises(TypeError, match="tl.SwizzleMode"):
        tvm_ffi.get_global_func("tl.swizzle_mode_byte_width")(name)
