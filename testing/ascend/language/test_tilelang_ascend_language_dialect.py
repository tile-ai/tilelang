"""The Ascend dialect exposes its extensions without changing common APIs."""

import pytest
import tilelang
import tilelang.language.common as common_language
from tilelang.ascend import language as T
from tilelang._compat import tirx
from testing.ascend._ir import allocated_buffer, nodes
from tvm_tirx import ir

ASCEND_ONLY_NAMES = {
    "AscendTileScheduler",
    "Cube",
    "MixedKernel",
    "SimdVF",
    "SimtVF",
    "Vector",
    "alloc_l0a",
    "alloc_l0a_sf",
    "alloc_l0b_sf",
    "alloc_l0b",
    "alloc_l0c",
    "alloc_l1",
    "dual_copy",
}


def test_ascend_language_composes_common_and_ascend_symbols():
    from tilelang.ascend import language as T

    assert T.__tilelang_dialect__ == "ascend"
    assert set(T.__all__) >= set(common_language.__all__)
    assert set(T.__all__) >= ASCEND_ONLY_NAMES
    for name in ASCEND_ONLY_NAMES:
        assert hasattr(T, name)


def test_ascend_dialect_owns_its_backend_knobs():
    """#3203: each dialect declares the op hints its backend honors."""
    from tilelang.ascend import language as T

    # The Ascend copy hints must not be advertised by the common surface.
    import inspect

    import tilelang.language.copy_op as common_copy_op

    common_copy = inspect.signature(common_copy_op.copy).parameters
    assert {"transpose", "l2_cache_ctrl", "unit_flag_ctrl", "sub_blockid", "pad_value", "data_select"}.isdisjoint(common_copy)
    assert {"coalesced_width", "annotations", "loop_layout"} <= set(common_copy)

    ascend_copy = inspect.signature(T.copy).parameters
    assert {"transpose", "l2_cache_ctrl", "unit_flag_ctrl", "sub_blockid", "pad_value", "data_select"} <= set(ascend_copy)


def test_ascend_dialect_does_not_leak_into_common():
    assert ASCEND_ONLY_NAMES.isdisjoint(common_language.__all__)
    # copy/gemm/unroll are common names the Ascend dialect shadows, so they are
    # deliberately not part of the disjointness check above.


@pytest.mark.parametrize("leading", [(), (3,), (2, 3)])
def test_allocation_records_binding_without_a_consumer(leading):
    @T.prim_func
    def main():
        with T.Kernel(1):
            a = T.alloc_l0a((*leading, 32, 128), "float8_e4m3fn")
            _sf = T.alloc_l0a_sf(a)
            T.evaluate(0)

    a, sf = allocated_buffer(main, "a"), allocated_buffer(main, "_sf")
    assert tuple(int(x) for x in sf.shape) == (*leading, 32, 2)
    bindings = [region.attrs["tl.l0_sf_bindings"] for region in nodes(main, ir.RegionStmt) if "tl.l0_sf_bindings" in region.attrs]
    assert len(bindings) == 1 and bindings[0][sf].same_as(a)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "float16", "bfloat16", "int32", "uint32", "float32"])
def test_copy_pad_value_infers_native_primitive_type(dtype):
    value = ir.Var("value", dtype)
    call = T.ascend_set_copy_pad_value(value)
    assert call.op.name == "tl.ascend_set_copy_pad_value"
    assert call.ty == ir.PrimType("void")
    assert str(call.args[0].ty.dtype) == dtype


def test_copy_pad_value_requires_explicit_literal_type():
    with pytest.raises(ValueError, match="requires dtype for Python literals"):
        T.ascend_set_copy_pad_value(1.0)


def test_copy_pad_value_emits_a_scheduled_native_register_write():
    @T.prim_func
    def main(source: T.Tensor((4, 17), "float32"), output: T.Tensor((4, 17), "float32")):
        with T.Kernel(1):
            storage = T.alloc_shared((4, 17), "float32")
            T.copy(source, storage, pad_value=3.5)
            T.copy(storage, output)

    calls = nodes(main, ir.Call)
    pad = next(call for call in calls if call.op.name == "tl.ascend_set_copy_pad_value")
    copy = next(call for call in calls if call.op.name == "tl.tileop.ascend_copy")
    assert pad.ty == ir.PrimType("void")
    assert str(pad.args[0].ty.dtype) == "float32"
    assert int(copy.attrs["data_select"]) == 1
    artifact = tilelang.lower(main, target="ascend", enable_host_codegen=False, enable_device_compile=False)
    assert "asc_set_copy_pad_val" in artifact.kernel_source
    assert artifact.kernel_source.index("asc_set_copy_pad_val") < artifact.kernel_source.index("asc_copy_gm2ub")
