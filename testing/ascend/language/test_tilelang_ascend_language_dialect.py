"""The Ascend dialect exposes its extensions without changing common APIs."""

import pytest
import tilelang.language.common as common_language
from tilelang.ascend import language as T
from tvm import tirx
from testing.ascend._ir import allocated_buffer, nodes

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
    bindings = [block.annotations["tl.l0_sf_bindings"] for block in nodes(main, tirx.SBlock) if "tl.l0_sf_bindings" in block.annotations]
    assert len(bindings) == 1 and bindings[0][sf.data].same_as(a.data)
