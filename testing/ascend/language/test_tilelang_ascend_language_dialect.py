"""The Ascend dialect exposes its extensions without changing common APIs."""

import tilelang.language.common as common_language

ASCEND_ONLY_NAMES = {
    "AscendTileScheduler",
    "Cube",
    "MixedKernel",
    "SimdVF",
    "SimtVF",
    "Vector",
    "alloc_l0a",
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
