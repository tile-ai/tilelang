"""Ascend counterpart to testing/python/language/test_tilelang_language_dialect.py.

The upstream test covers the CUDA/ROCm/Metal/CPU dialects introduced by #2734.
This file covers the Ascend dialect that this fork adds on the same pattern:
``tilelang.ascend.language`` = the common surface plus Ascend extensions.
"""

import importlib

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

# Symbols the Ascend dialect shadows with its own implementation instead of
# re-exporting the common one. #3203 moved each backend's op hints into its
# owning dialect, so Ascend now declares the copy/gemm/unroll knobs the Ascend
# backend actually honors rather than advertising them on the common surface.
ASCEND_OWNED_SYMBOLS = {
    "Kernel": "tilelang.ascend.language.kernel",
    "MixedKernel": "tilelang.ascend.language.kernel",
    "copy": "tilelang.ascend.language.copy_op",
    "gemm": "tilelang.ascend.language.gemm_op",
    "unroll": "tilelang.ascend.language.loop",
}


def test_ascend_language_composes_common_and_ascend_symbols():
    from tilelang.ascend import language as T

    assert T.__tilelang_dialect__ == "ascend"
    assert set(T.__all__) >= set(common_language.__all__)
    assert set(T.__all__) >= ASCEND_ONLY_NAMES
    for name in ASCEND_ONLY_NAMES:
        assert hasattr(T, name)


def test_ascend_whole_module_implementations_live_under_language():
    from tilelang.ascend import language as T

    assert T.SimtVF.__module__ == "tilelang.ascend.language.frame"
    assert T.Cube.__module__ == "tilelang.ascend.language.frame"
    assert T.Vector.__module__ == "tilelang.ascend.language.frame"


def test_ascend_intrinsics_live_in_themed_dialect_submodules():
    from tilelang.ascend import language as T

    assert T.ascend_pipe_barrier.__module__ == "tilelang.ascend.language.sync"
    assert T.ascend_set_copy_pad_value.__module__ == "tilelang.ascend.language.dma"
    assert T.set_atomic.__module__ == "tilelang.ascend.language.mode"


def test_ascend_dialect_owns_its_backend_knobs():
    """#3203: each dialect declares the op hints its backend honors."""
    from tilelang.ascend import language as T

    for name, module in ASCEND_OWNED_SYMBOLS.items():
        assert getattr(T, name).__module__ == module, name

    # The Ascend copy hints must not be advertised by the common surface.
    import inspect

    import tilelang.language.copy_op as common_copy_op

    common_copy = inspect.signature(common_copy_op.copy).parameters
    assert {"transpose", "l2_cache_ctrl", "unit_flag_ctrl", "sub_blockid", "scale", "pad_value", "data_select"}.isdisjoint(common_copy)
    assert {"coalesced_width", "annotations", "loop_layout"} <= set(common_copy)

    ascend_copy = inspect.signature(T.copy).parameters
    assert {"transpose", "l2_cache_ctrl", "unit_flag_ctrl", "sub_blockid", "scale", "pad_value", "data_select"} <= set(ascend_copy)


def test_ascend_dialect_does_not_leak_into_common():
    assert ASCEND_ONLY_NAMES.isdisjoint(common_language.__all__)
    # copy/gemm/unroll are common names the Ascend dialect shadows, so they are
    # deliberately not part of the disjointness check above.


def test_ascend_submodules_are_exposed():
    from tilelang.ascend import language as T

    assert T.simd is importlib.import_module("tilelang.ascend.language.simd")
