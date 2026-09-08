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


def test_ascend_language_composes_common_and_ascend_symbols():
    from tilelang.ascend import language as T

    assert T.__tilelang_dialect__ == "ascend"
    assert T.copy is common_language.copy
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


def test_ascend_dialect_does_not_leak_into_common():
    assert ASCEND_ONLY_NAMES.isdisjoint(common_language.__all__)


def test_ascend_submodules_are_exposed():
    from tilelang.ascend import language as T

    assert T.simd is importlib.import_module("tilelang.ascend.language.simd")
    assert T.vmi is importlib.import_module("tilelang.ascend.language.vmi")
