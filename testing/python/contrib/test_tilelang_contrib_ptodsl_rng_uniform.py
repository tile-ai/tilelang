"""Regression tests for the float32 range of the PTODSL Philox uniform mapping.

`ptodsl` is a device-side package that is not installed in CI, so the tests
load `rng.py` directly (with a stub `pto`) and re-derive the mapping from the
constants `PhiloxRNG` uses: no uint32 draw may round up to 1.0.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_RNG_PATH = Path(__file__).resolve().parents[3] / "tilelang" / "contrib" / "ptodsl" / "rng.py"
_spec = importlib.util.spec_from_file_location("_tilelang_ptodsl_rng", _RNG_PATH)
_module = importlib.util.module_from_spec(_spec)
# rng.py only needs `pto` at call time. Install a stub just for this load and
# restore `sys.modules` afterwards, so a later test that imports the real
# `ptodsl` package (or its submodules) is not handed a bare module stub.
_stub = types.ModuleType("ptodsl")
_stub.pto = types.SimpleNamespace()
_previous = sys.modules.get("ptodsl")
sys.modules["ptodsl"] = _stub
try:
    _spec.loader.exec_module(_module)
finally:
    if _previous is None:
        del sys.modules["ptodsl"]
    else:
        sys.modules["ptodsl"] = _previous
PhiloxRNG = _module.PhiloxRNG


def _uniform(u: int) -> np.float32:
    k = np.uint32(u) >> np.uint32(PhiloxRNG._RAND_UNIFORM_SHIFT)
    # Every integer below 2**24 is exact in float32, and the 2**-24 scaling is
    # exact as well, so this mirrors the device-side expression exactly.
    return np.float32(np.float32(k) * np.float32(PhiloxRNG._RAND_2POW24_INV))


def test_top_of_range_stays_below_one():
    assert PhiloxRNG._RAND_UNIFORM_SHIFT == 8
    assert _uniform(0xFFFFFFFF) < np.float32(1.0)
    assert _uniform(0xFFFFFFFF) == np.float32(1.0 - 2.0**-24)


def test_every_bucket_is_exact_and_below_one():
    buckets = np.arange(1 << 24, dtype=np.uint32).astype(np.float32)
    values = buckets * np.float32(PhiloxRNG._RAND_2POW24_INV)
    assert values.dtype == np.float32
    assert float(values.min()) == 0.0
    assert float(values.max()) == 1.0 - 2.0**-24
    assert np.all(values < np.float32(1.0))


def test_old_scaling_would_reach_one():
    """Documents the defect: rounding the full word to float32 can hit 1.0."""
    inv = np.float32(2.3283064e-10)
    half = np.float32(inv / np.float32(2.0))
    rounded = np.float32(np.uint32(0xFFFFFFFF))
    assert np.float32(rounded * inv + half) == np.float32(1.0)
