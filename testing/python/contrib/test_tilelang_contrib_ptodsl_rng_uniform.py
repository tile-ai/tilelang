"""Exercise the production uniform conversion without a PTO installation or NPU."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def uniform(monkeypatch):
    # Only emulate the scalar operations used by rand_uniform; keep its actual
    # conversion and endpoint selection in the production method under test.
    pto = types.SimpleNamespace(
        ui32=np.uint32,
        f32=np.float32,
        cast=lambda value, dtype, **kwargs: np.asarray(value).astype(dtype)[()],
        const=lambda value, dtype: dtype(value),
        select=lambda condition, true_value, false_value: true_value if condition else false_value,
    )
    stub = types.ModuleType("ptodsl")
    stub.pto = pto
    path = Path(__file__).resolve().parents[3] / "tilelang" / "contrib" / "ptodsl" / "rng.py"
    spec = importlib.util.spec_from_file_location("_tilelang_ptodsl_rng_uniform_test", path)
    module = importlib.util.module_from_spec(spec)
    # Restore even an already-imported real ptodsl immediately after loading.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "ptodsl", stub)
        spec.loader.exec_module(module)

    def draw(word):
        rng = module.PhiloxRNG.__new__(module.PhiloxRNG)
        # The production integer RNG carries its uint32 words as i32 bits.
        rng.rand = Mock(return_value=(np.uint32(word).view(np.int32), 8))
        result, next_index = rng.rand_uniform(7)
        rng.rand.assert_called_once_with(7)
        assert next_index == 8
        assert isinstance(result, np.float32)
        return result

    return draw


def test_uniform_excludes_rounded_endpoint(uniform):
    expected = np.nextafter(np.float32(1), np.float32(0))
    # These are all uint32 inputs whose nearest-even float32 cast reaches 2**32.
    for word in range(0xFFFFFF80, 0x100000000):
        result = uniform(word)
        assert result < np.float32(1), hex(word)
        assert result == expected, hex(word)


@pytest.mark.parametrize("word", [0, 1, 255, 256, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFF00, 0xFFFFFF7F])
def test_uniform_preserves_non_endpoint_draws(uniform, word):
    # Compatibility oracle: the original full-word conversion, independently
    # expressed with exact powers of two rather than production constants.
    expected = np.float32(word) * np.float32(2.0**-32) + np.float32(2.0**-33)
    result = uniform(word)
    assert np.float32(0) < result < np.float32(1)
    assert result == expected
