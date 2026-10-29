"""Tests for ``tilelang.testing.without_kernel_cache``."""

import pytest

import tilelang.testing
from tilelang.env import CacheState


@pytest.fixture(autouse=True)
def _restore_cache_state():
    enabled = CacheState.is_enabled()
    try:
        yield
    finally:
        if enabled:
            CacheState.enable()
        else:
            CacheState.disable()


def _set_cache_state(enabled: bool) -> None:
    if enabled:
        CacheState.enable()
    else:
        CacheState.disable()


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        assert not CacheState.is_enabled()

    assert CacheState.is_enabled() is initially_enabled


@pytest.mark.parametrize("initially_enabled", [True, False])
def test_without_kernel_cache_restores_the_previous_state_after_the_body_reenables(initially_enabled):
    _set_cache_state(initially_enabled)

    with tilelang.testing.without_kernel_cache():
        CacheState.enable()

    assert CacheState.is_enabled() is initially_enabled


if __name__ == "__main__":
    tilelang.testing.main()
