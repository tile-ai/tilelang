"""PTO cache dispatch selects the PTO target variant for an Ascend target."""

import pytest

from tilelang.cache import _resolve_cache_dispatch


@pytest.mark.pto
def test_ascend_target_with_pto_execution_uses_pto_cache_dispatch():
    _, context, _ = _resolve_cache_dispatch("ascend", None, "pto", False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


@pytest.mark.pto
def test_ascend_env_defaults_can_select_pto(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "ascend")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "pto")

    _, context, _ = _resolve_cache_dispatch(None, None, None, False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys
