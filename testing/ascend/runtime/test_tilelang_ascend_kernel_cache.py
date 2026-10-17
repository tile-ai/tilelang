"""Ascend cache artifacts preserve the selected execution backend's ABI."""

import pytest
from tilelang.cache import _resolve_cache_dispatch


@pytest.mark.parametrize("backend, host_suffix", [("tvm_ffi", ".c"), ("cython", ".asc")])
def test_ascend_cache_source_suffixes(backend, host_suffix):
    cache, context, _ = _resolve_cache_dispatch("ascend", None, backend, False)
    assert context.module.name == "ascend"
    assert context.execution_backend.name == backend
    assert cache.device_kernel_path.endswith(".asc")
    assert cache.host_kernel_path.endswith(host_suffix)


def test_ascend_target_with_pto_execution_uses_pto_cache_dispatch():
    _, context, _ = _resolve_cache_dispatch("ascend", None, "pto", False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


def test_ascend_env_defaults_can_select_pto(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "ascend")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "pto")

    _, context, _ = _resolve_cache_dispatch(None, None, None, False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys
