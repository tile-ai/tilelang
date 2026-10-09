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
