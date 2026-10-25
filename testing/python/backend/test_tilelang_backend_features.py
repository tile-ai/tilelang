import pytest
import tvm_ffi

import tilelang.testing


@pytest.mark.parametrize("available", [False, True])
def test_ascend_backend_detection_requires_native_pass_factory(monkeypatch, available):
    requested = []
    factory = object() if available else None

    def lookup(name, *, allow_missing=False):
        requested.append((name, allow_missing))
        return factory

    monkeypatch.setattr(tvm_ffi, "get_global_func", lookup)
    assert tilelang.testing.ascend_backend_compiled() is available
    assert requested == [("tl.transform.AutoSchedule", True)]
