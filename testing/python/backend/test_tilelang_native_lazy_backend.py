"""Cold public backend/cache lookups preserve their owning implementations."""

import os
import subprocess
import sys

import pytest


def _process(source, tmp_path):
    environment = {**os.environ, "TILELANG_CACHE_DIR": str(tmp_path / "cache")}
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["cpu", "cuda", "cutedsl", "rocm", "metal", "webgpu", "ascend", "pto"])
def test_cold_lookup_loads_the_builtin_backend(name, tmp_path):
    _process(
        f"""
import sys
from tilelang.backend import get_backend, list_backends
module = get_backend({name!r})
assert module.name == {name!r}
assert get_backend({name!r}) is module
assert 'tvm' not in sys.modules
if {name!r} == 'cpu':
    assert 'tilelang.cuda.pipeline' not in sys.modules
    assert 'tilelang.ascend.pipeline' not in sys.modules
# Explicit enumeration loads the built-ins after the cold-lookup check.
backends = list_backends()
assert backends[{name!r}] is module
assert set(("cpu", "cuda", "cutedsl", "rocm", "metal", "webgpu", "ascend", "pto")) <= set(backends)
assert 'tvm' not in sys.modules
""",
        tmp_path,
    )


def test_unknown_backend_keeps_public_diagnostic(tmp_path):
    _process(
        """
from tilelang.backend import get_backend
try:
    get_backend('missing-native-test-backend')
except ValueError as error:
    assert "Unknown backend 'missing-native-test-backend'" in str(error)
else:
    raise AssertionError('Unknown backend accepted')
""",
        tmp_path,
    )


@pytest.mark.parametrize("first", ["base", "ffi", "nvrtc"])
def test_cache_singletons_keep_concrete_types_under_lazy_and_concurrent_use(first, tmp_path):
    _process(
        f"""
from concurrent.futures import ThreadPoolExecutor
from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
classes = {{'base': KernelCache, 'ffi': TVMFFIKernelCache, 'nvrtc': NVRTCKernelCache, 'cython': CythonKernelCache}}
classes[{first!r}]()
with ThreadPoolExecutor(max_workers=8) as pool:
    values = list(pool.map(lambda cls: (cls, cls()), list(classes.values()) * 16))
instances = {{}}
for cls, instance in values:
    assert type(instance) is cls, (cls, type(instance))
    assert instances.setdefault(cls, instance) is instance
assert len({{id(value) for value in instances.values()}}) == len(classes)
for key, cls in classes.items():
    cls()._memory_cache['native-singleton-test'] = key
for key, cls in classes.items():
    assert cls()._memory_cache['native-singleton-test'] == key
""",
        tmp_path,
    )
