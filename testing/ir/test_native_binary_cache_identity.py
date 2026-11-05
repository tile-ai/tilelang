"""Binary cache opt-in identities include the compiler and FFI actually loaded."""

from hashlib import sha256
import json
from pathlib import Path

import pytest
import tvm_ffi
from tvm_tirx import base

from tilelang.cache.ascend_binary_cache import AscendBinaryCache
from tilelang.cache.binary_cache import BinaryCache
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env


@pytest.mark.parametrize("cache", [CUDABinaryCache, AscendBinaryCache])
def test_enabled_stamp_changes_key_and_identifies_loaded_libraries(cache, monkeypatch):
    cache._get_tilelang_lib_stamp.cache_clear()
    KernelCache._get_tilelang_lib_stamp.cache_clear()
    original = {"source": "unchanged device source", "options": ["same compiler options"]}
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    without_stamp = cache._finalize_key(dict(original))
    assert without_stamp == sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "1")
    with_stamp = cache._finalize_key(dict(original))
    assert with_stamp != without_stamp
    stamp = cache._get_tilelang_lib_stamp()
    assert stamp
    entries = dict(item.split(":", 1) for item in stamp.split("|"))
    libraries = dict(base._LOADED_LIBS)
    libraries["tvm_ffi"] = tvm_ffi.LIB
    assert {"tvm_tirx_compiler", "tvm_ffi"} <= libraries.keys()
    assert entries.keys() == libraries.keys()
    for name, library in libraries.items():
        path = Path(library._name).resolve()
        assert entries[name] == sha256(path.read_bytes()).hexdigest()
    assert with_stamp == sha256(json.dumps(dict(original, tilelang_lib=stamp), sort_keys=True).encode()).hexdigest()


def test_kernel_cache_stamp_uses_the_fixed_ffi_loaded_handle():
    KernelCache._get_tilelang_lib_stamp.cache_clear()
    stamp = KernelCache._get_tilelang_lib_stamp()
    entries = dict(item.split(":", 1) for item in stamp.split("|"))
    assert entries["tvm_ffi"] == sha256(Path(tvm_ffi.LIB._name).resolve().read_bytes()).hexdigest()


@pytest.mark.parametrize("cache", [CUDABinaryCache, AscendBinaryCache])
def test_disabled_stamp_does_not_query_native_library_identity(cache, monkeypatch):
    def unexpected_read():
        pytest.fail("Disabled library stamps must not read compiler libraries")

    monkeypatch.setattr(BinaryCache, "_get_tilelang_lib_stamp", staticmethod(unexpected_read))
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    key_data = {"source": "unchanged device source", "options": []}
    assert cache._finalize_key(dict(key_data)) == sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
