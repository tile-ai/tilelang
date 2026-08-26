"""Regression tests for cache-entry integrity verification.

A crashed writer (or crashed network-filesystem client) can publish a cache
entry whose kernel_lib.so is truncated or partially written. Such a binary can
dlopen fine and only fail at first kernel launch with CUDA_ERROR_INVALID_IMAGE,
where nothing repairs the entry. The kernel cache therefore records a manifest
of file sizes/hashes on save and verifies it on load, treating mismatches as
cache misses.
"""

import json
from hashlib import sha256
from pathlib import Path

import cloudpickle
import pytest

import tilelang.cache.kernel_cache as kernel_cache_mod
from tilelang.backend import create_backend_context
from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env


class _FakeAdapter:
    def __init__(self, libpath: str):
        self.libpath = libpath

    def get_kernel_source(self):
        return "// host kernel"


class _FakeKernel:
    def __init__(self, libpath: str):
        self.adapter = _FakeAdapter(libpath)
        self.kernel_source = "// device kernel"
        self.params = ["param"]


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    return cache_dir


def _make_fake_kernel(tmp_path):
    lib_path = tmp_path / "kernel_lib.so"
    lib_path.write_bytes(b"fake-so-contents")
    return _FakeKernel(str(lib_path))


def _load_expecting_no_build(cache: KernelCache, key: str, monkeypatch):
    def fail_from_database(cls, **kwargs):
        raise AssertionError("corrupted cache entries must miss before rebuilding from database")

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fail_from_database))
    return cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("cuda", execution_backend="tvm_ffi"),
        out_idx=[0],
        pass_configs=None,
        compile_flags=None,
        func=None,
    )


def _load_expecting_hit(cache: KernelCache, key: str, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(lambda cls, **kwargs: sentinel))
    loaded = cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("cuda", execution_backend="tvm_ffi"),
        out_idx=[0],
        pass_configs=None,
        compile_flags=None,
        func=None,
    )
    return loaded is sentinel


def test_save_writes_manifest_covering_all_files(cache_dirs, tmp_path):
    cache = KernelCache()
    key = "manifest-on-save"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))

    cache_path = Path(cache._get_cache_path(key))
    manifest = json.loads((cache_path / cache.manifest_path).read_text())

    expected_files = {
        cache.device_kernel_path,
        cache.host_kernel_path,
        cache.kernel_lib_path,
        cache.params_path,
    }
    assert set(manifest["files"]) == expected_files
    lib_bytes = (cache_path / cache.kernel_lib_path).read_bytes()
    assert manifest["files"][cache.kernel_lib_path]["size"] == len(lib_bytes)
    assert manifest["files"][cache.kernel_lib_path]["sha256"] == sha256(lib_bytes).hexdigest()


def test_load_removes_entry_with_truncated_kernel_lib(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "truncated-lib"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_cache_path(key))

    lib_file = cache_path / cache.kernel_lib_path
    lib_file.write_bytes(lib_file.read_bytes()[: len(b"fake-so-contents") // 2])

    assert _load_expecting_no_build(cache, key, monkeypatch) is None
    assert not cache_path.exists()


def test_load_removes_entry_with_same_size_corruption(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "bitflipped-lib"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_cache_path(key))

    lib_file = cache_path / cache.kernel_lib_path
    original = lib_file.read_bytes()
    lib_file.write_bytes(b"\x00" * len(original))

    assert _load_expecting_no_build(cache, key, monkeypatch) is None
    assert not cache_path.exists()


def test_hash_check_disabled_still_catches_truncation(cache_dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(env, "TILELANG_CACHE_VERIFY_HASH", "0")
    cache = KernelCache()
    key = "size-only-check"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_cache_path(key))

    lib_file = cache_path / cache.kernel_lib_path
    lib_file.write_bytes(b"short")

    assert _load_expecting_no_build(cache, key, monkeypatch) is None
    assert not cache_path.exists()


def test_source_files_are_size_checked_but_not_hashed(cache_dirs, tmp_path, monkeypatch):
    # Hashing sources on load would regress lazy source loading; only their
    # size is checked, so a same-size rewrite of a source file must not
    # invalidate the entry.
    cache = KernelCache()
    key = "source-not-hashed"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_cache_path(key))

    device_file = cache_path / cache.device_kernel_path
    device_file.write_text("X" * len(device_file.read_text()))

    assert _load_expecting_hit(cache, key, monkeypatch)


def test_legacy_entry_without_manifest_misses_and_save_repairs(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "legacy-no-manifest"
    cache_path = Path(cache._get_cache_path(key))
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device kernel")
    (cache_path / cache.host_kernel_path).write_text("// host kernel")
    (cache_path / cache.kernel_lib_path).write_bytes(b"legacy-truncated")
    with (cache_path / cache.params_path).open("wb") as f:
        cloudpickle.dump(["param"], f)

    assert _load_expecting_no_build(cache, key, monkeypatch) is None

    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    assert (cache_path / cache.manifest_path).exists()
    assert (cache_path / cache.kernel_lib_path).read_bytes() == b"fake-so-contents"
    assert _load_expecting_hit(cache, key, monkeypatch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
