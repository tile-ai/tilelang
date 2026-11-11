"""Regression tests for cache-entry integrity verification.

A crashed writer (or crashed network-filesystem client) can publish a cache
entry whose kernel_lib.so is truncated or partially written. Such a binary can
dlopen fine and only fail at first kernel launch with CUDA_ERROR_INVALID_IMAGE,
where nothing repairs the entry. The kernel cache therefore records a manifest
of file sizes/hashes on save and verifies it on load, raising on mismatches without modifying published entries.
"""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from tilelang import tvm

import tilelang.cache.kernel_cache as kernel_cache_mod
from tilelang.backend import create_backend_context
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam, dump_kernel_params
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
        self.params = [KernelParam(tvm.DataType("float32"), [4])]


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
        raise AssertionError("corrupted cache entries must raise before rebuilding from database")

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

    cache_path = Path(cache._get_write_cache_path(key))
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


def test_load_preserves_entry_with_truncated_kernel_lib(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "truncated-lib"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_write_cache_path(key))

    lib_file = cache_path / cache.kernel_lib_path
    lib_file.write_bytes(lib_file.read_bytes()[: len(b"fake-so-contents") // 2])

    with pytest.raises(ValueError):
        _load_expecting_no_build(cache, key, monkeypatch)
    assert cache_path.exists()


def test_load_preserves_entry_with_same_size_corruption(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "bitflipped-lib"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_write_cache_path(key))

    lib_file = cache_path / cache.kernel_lib_path
    original = lib_file.read_bytes()
    lib_file.write_bytes(b"\x00" * len(original))

    with pytest.raises(ValueError):
        _load_expecting_no_build(cache, key, monkeypatch)
    assert cache_path.exists()


def test_source_files_are_size_checked_but_not_hashed(cache_dirs, tmp_path, monkeypatch):
    # Hashing sources on load would regress lazy source loading; only their
    # size is checked, so a same-size rewrite of a source file must not
    # invalidate the entry.
    cache = KernelCache()
    key = "source-not-hashed"
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    cache_path = Path(cache._get_write_cache_path(key))

    device_file = cache_path / cache.device_kernel_path
    device_file.write_text("X" * len(device_file.read_text()))

    assert _load_expecting_hit(cache, key, monkeypatch)


def test_legacy_entry_requires_whole_entry_removal(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "legacy-no-manifest"
    cache_path = Path(cache._get_write_cache_path(key))
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device kernel")
    (cache_path / cache.host_kernel_path).write_text("// host kernel")
    (cache_path / cache.kernel_lib_path).write_bytes(b"legacy-truncated")
    (cache_path / cache.params_path).write_text(dump_kernel_params([KernelParam(tvm.DataType("float32"), [4])]))

    with pytest.raises(ValueError):
        _load_expecting_no_build(cache, key, monkeypatch)

    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    assert not (cache_path / cache.manifest_path).exists()
    import shutil

    shutil.rmtree(cache_path)
    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))
    assert (cache_path / cache.manifest_path).exists()
    assert (cache_path / cache.kernel_lib_path).read_bytes() == b"fake-so-contents"
    assert _load_expecting_hit(cache, key, monkeypatch)


@pytest.mark.parametrize("invalid", ["missing", "corrupt", "load_failure"])
def test_kernel_cache_multiple_roots(tmp_path, monkeypatch, invalid):
    import os
    from types import SimpleNamespace
    from tilelang.autotuner.tuner import AutoTuner

    roots = [tmp_path / name for name in ("local", "shared", "other")]
    cache = KernelCache()
    kernel = _make_fake_kernel(tmp_path)
    for root in roots:
        monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(root))
        cache._save_kernel_to_disk("key", kernel)
    relative = Path(cache._get_write_cache_path("key")).relative_to(roots[-1])
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", os.pathsep.join(map(str, roots)))
    assert AutoTuner._get_cache_dir(env.get_cache_write_dir()).is_relative_to(roots[0])

    def build(cls, **kwargs):
        path = Path(kwargs["kernel_lib_path"])
        if invalid == "load_failure" and not path.is_relative_to(roots[2]):
            raise RuntimeError("unloadable entry")
        return SimpleNamespace(loaded_path=path)

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(build))
    context = create_backend_context("cuda", execution_backend="tvm_ffi")
    if invalid != "load_failure":
        assert cache._load_kernel_from_disk("key", context).loaded_path.is_relative_to(roots[0])
        for root in roots[:2]:
            binary = root / relative / cache.kernel_lib_path
            if invalid == "missing":
                binary.unlink()
            else:
                binary.write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError)):
        cache._load_kernel_from_disk("key", context)
    assert (roots[0] / relative).is_dir()
    assert (roots[1] / relative).is_dir()
    import shutil

    shutil.rmtree(roots[0] / relative)
    with pytest.raises((ValueError, RuntimeError)):
        cache._load_kernel_from_disk("key", context)
    shutil.rmtree(roots[1] / relative)
    loaded = cache._load_kernel_from_disk("key", context)
    assert loaded.loaded_path.is_relative_to(roots[2])
    assert loaded._tilelang_cache_key == "key"
    assert Path(loaded._tilelang_cache_path) == roots[2] / relative
    cache._save_kernel_to_disk("new", kernel)
    new_relative = Path(cache._get_write_cache_path("new")).relative_to(roots[0])
    assert (roots[0] / new_relative).is_dir()
    assert not (roots[1] / new_relative).exists()
    cache.clear_cache()
    assert not (roots[0] / relative).exists()
    assert (roots[2] / relative).is_dir()


@pytest.mark.parametrize("backend", ["cutedsl", "nvrtc"])
def test_cached_module_import_does_not_write_bytecode(tmp_path, monkeypatch, backend):
    import sys

    if backend == "cutedsl":
        from tilelang.jit.adapter.cutedsl.libgen import CuTeDSLLibraryGenerator as generator
    else:
        from tilelang.jit.adapter.nvrtc import is_nvrtc_available, NVRTC_UNAVAILABLE_MESSAGE

        if not is_nvrtc_available:
            pytest.skip(NVRTC_UNAVAILABLE_MESSAGE)
        from tilelang.jit.adapter.nvrtc.libgen import NVRTCLibraryGenerator as generator

    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    path = tmp_path / "kernel.py"
    path.write_text("value = 42\n")
    module = generator.import_from_file("cached_kernel", str(path))
    assert module.value == 42
    assert list(tmp_path.iterdir()) == [path]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
