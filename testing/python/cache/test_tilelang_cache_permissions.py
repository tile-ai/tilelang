"""Regression tests for shared cache entry permissions (CPU-only, no GPU)."""

import os
import stat
from pathlib import Path

import pytest

from tilelang import env, tvm
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX permission bits")


class _FakeKernel:
    def __init__(self, libpath):
        self.adapter = type("_Adapter", (), {"libpath": libpath, "get_kernel_source": lambda self: "// host"})()
        self.kernel_source = "// device kernel"
        self.params = [KernelParam(tvm.DataType("float32"), [4])]


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point TILELANG_CACHE_DIR at a fresh root; ``shared`` toggles the SGID bit."""

    def make(shared: bool = True) -> Path:
        root = tmp_path / "cache"
        root.mkdir()
        if shared:
            root.chmod(0o2755)
            # Skip (instead of failing) on filesystems without POSIX mode/SGID
            # support, e.g. vfat or some network mounts.
            if _mode(root) != 0o2755 or not os.stat(root).st_mode & stat.S_ISGID:
                pytest.skip("filesystem does not support POSIX modes with SGID")
        monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(root))
        return root

    return make


def _save(cache: KernelCache, key: str, tmp_path: Path) -> None:
    lib = tmp_path / f"{key}.so"
    lib.write_bytes(b"fake-so")
    previous = os.umask(0o077)
    try:
        cache._save_kernel_to_disk(key, _FakeKernel(str(lib)))
    finally:
        os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_shared_root_publishes_group_accessible_entry(cache_root, tmp_path):
    cache_root(shared=True)
    cache = KernelCache()
    _save(cache, "k", tmp_path)

    entry = Path(cache._get_cache_path("k"))
    assert _mode(entry) == 0o2755
    assert os.stat(entry).st_mode & stat.S_ISGID
    for name in (cache.device_kernel_path, cache.host_kernel_path, cache.kernel_lib_path, cache.params_path):
        file_path = entry / name
        assert _mode(file_path) == 0o644
        assert not os.stat(file_path).st_mode & stat.S_ISGID


def test_private_root_keeps_umask_permissions(cache_root, tmp_path):
    probe = tmp_path / "umask_probe"
    previous = os.umask(0o077)
    try:
        probe.mkdir()
    finally:
        os.umask(previous)
    if _mode(probe) != 0o700:
        pytest.skip("filesystem does not honor umask")

    cache_root(shared=False)
    cache = KernelCache()
    _save(cache, "k", tmp_path)

    entry = Path(cache._get_cache_path("k"))
    assert _mode(entry) == 0o700
    assert _mode(entry / cache.kernel_lib_path) == 0o600
    assert not os.stat(entry).st_mode & stat.S_ISGID
