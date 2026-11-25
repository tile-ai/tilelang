"""Regression tests for shared cache entry permissions (CPU-only, no GPU)."""

import contextlib
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


@contextlib.contextmanager
def _umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point TILELANG_CACHE_DIR at a fresh root; ``shared`` toggles the SGID bit."""

    def make(shared: bool = True, mode: int = 0o2755, name: str = "cache") -> Path:
        root = tmp_path / name
        root.mkdir()
        if shared:
            root.chmod(mode)
            # Skip (instead of failing) on filesystems without POSIX mode/SGID
            # support, e.g. vfat or some network mounts.
            if _mode(root) != mode or not os.stat(root).st_mode & stat.S_ISGID:
                pytest.skip("filesystem does not support POSIX modes with SGID")
        monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(root))
        return root

    return make


def _save(cache: KernelCache, key: str, tmp_path: Path) -> None:
    lib = tmp_path / f"{key}.so"
    lib.write_bytes(b"fake-so")
    with _umask(0o077):
        cache._save_kernel_to_disk(key, _FakeKernel(str(lib)))


@pytest.mark.parametrize("root_mode", [0o2755, 0o2775], ids=["2755", "2775"])
def test_shared_root_publishes_group_accessible_entry(cache_root, tmp_path, monkeypatch, root_mode):
    root = cache_root(shared=True, mode=root_mode)
    with _umask(0o077):
        cache = KernelCache()
        KernelCache._create_dirs()
        _save(cache, "k", tmp_path)

    namespace_root = Path(cache._get_namespace_root(str(root)))
    for directory in (
        namespace_root.parent,
        namespace_root,
        Path(cache._get_write_cache_root()),
        Path(cache._get_staging_root()),
    ):
        assert _mode(directory) == root_mode
        assert os.stat(directory).st_mode & stat.S_ISGID

    # Existing intermediate directories must not be re-chmodded by later writers
    # (which would fail with EPERM when owned by another group member).
    chmodded = []
    real_chmod = os.chmod
    monkeypatch.setattr(os, "chmod", lambda path, mode, *a, **kw: (chmodded.append(Path(path)), real_chmod(path, mode, *a, **kw))[1])
    KernelCache._create_dirs()
    assert chmodded == []
    monkeypatch.setattr(os, "chmod", real_chmod)

    entry = Path(cache._get_write_cache_path("k"))
    assert _mode(entry) == 0o2755
    assert os.stat(entry).st_mode & stat.S_ISGID
    for name in (
        cache.device_kernel_path,
        cache.host_kernel_path,
        cache.kernel_lib_path,
        cache.params_path,
        cache.manifest_path,
    ):
        file_path = entry / name
        assert _mode(file_path) == 0o644
        assert not os.stat(file_path).st_mode & stat.S_ISGID

    staged_exec = tmp_path / "staged_exec"
    staged_exec.mkdir()
    exec_file = staged_exec / "executable.so"
    with _umask(0o077):
        exec_file.write_bytes(b"fake-exec")
        exec_file.chmod(0o700)
        KernelCache._legalize_staged_entry(str(staged_exec))
    assert _mode(exec_file) == 0o755


def test_shared_multi_root_uses_write_dir(cache_root, tmp_path, monkeypatch):
    shared_write = cache_root(shared=True, mode=0o2775, name="shared_write")
    readonly_fallback = tmp_path / "readonly_fallback"
    readonly_fallback.mkdir(mode=0o700)
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", f"{shared_write}{os.pathsep}{readonly_fallback}")

    cache = KernelCache()
    _save(cache, "multi", tmp_path)

    entry = Path(cache._get_write_cache_path("multi"))
    assert entry.is_relative_to(shared_write)
    assert _mode(entry) == 0o2755
    assert _mode(entry / cache.kernel_lib_path) == 0o644
    assert list(readonly_fallback.iterdir()) == []


def test_private_root_keeps_umask_permissions(cache_root, tmp_path):
    probe = tmp_path / "umask_probe"
    with _umask(0o077):
        probe.mkdir()
    if _mode(probe) != 0o700:
        pytest.skip("filesystem does not honor umask")

    root = cache_root(shared=False)
    with _umask(0o077):
        cache = KernelCache()
        KernelCache._create_dirs()
        _save(cache, "k", tmp_path)

    namespace_root = Path(cache._get_namespace_root(str(root)))
    assert _mode(namespace_root) == 0o700
    entry = Path(cache._get_write_cache_path("k"))
    assert _mode(entry) == 0o700
    assert _mode(entry / cache.kernel_lib_path) == 0o600
    assert not os.stat(entry).st_mode & stat.S_ISGID
