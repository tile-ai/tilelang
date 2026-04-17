import errno
import pytest

from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache


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
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    return cache_dir


def _make_fake_kernel(tmp_path):
    lib_path = tmp_path / "kernel_lib.so"
    lib_path.write_bytes(b"fake-so")
    return _FakeKernel(str(lib_path))


def _make_fake_nvrtc_kernel(tmp_path):
    lib_path = tmp_path / "kernel.cubin"
    lib_path.write_bytes(b"fake-cubin")
    lib_path.with_suffix(".py").write_text("# fake launcher")
    return _FakeKernel(str(lib_path))


def test_kernel_cache_rewrites_incomplete_cache_dir(cache_dirs, tmp_path):
    cache = KernelCache()
    key = "atomic-repair"
    cache_path = cache_dirs / key
    cache_path.mkdir()
    (cache_path / "stale.txt").write_text("partial")

    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))

    assert (cache_path / cache.device_kernel_path).exists()
    assert (cache_path / cache.host_kernel_path).exists()
    assert (cache_path / cache.kernel_lib_path).exists()
    assert (cache_path / cache.params_path).exists()
    assert not (cache_path / "stale.txt").exists()


def test_kernel_cache_logs_write_oserror_instead_of_treating_it_as_race(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "atomic-write-error"
    logged = []

    def raise_write_error(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(cache, "_save_so_cubin_to_disk", raise_write_error)
    monkeypatch.setattr(cache.logger, "exception", record_exception)

    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))

    assert f"{key}" not in {path.name for path in cache_dirs.iterdir()}
    assert "Error during atomic cache save" in logged
    assert not any(path.name.startswith(".staging_") for path in cache_dirs.iterdir())


def test_kernel_cache_does_not_publish_incomplete_dir_when_device_source_is_missing(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "atomic-missing-device-source"
    kernel = _make_fake_kernel(tmp_path)
    kernel.kernel_source = None
    logged = []

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(cache.logger, "exception", record_exception)

    cache._save_kernel_to_disk(key, kernel)

    assert f"{key}" not in {path.name for path in cache_dirs.iterdir()}
    assert "Error during atomic cache save" in logged
    assert not any(path.name.startswith(".staging_") for path in cache_dirs.iterdir())


def test_nvrtc_kernel_cache_rewrites_dir_missing_launcher(cache_dirs, tmp_path):
    cache = NVRTCKernelCache()
    key = "nvrtc-atomic-repair"
    cache_path = cache_dirs / key
    cache_path.mkdir()
    (cache_path / cache.device_kernel_path).write_text("// device kernel")
    (cache_path / cache.host_kernel_path).write_text("// host kernel")
    (cache_path / cache.kernel_lib_path).write_bytes(b"old-cubin")
    (cache_path / cache.params_path).write_bytes(b"old-params")
    (cache_path / "legacy.txt").write_text("stale")

    cache._save_kernel_to_disk(key, _make_fake_nvrtc_kernel(tmp_path))

    assert (cache_path / cache.kernel_py_path).exists()
    assert not (cache_path / "legacy.txt").exists()
