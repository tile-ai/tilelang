import errno

import pytest

from tilelang.autotuner import param as autotune_param
from tilelang.autotuner.param import (
    AutotuneResult,
    BEST_CONFIG_PATH,
    FUNCTION_PATH,
    LATENCY_PATH,
    DEVICE_KERNEL_PATH,
    HOST_KERNEL_PATH,
    KERNEL_LIB_PATH,
    PARAMS_PATH,
)
from tilelang.engine.param import KernelParam
from tilelang.env import env
from tilelang import tvm


class _FakeAdapter:
    def __init__(self, libpath: str):
        self.libpath = libpath

    def get_kernel_source(self):
        return "// host kernel"

    def get_host_source(self):
        return "// host kernel"


class _FakeKernel:
    def __init__(self, libpath: str, execution_backend: str = "cython"):
        self.execution_backend = execution_backend
        self.adapter = _FakeAdapter(libpath)
        self.kernel_source = "// device kernel"
        self.params = [KernelParam(tvm.DataType("float32"), [4])]


def _fake_func():
    return None


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    return cache_dir


def _make_result(tmp_path, execution_backend: str = "cython"):
    lib_path = tmp_path / "kernel_lib.so"
    lib_path.write_bytes(b"fake-so")
    _fake_func.attrs = None
    return AutotuneResult(
        latency=1.0,
        config={"threads": 128},
        ref_latency=2.0,
        libcode="// libcode",
        func=_fake_func,
        kernel=_FakeKernel(str(lib_path), execution_backend=execution_backend),
    )


def test_autotune_save_rewrites_incomplete_cache_dir(cache_dirs, tmp_path):
    result = _make_result(tmp_path)
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-entry"
    path.mkdir(parents=True)
    (path / "stale.txt").write_text("partial")

    result.save_to_disk(path)

    for filename in (
        BEST_CONFIG_PATH,
        FUNCTION_PATH,
        LATENCY_PATH,
        DEVICE_KERNEL_PATH,
        HOST_KERNEL_PATH,
        KERNEL_LIB_PATH,
        PARAMS_PATH,
    ):
        assert (path / filename).exists()
    assert not (path / "stale.txt").exists()


def test_autotune_save_logs_write_oserror_instead_of_treating_it_as_race(cache_dirs, tmp_path, monkeypatch):
    result = _make_result(tmp_path)
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-error"
    logged = []
    staging_root = path.parent.parent / ".staging"

    def raise_write_error(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(AutotuneResult, "_save_kernel_to_disk", raise_write_error)
    monkeypatch.setattr(autotune_param.logger, "exception", record_exception)

    result.save_to_disk(path)

    assert not path.exists()
    assert "Error during atomic autotune result save" in logged
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_autotune_save_does_not_publish_incomplete_dir_when_device_source_is_missing(cache_dirs, tmp_path, monkeypatch):
    result = _make_result(tmp_path)
    result.kernel.kernel_source = None
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-missing-device-source"
    logged = []
    staging_root = path.parent.parent / ".staging"

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(autotune_param.logger, "exception", record_exception)

    result.save_to_disk(path)

    assert not path.exists()
    assert "Error during atomic autotune result save" in logged
    assert not staging_root.exists() or not any(staging_root.iterdir())
