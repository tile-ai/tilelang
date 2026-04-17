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
    KERNEL_CUBIN_PATH,
    KERNEL_LIB_PATH,
    KERNEL_PY_PATH,
    PARAMS_PATH,
)
from tilelang.env import env


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
        self.params = ["param"]


def _fake_func():
    return None


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    return cache_dir


def _make_result(tmp_path, execution_backend: str = "cython"):
    if execution_backend == "nvrtc":
        lib_path = tmp_path / "kernel.cubin"
        lib_path.write_bytes(b"fake-cubin")
        lib_path.with_suffix(".py").write_text("# fake launcher")
    else:
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
    path = cache_dirs / "autotune-entry"
    path.mkdir()
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
    path = cache_dirs / "autotune-error"
    logged = []

    def raise_write_error(self, *args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(AutotuneResult, "_save_kernel_to_disk", raise_write_error)
    monkeypatch.setattr(autotune_param.logger, "exception", record_exception)

    result.save_to_disk(path)

    assert not path.exists()
    assert "Error during atomic autotune result save" in logged
    assert not any(child.name.startswith(".staging_") for child in cache_dirs.iterdir())


def test_autotune_save_does_not_publish_incomplete_dir_when_device_source_is_missing(cache_dirs, tmp_path, monkeypatch):
    result = _make_result(tmp_path)
    result.kernel.kernel_source = None
    path = cache_dirs / "autotune-missing-device-source"
    logged = []

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(autotune_param.logger, "exception", record_exception)

    result.save_to_disk(path)

    assert not path.exists()
    assert "Error during atomic autotune result save" in logged
    assert not any(child.name.startswith(".staging_") for child in cache_dirs.iterdir())


def test_autotune_save_rewrites_nvrtc_dir_missing_launcher(cache_dirs, tmp_path):
    result = _make_result(tmp_path, execution_backend="nvrtc")
    path = cache_dirs / "autotune-nvrtc-entry"
    path.mkdir()
    (path / BEST_CONFIG_PATH).write_text("{}")
    (path / FUNCTION_PATH).write_bytes(b"old-func")
    (path / LATENCY_PATH).write_text('{"latency": 1.0, "ref_latency": 2.0}')
    (path / DEVICE_KERNEL_PATH).write_text("// device kernel")
    (path / HOST_KERNEL_PATH).write_text("// host kernel")
    (path / KERNEL_CUBIN_PATH).write_bytes(b"old-cubin")
    (path / PARAMS_PATH).write_bytes(b"old-params")
    (path / "legacy.txt").write_text("stale")

    result.save_to_disk(path)

    assert (path / KERNEL_PY_PATH).exists()
    assert not (path / "legacy.txt").exists()
