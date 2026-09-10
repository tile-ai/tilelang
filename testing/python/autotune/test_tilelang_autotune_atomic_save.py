import errno

import pytest

import tilelang.testing
from tilelang.autotuner import param as autotune_param
from tilelang.autotuner.param import (
    AutotuneResult,
    CompileArgs,
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
from tilelang.engine.param import KernelParam
from tilelang.backend import create_backend_context
from tilelang.env import env
from tilelang import tvm


class _FakeAdapter:
    def __init__(self, libpath: str | None):
        self.libpath = libpath
        self.tileir_artifact = object()

    @staticmethod
    def _serialize_tileir_artifact(artifact):
        assert artifact is not None
        return b"fake-tileir-artifact"

    def get_kernel_source(self):
        return "// wrapped kernel"

    def get_host_source(self):
        return "// host kernel"


class _FakeKernel:
    def __init__(self, libpath: str | None, execution_backend: str = "cython"):
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
    if execution_backend == "nvrtc":
        lib_path = tmp_path / "kernel.cubin"
        lib_path.write_bytes(b"fake-cubin")
        lib_path.with_suffix(".py").write_text("# fake launcher")
    elif execution_backend == "tileir":
        lib_path = None
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
        kernel=_FakeKernel(str(lib_path) if lib_path is not None else None, execution_backend=execution_backend),
    )


def test_compile_args_forwards_execution_backend(monkeypatch):
    captured = {}

    def fake_compile(program, **kwargs):
        captured["program"] = program
        captured.update(kwargs)
        return "compiled"

    monkeypatch.setattr(autotune_param.tilelang, "compile", fake_compile)

    program = object()
    result = CompileArgs(out_idx=[-1], execution_backend="tileir", target="cuda").compile_program(program)

    assert result == "compiled"
    assert captured["program"] is program
    assert captured["execution_backend"] == "tileir"
    assert captured["target"] == "cuda"
    assert captured["out_idx"] == [-1]


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


def test_autotune_save_rewrites_nvrtc_dir_missing_launcher(cache_dirs, tmp_path):
    result = _make_result(tmp_path, execution_backend="nvrtc")
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-nvrtc-entry"
    path.mkdir(parents=True)
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


def test_autotune_save_tileir_uses_tileir_artifact_file(cache_dirs, tmp_path):
    result = _make_result(tmp_path, execution_backend="tileir")
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-tileir-entry"

    result.save_to_disk(path)

    assert AutotuneResult._get_kernel_lib_file("tileir") == "kernel.tileir.json"
    assert (path / "kernel.tileir.json").read_bytes() == b"fake-tileir-artifact"
    assert (path / HOST_KERNEL_PATH).read_text() == "// host kernel"
    assert not (path / KERNEL_CUBIN_PATH).exists()
    assert not (path / KERNEL_LIB_PATH).exists()


@pytest.mark.parametrize(
    "reload_error",
    [
        "Unsupported TileIR cache artifact version 1",
        "TileIR cache artifact compatibility does not match the active target/toolchain",
    ],
)
@tilelang.testing.requires_cuda
def test_autotune_tileir_reload_error_is_treated_as_cache_miss(cache_dirs, tmp_path, monkeypatch, reload_error):
    result = _make_result(tmp_path, execution_backend="tileir")
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-tileir-stale"
    result.save_to_disk(path)

    def reject_stale_cache(**kwargs):
        del kwargs
        raise ValueError(reload_error)

    monkeypatch.setattr(autotune_param.JITKernel, "from_database", reject_stale_cache)

    loaded = result._load_kernel_from_disk(
        path,
        backend_context=create_backend_context("tileir -arch=sm_120", "c", "tileir"),
        func=_fake_func,
    )

    assert loaded is None
    assert not path.exists()
