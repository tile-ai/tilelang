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


def test_autotune_save_preserves_incomplete_cache_dir(cache_dirs, tmp_path):
    result = _make_result(tmp_path)
    path = cache_dirs / "test-namespace" / "autotuner" / "autotune-entry"
    path.mkdir(parents=True)
    (path / "stale.txt").write_text("partial")

    result.save_to_disk(path)

    assert [p.name for p in path.iterdir()] == ["stale.txt"]
    assert (path / "stale.txt").read_text() == "partial"


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


def test_autotune_save_preserves_nvrtc_dir_missing_launcher(cache_dirs, tmp_path):
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

    assert not (path / KERNEL_PY_PATH).exists()
    assert (path / "legacy.txt").exists()


@pytest.mark.parametrize("invalid", ["absent", "missing_metadata", "corrupt_metadata", "missing_binary", "load_failure"])
def test_autotune_cache_multiple_roots(tmp_path, monkeypatch, invalid):
    import os
    from pathlib import Path
    from dataclasses import replace
    from types import SimpleNamespace
    from tilelang.autotuner import AutoTuner
    from tilelang.autotuner.param import CompileArgs

    roots = [tmp_path / name for name in ("local", "shared", "other")]
    tuner = AutoTuner(_fake_func, configs=[])
    tuner.compile_args = CompileArgs(target="cuda", execution_backend="cython")
    result = _make_result(tmp_path)
    paths = []
    for index, root in enumerate(roots):
        monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(root))
        result = replace(result, config={"threads": 128 * (index + 1)})
        tuner._save_result_to_disk("key", result)
        paths.append(tuner.write_cache_dir / "key")
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", os.pathsep.join(map(str, roots)))
    fail_load = False

    def from_database(cls, **kwargs):
        if fail_load and not Path(kwargs["kernel_lib_path"]).is_relative_to(roots[2]):
            raise RuntimeError("unloadable kernel")
        return SimpleNamespace(update_tuner_result=lambda **kwargs: None, get_kernel_source=lambda: "// device kernel")

    monkeypatch.setattr(autotune_param.JITKernel, "from_database", classmethod(from_database))
    assert tuner._load_result_from_disk("key").config == {"threads": 128}
    for path in paths[:2]:
        if invalid == "absent":
            path.rename(path.with_name("unused"))
        elif invalid == "missing_metadata":
            (path / BEST_CONFIG_PATH).unlink()
        elif invalid == "corrupt_metadata":
            (path / BEST_CONFIG_PATH).write_text("invalid json")
        elif invalid == "missing_binary":
            (path / KERNEL_LIB_PATH).unlink()
        else:
            fail_load = True
    # Snapshot secondary roots after injecting failures: lookup must not mutate them.
    before = {p: p.read_bytes() for root in roots[1:] for p in root.rglob("*") if p.is_file()}
    if invalid == "absent":
        assert tuner._load_result_from_disk("key").config == {"threads": 384}
    else:
        with pytest.raises((OSError, ValueError, RuntimeError)):
            tuner._load_result_from_disk("key")
        assert paths[0].exists()
        import shutil

        shutil.rmtree(paths[0])
        with pytest.raises((OSError, ValueError, RuntimeError)):
            tuner._load_result_from_disk("key")
    assert tuner._load_result_from_disk("missing") is None
    assert before == {p: p.read_bytes() for root in roots[1:] for p in root.rglob("*") if p.is_file()}
    tuner._save_result_to_disk("new", result)
    assert (tuner.write_cache_dir / "new" / BEST_CONFIG_PATH).is_file()
    assert not (paths[1].parent / "new").exists()
    assert not (paths[2].parent / "new").exists()


@pytest.mark.parametrize("invalid", ["metadata", "params", "load_failure"])
def test_autotune_cache_requires_whole_entry_removal(cache_dirs, tmp_path, monkeypatch, invalid):
    from types import SimpleNamespace
    from tilelang.autotuner import AutoTuner
    from tilelang.autotuner.param import CompileArgs

    tuner = AutoTuner(_fake_func, configs=[])
    tuner.compile_args = CompileArgs(target="cuda", execution_backend="cython")
    result = _make_result(tmp_path)
    tuner._save_result_to_disk("key", result)
    path = tuner.write_cache_dir / "key"
    fail_load = invalid == "load_failure"

    def from_database(cls, **kwargs):
        if fail_load:
            raise RuntimeError("unloadable kernel")
        return SimpleNamespace(update_tuner_result=lambda **kwargs: None, get_kernel_source=lambda: "// device kernel")

    monkeypatch.setattr(autotune_param.JITKernel, "from_database", classmethod(from_database))
    if invalid != "load_failure":
        (path / (BEST_CONFIG_PATH if invalid == "metadata" else PARAMS_PATH)).write_text("invalid json")
    with pytest.raises((ValueError, RuntimeError)):
        tuner._load_result_from_disk("key")
    assert path.is_dir()
    import shutil

    shutil.rmtree(path)
    fail_load = False
    tuner._save_result_to_disk("key", result)
    assert tuner._load_result_from_disk("key").config == result.config
    assert path.is_dir()


@pytest.mark.parametrize("invalid", [False, True])
def test_autotune_cache_read_preserves_concurrent_publication(cache_dirs, tmp_path, monkeypatch, invalid):
    from types import SimpleNamespace
    from tilelang.autotuner import AutoTuner
    from tilelang.autotuner.param import CompileArgs

    tuner = AutoTuner(_fake_func, configs=[])
    tuner.compile_args = CompileArgs(target="cuda", execution_backend="cython")
    monkeypatch.setattr(
        autotune_param.JITKernel,
        "from_database",
        classmethod(
            lambda cls, **kwargs: SimpleNamespace(update_tuner_result=lambda **kwargs: None, get_kernel_source=lambda: "// kernel")
        ),
    )
    result = _make_result(tmp_path)
    path = tuner.write_cache_dir / "key"
    original_load = AutotuneResult.load_from_disk
    if invalid:
        result.save_to_disk(path)
        (path / BEST_CONFIG_PATH).write_text("invalid json")

    def load_then_publish(cls, path, compile_args):
        import shutil

        try:
            loaded = original_load(path, compile_args)
        except ValueError:
            # Another process replaces the entry after this reader's load fails.
            shutil.rmtree(path)
            result.save_to_disk(path)
            raise
        assert loaded is None
        result.save_to_disk(path)
        return loaded

    monkeypatch.setattr(AutotuneResult, "load_from_disk", classmethod(load_then_publish))
    if invalid:
        with pytest.raises(ValueError):
            tuner._load_result_from_disk("key")
    else:
        assert tuner._load_result_from_disk("key") is None
    assert original_load(path, tuner.compile_args).config == result.config
