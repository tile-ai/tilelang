import errno
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

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
    """Replace an incomplete NVRTC entry with a complete launcher and binary."""
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


def _save_in_process(workdir, path, writer, cleanup_entered, release_cleanup, staged, finished, observations):
    """Run the real save path with fake artifacts and a controlled cleanup pause."""
    try:
        workdir = Path(workdir)
        workdir.mkdir()
        result = _make_result(workdir)
        result.config["writer"] = writer
        original_remove = autotune_param.shutil.rmtree
        original_save = AutotuneResult._save_kernel_to_disk

        def remove(directory, *args, **kwargs):
            """Pause only A's removal of the stale public directory."""
            if writer == "A" and Path(directory) == Path(path):
                cleanup_entered.set()
                if not release_cleanup.wait(20):
                    raise TimeoutError("cleanup was not released")
            return original_remove(directory, *args, **kwargs)

        def save_kernel(self, *args, **kwargs):
            """Signal when this process has prepared its private artifacts."""
            original_save(self, *args, **kwargs)
            staged.set()

        with patch.object(autotune_param.shutil, "rmtree", remove), patch.object(AutotuneResult, "_save_kernel_to_disk", save_kernel):
            result.save_to_disk(Path(path))
        with open(Path(path) / BEST_CONFIG_PATH) as config_file:
            observations.put({"writer": writer, "observed": json.load(config_file)["writer"]})
    except Exception as error:
        observations.put({"writer": writer, "error": repr(error)})
    finally:
        finished.set()


def test_autotune_repair_preserves_first_published_result(cache_dirs, tmp_path):
    """A delayed stale-entry cleanup must preserve another process's publication."""
    path = cache_dirs / "test-namespace" / "autotuner" / "concurrent-repair"
    path.mkdir(parents=True)
    (path / "stale.txt").write_text("incomplete")
    context = multiprocessing.get_context("spawn")
    cleanup_entered, release_cleanup = context.Event(), context.Event()
    staged_a, staged_b = context.Event(), context.Event()
    finished_a, finished_b = context.Event(), context.Event()
    observations = context.Queue()
    workers = [
        context.Process(
            target=_save_in_process,
            args=(str(tmp_path / "writer-a"), str(path), "A", cleanup_entered, release_cleanup, staged_a, finished_a, observations),
        ),
        context.Process(
            target=_save_in_process,
            args=(str(tmp_path / "writer-b"), str(path), "B", cleanup_entered, release_cleanup, staged_b, finished_b, observations),
        ),
    ]
    try:
        workers[0].start()
        assert cleanup_entered.wait(20), "writer A did not reach stale-entry cleanup"
        workers[1].start()
        assert staged_b.wait(20), "writer B did not finish preparing its artifacts"
        # Without publication locking B can finish while A's obsolete removal is
        # paused. With locking B must wait until A is allowed to publish.
        finished_b.wait(2)
        release_cleanup.set()
        for worker in workers:
            worker.join(20)
            assert worker.exitcode == 0
        records = [observations.get(timeout=5), observations.get(timeout=5)]
        assert all("error" not in record for record in records), records
        with open(path / BEST_CONFIG_PATH) as config_file:
            published_writer = json.load(config_file)["writer"]
        assert all(record["observed"] == published_writer for record in records), records
        assert AutotuneResult._is_complete_result_dir(path, "cython")
        staging_root = path.parent.parent / ".staging"
        assert not any(staging_root.iterdir())
    finally:
        release_cleanup.set()
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(5)
        observations.close()
        observations.join_thread()


def test_autotune_lock_failure_preserves_existing_entry(cache_dirs, tmp_path, monkeypatch):
    """A failed lock acquisition must leave existing data and no private staging."""
    result = _make_result(tmp_path)
    path = cache_dirs / "test-namespace" / "autotuner" / "lock-failure"
    path.mkdir(parents=True)
    (path / "stale.txt").write_text("existing entry")
    logged = []

    def fail_lock(*args, **kwargs):
        """Inject a lock-acquisition failure without delaying the test."""
        raise TimeoutError("lock unavailable")

    monkeypatch.setattr(autotune_param, "FileLock", fail_lock)
    monkeypatch.setattr(autotune_param.logger, "exception", lambda message: logged.append(message))
    result.save_to_disk(path)

    assert (path / "stale.txt").read_text() == "existing entry"
    assert not (path / BEST_CONFIG_PATH).exists()
    assert logged == ["Error during atomic autotune result save"]
    assert not any((path.parent.parent / ".staging").iterdir())
