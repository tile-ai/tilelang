import logging
import subprocess

import tilelang.testing
import pytest


@pytest.fixture
def capture_tilelang_logs():
    import tilelang  # noqa: F401

    tilelang_logger = logging.getLogger("tilelang")
    old_propagate = tilelang_logger.propagate
    tilelang_logger.propagate = True
    try:
        yield
    finally:
        tilelang_logger.propagate = old_propagate


class _FakeProcess:
    returncode = 0

    def __init__(self):
        self.killed = False

    def communicate(self, timeout=None):
        if timeout == 0.25:
            raise subprocess.TimeoutExpired(["nvcc"], timeout)
        return b"", None

    def kill(self):
        self.killed = True


class _HangingProcess:
    returncode = 0

    def communicate(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(["nvcc"], timeout, output=b"simulated nvcc hang")
        return b"simulated nvcc hang cleaned up", None

    def kill(self):
        pass


class _ScriptableFunc:
    attrs = {"global_symbol": "cache_diag_kernel"}

    def script(self, show_meta=True):
        return "cache_diag_kernel"


def _make_cuda_compile_probe():
    from tilelang import language as T

    @T.prim_func
    def add_one(a: T.Tensor((128,), "float32"), b: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128) as _:
            i = T.get_thread_binding(0)
            b[i] = a[i] + 1.0

    return add_one


def test_jit_phase_logs_start_done_and_context(caplog, capture_tilelang_logs):
    from tilelang.jit.diagnostics import jit_phase

    caplog.set_level(logging.INFO, logger="tilelang.jit.diagnostics")

    with jit_phase("unit.phase", enabled=True, kernel="kernel_a", target="cuda -arch=sm_120"):
        pass

    messages = [record.getMessage() for record in caplog.records]
    assert any("TileLang JIT phase start: unit.phase" in message for message in messages)
    assert any("TileLang JIT phase done: unit.phase" in message for message in messages)
    assert any("kernel_a" in message and "sm_120" in message for message in messages)


def test_jit_phase_logs_failure_and_preserves_exception(caplog, capture_tilelang_logs):
    from tilelang.jit.diagnostics import jit_phase

    caplog.set_level(logging.INFO, logger="tilelang.jit.diagnostics")

    with pytest.raises(ValueError, match="boom"), jit_phase("unit.failure", enabled=True, backend="tvm_ffi"):
        raise ValueError("boom")

    messages = [record.getMessage() for record in caplog.records]
    assert any("TileLang JIT phase failed: unit.failure" in message for message in messages)
    assert any("backend='tvm_ffi'" in message or '"backend": "tvm_ffi"' in message for message in messages)


def test_jit_phase_is_silent_when_disabled(caplog, capture_tilelang_logs):
    from tilelang.jit.diagnostics import jit_phase

    caplog.set_level(logging.INFO, logger="tilelang.jit.diagnostics")

    with jit_phase("unit.silent", enabled=False):
        pass

    assert not caplog.records


def test_nvcc_compile_cuda_honors_tilelang_timeout(monkeypatch):
    from tilelang.contrib import nvcc

    fake_proc = _FakeProcess()

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr(nvcc.env, "TILELANG_CLEANUP_TEMP_FILES", "0")
    monkeypatch.setattr(nvcc, "get_nvcc_compiler", lambda: "nvcc")
    monkeypatch.setattr(nvcc, "get_target_compute_version", lambda target=None: "12.0")
    monkeypatch.setattr(nvcc, "get_target_arch", lambda compute_version: "120")
    monkeypatch.setattr(nvcc, "get_nvcc_subprocess_env", lambda: None)
    monkeypatch.setattr(nvcc.subprocess, "Popen", lambda *args, **kwargs: fake_proc)

    with pytest.raises(RuntimeError) as exc_info:
        nvcc.compile_cuda("__global__ void kernel() {}", target_format="ptx", verbose=False)

    message = str(exc_info.value)
    assert fake_proc.killed
    assert "timed out after 0.25 seconds" in message
    assert "Command:" in message
    assert "Source:" in message


def test_jit_compile_reports_timeout_for_hanging_nvcc(monkeypatch, tmp_path, caplog, capture_tilelang_logs):
    import tilelang
    from tilelang.contrib import nvcc
    from tilelang.env import env

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TILELANG_JIT_DIAGNOSTICS", "1")
    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setattr(nvcc.subprocess, "Popen", lambda *args, **kwargs: _HangingProcess())

    caplog.set_level(logging.INFO, logger="tilelang.jit.diagnostics")

    with pytest.raises(RuntimeError) as exc_info:
        tilelang.compile(
            _make_cuda_compile_probe(),
            out_idx=-1,
            target={"kind": "cuda", "arch": "sm_120"},
        )

    message = str(exc_info.value)
    assert "NVCC compilation timed out after 0.25 seconds" in message
    assert "Command:" in message
    assert "Source:" in message
    assert "Target:" in message

    messages = [record.getMessage() for record in caplog.records]
    assert any("TileLang JIT phase start: cache.compile" in message for message in messages)
    assert any("TileLang JIT phase start: lower" in message for message in messages)
    assert any("TileLang JIT phase failed: lower" in message for message in messages)
    assert any("TileLang JIT phase failed: cache.compile" in message for message in messages)


def test_compile_timeout_env_parser_accepts_empty_zero_and_positive(monkeypatch):
    from tilelang.contrib.nvcc import _get_compile_timeout_seconds

    monkeypatch.delenv("TILELANG_COMPILE_TIMEOUT_SECONDS", raising=False)
    assert _get_compile_timeout_seconds() is None

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "0")
    assert _get_compile_timeout_seconds() is None

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "1.5")
    assert _get_compile_timeout_seconds() == 1.5

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "bad")
    with pytest.raises(ValueError, match="TILELANG_COMPILE_TIMEOUT_SECONDS"):
        _get_compile_timeout_seconds()

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "nan")
    with pytest.raises(ValueError, match="TILELANG_COMPILE_TIMEOUT_SECONDS"):
        _get_compile_timeout_seconds()

    monkeypatch.setenv("TILELANG_COMPILE_TIMEOUT_SECONDS", "inf")
    with pytest.raises(ValueError, match="TILELANG_COMPILE_TIMEOUT_SECONDS"):
        _get_compile_timeout_seconds()


def test_kernel_cache_miss_compile_logs_context(monkeypatch, tmp_path, caplog, capture_tilelang_logs):
    import tilelang.cache.kernel_cache as kernel_cache_mod
    from tilelang.cache.kernel_cache import KernelCache
    from tilelang.env import env

    cache_dir = tmp_path / "cache"
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    monkeypatch.setenv("TILELANG_JIT_DIAGNOSTICS", "1")

    class _FakeKernel:
        params = []
        kernel_source = "// device"

        def __init__(self, *args, **kwargs):
            raise RuntimeError("compile failed")

    monkeypatch.setattr(kernel_cache_mod, "JITKernel", _FakeKernel)

    caplog.set_level(logging.INFO, logger="tilelang.jit.diagnostics")
    cache = KernelCache()

    with pytest.raises(RuntimeError, match="compile failed"):
        cache.cached(
            _ScriptableFunc(),
            out_idx=[],
            target="cuda -arch=sm_120",
            target_host=None,
            execution_backend="tvm_ffi",
            verbose=False,
            pass_configs=None,
            compile_flags=None,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("TileLang JIT phase start: cache.compile" in message for message in messages)
    assert any("TileLang JIT phase failed: cache.compile" in message for message in messages)
    assert any("cache_diag_kernel" in message and "sm_120" in message for message in messages)


def test_explicit_cuda_arch_source_generation_probe():
    import tilelang
    from tilelang import tvm
    from tilelang.engine.lower import lower as tilelang_lower

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_120"})
    tilelang.disable_cache()
    try:
        with target:
            artifact = tilelang_lower(
                _make_cuda_compile_probe(),
                target=target,
                enable_device_compile=False,
            )
    finally:
        tilelang.enable_cache()

    assert "__global__" in artifact.kernel_source
    assert "add_one" in artifact.kernel_source


@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_explicit_cuda_arch_runtime_probe_completes():
    import tilelang
    import torch

    kernel = tilelang.compile(
        _make_cuda_compile_probe(),
        out_idx=-1,
        target={"kind": "cuda", "arch": "sm_120"},
    )
    x = torch.ones(128, device="cuda", dtype=torch.float32)
    y = kernel(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
