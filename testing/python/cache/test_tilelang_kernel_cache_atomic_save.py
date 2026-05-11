import builtins
import errno
from pathlib import Path

import cloudpickle
import pytest

import tilelang.cache.kernel_cache as kernel_cache_mod
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


def _write_complete_kernel_cache_entry(
    cache: KernelCache,
    key: str,
    device_source: str = "// device kernel",
    host_source: str = "// host kernel",
    prim_func=None,
) -> Path:
    cache_path = Path(cache._get_cache_path(key))
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text(device_source)
    (cache_path / cache.host_kernel_path).write_text(host_source)
    (cache_path / cache.kernel_lib_path).write_bytes(b"fake-so")
    with (cache_path / cache.params_path).open("wb") as f:
        cloudpickle.dump(["param"], f)
    if prim_func is not None:
        with (cache_path / cache.prim_func_path).open("wb") as f:
            cloudpickle.dump(prim_func, f)
    return cache_path


def test_kernel_cache_disk_hit_defers_source_loading(cache_dirs, monkeypatch):
    cache = KernelCache()
    key = "lazy-source-load"
    cache_path = _write_complete_kernel_cache_entry(cache, key)

    sentinel = object()
    captured = {}

    def fail_source_load(*args, **kwargs):
        raise AssertionError("disk cache hit should pass source paths through for lazy loading")

    def fake_from_database(cls, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cache, "_load_kernel_source", fail_source_load)
    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fake_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        target="cuda",
        target_host=None,
        out_idx=[0],
        execution_backend="tvm_ffi",
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is sentinel
    assert captured["host_kernel_source"] is None
    assert captured["device_kernel_source"] is None
    assert captured["host_kernel_source_path"] == str(cache_path / cache.host_kernel_path)
    assert captured["device_kernel_source_path"] == str(cache_path / cache.device_kernel_path)
    assert captured["kernel_lib_path"] == str(cache_path / cache.kernel_lib_path)
    assert captured["params"] == ["param"]


def test_kernel_cache_disk_hit_perf_skips_large_source_file_reads(cache_dirs, monkeypatch):
    cache = KernelCache()
    key = "lazy-source-load-perf"
    large_source = "// source\n" + ("x" * (2 * 1024 * 1024))
    cache_path = _write_complete_kernel_cache_entry(
        cache,
        key,
        device_source=large_source,
        host_source=large_source,
    )
    source_paths = {
        (cache_path / cache.device_kernel_path).resolve(),
        (cache_path / cache.host_kernel_path).resolve(),
    }
    source_read_count = 0
    sentinel = object()

    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        nonlocal source_read_count
        mode = args[0] if args else kwargs.get("mode", "r")
        try:
            path = Path(file).resolve()
        except TypeError:
            return real_open(file, *args, **kwargs)
        if "r" in mode and path in source_paths:
            source_read_count += 1
            raise AssertionError("cache perf regression: source file read during disk cache hit")
        return real_open(file, *args, **kwargs)

    def fake_from_database(cls, **kwargs):
        return sentinel

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fake_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        target="cuda",
        target_host=None,
        out_idx=[0],
        execution_backend="tvm_ffi",
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is sentinel
    assert source_read_count == 0


def test_kernel_cache_frontend_hit_loads_serialized_prim_func(cache_dirs, monkeypatch):
    cache = KernelCache()
    key = "frontend-kernel-key"
    prim_func = {"name": "cached_prim_func"}
    cache_path = _write_complete_kernel_cache_entry(cache, key, prim_func=prim_func)
    cache.store_frontend_cache("frontend-key", key)

    sentinel = object()
    captured = {}

    def fake_from_database(cls, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fake_from_database))

    loaded = cache.load_frontend_cached(
        "frontend-key",
        target="cuda",
        target_host=None,
        out_idx=[0],
        execution_backend="tvm_ffi",
        pass_configs=None,
        compile_flags=None,
    )

    assert loaded is sentinel
    assert captured["func"] == prim_func
    assert captured["host_kernel_source"] is None
    assert captured["device_kernel_source"] is None
    assert captured["host_kernel_source_path"] == str(cache_path / cache.host_kernel_path)
    assert captured["device_kernel_source_path"] == str(cache_path / cache.device_kernel_path)


def test_jit_frontend_cache_hit_skips_tir_elaboration(monkeypatch):
    import tilelang
    import tilelang.language as T
    from tilelang.jit import JITImpl

    sentinel = object()
    calls = []

    @tilelang.jit
    def frontend_cached_kernel(block_m: int = 16):
        @T.prim_func
        def kernel():
            T.evaluate(0)

        return kernel

    def fake_load_frontend_cached(frontend_key_data, **kwargs):
        calls.append((frontend_key_data, kwargs))
        return sentinel

    def fail_compile(self, *args, **kwargs):
        raise AssertionError("frontend cache hit should not elaborate TIR")

    monkeypatch.setattr("tilelang.cache.load_frontend_cached", fake_load_frontend_cached)
    monkeypatch.setattr(JITImpl, "compile", fail_compile)

    assert frontend_cached_kernel(block_m=32) is sentinel
    assert calls
    assert "frontend_cached_kernel" in calls[0][0]["function"]


def test_kernel_cache_disk_hit_rejects_entries_missing_sources(cache_dirs, monkeypatch):
    cache = KernelCache()
    key = "missing-source-entry"
    cache_path = Path(cache._get_cache_path(key))
    cache_path.mkdir(parents=True)
    (cache_path / cache.kernel_lib_path).write_bytes(b"fake-so")
    with (cache_path / cache.params_path).open("wb") as f:
        cloudpickle.dump(["param"], f)

    def fail_from_database(cls, **kwargs):
        raise AssertionError("incomplete cache entries should miss before rebuilding from database")

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fail_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        target="cuda",
        target_host=None,
        out_idx=[0],
        execution_backend="tvm_ffi",
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is None


def test_nvrtc_adapter_host_source_lazy_loads(tmp_path):
    pytest.importorskip("cuda.bindings.driver", reason="NVRTC adapter requires cuda-python")
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter

    host_source_path = tmp_path / "host_kernel.cu"
    host_source_path.write_text("// nvrtc host source")
    adapter = NVRTCKernelAdapter.__new__(NVRTCKernelAdapter)
    adapter.host_func = None
    adapter._host_kernel_source_path = str(host_source_path)

    assert adapter.get_host_source() == "// nvrtc host source"
    assert adapter.host_func == "// nvrtc host source"


def test_cutedsl_adapter_host_source_lazy_loads(tmp_path):
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    host_source_path = tmp_path / "kernel.py"
    host_source_path.write_text("# cutedsl host source")
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.host_kernel_source = None
    adapter.host_func = None
    adapter._host_kernel_source_path = str(host_source_path)

    assert adapter.get_host_source() == "# cutedsl host source"
    assert adapter.host_kernel_source == "# cutedsl host source"


def test_tvm_ffi_source_fallback_handles_missing_runtime_module():
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    adapter = TVMFFIKernelAdapter.__new__(TVMFFIKernelAdapter)
    adapter.host_kernel_source = None
    adapter.device_kernel_source = None
    adapter._host_kernel_source_path = None
    adapter._device_kernel_source_path = None
    adapter.rt_mod = None

    assert adapter.get_host_source() is None
    assert adapter.get_device_source() is None
    assert adapter.get_kernel_source(kernel_only=True) == ""
    assert adapter.get_kernel_source(kernel_only=False) == ""


def test_kernel_cache_rewrites_incomplete_cache_dir(cache_dirs, tmp_path):
    cache = KernelCache()
    key = "atomic-repair"
    cache_path = Path(cache._get_cache_path(key))
    cache_path.mkdir(parents=True)
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
    cache_path = Path(cache._get_cache_path(key))
    staging_root = Path(cache._get_staging_root())

    def raise_write_error(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(cache, "_save_so_cubin_to_disk", raise_write_error)
    monkeypatch.setattr(cache.logger, "exception", record_exception)

    cache._save_kernel_to_disk(key, _make_fake_kernel(tmp_path))

    assert not cache_path.exists()
    assert "Error during atomic cache save" in logged
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_kernel_cache_does_not_publish_incomplete_dir_when_device_source_is_missing(cache_dirs, tmp_path, monkeypatch):
    cache = KernelCache()
    key = "atomic-missing-device-source"
    kernel = _make_fake_kernel(tmp_path)
    kernel.kernel_source = None
    logged = []
    cache_path = Path(cache._get_cache_path(key))
    staging_root = Path(cache._get_staging_root())

    def record_exception(message, *args, **kwargs):
        logged.append(message)

    monkeypatch.setattr(cache.logger, "exception", record_exception)

    cache._save_kernel_to_disk(key, kernel)

    assert not cache_path.exists()
    assert "Error during atomic cache save" in logged
    assert not staging_root.exists() or not any(staging_root.iterdir())


def test_nvrtc_kernel_cache_rewrites_dir_missing_launcher(cache_dirs, tmp_path):
    cache = NVRTCKernelCache()
    key = "nvrtc-atomic-repair"
    cache_path = Path(cache._get_cache_path(key))
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device kernel")
    (cache_path / cache.host_kernel_path).write_text("// host kernel")
    (cache_path / cache.kernel_lib_path).write_bytes(b"old-cubin")
    (cache_path / cache.params_path).write_bytes(b"old-params")
    (cache_path / "legacy.txt").write_text("stale")

    cache._save_kernel_to_disk(key, _make_fake_nvrtc_kernel(tmp_path))

    assert (cache_path / cache.kernel_py_path).exists()
    assert not (cache_path / "legacy.txt").exists()
