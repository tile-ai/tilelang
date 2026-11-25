"""Toolchain upgrades must miss the caches that restore complete executables.

No NPU execution is required: real cache publication and lookup are exercised
with synthetic binary payloads and a stubbed executable loader.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang import tvm
from tilelang.autotuner import AutoTuner
from tilelang.cache import _resolve_cache_dispatch
from tilelang.cache.kernel_cache import KernelCache
from tilelang.contrib import bisheng, pto_toolchain
from tilelang.env import CacheState, env
from tilelang.jit import JITKernel


CASES = [
    ("ascend", "tvm_ffi", "bisheng"),
    ("ascend", "tvm_ffi", "ld_lld"),
    ("ascend", "cython", "bisheng"),
    *[pytest.param("pto", backend, tool, marks=pytest.mark.pto) for backend in ("tvm_ffi", "pto") for tool in ("bisheng", "ptoas", "cann")],
]


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    versions = {
        "bisheng": "2026-07-01T09:55:44+08:00 clang version 15.0.5 (revision)",
        "ld_lld": "LLD 15.0.5 (revision)",
        "ptoas": "ptoas 0.67",
        "cann": "/opt/cann (9.2.0)",
    }
    monkeypatch.setattr(bisheng, "get_bisheng_version", lambda: versions["bisheng"])
    monkeypatch.setattr(bisheng, "get_ld_lld_version", lambda: versions["ld_lld"])
    monkeypatch.setattr(pto_toolchain, "toolchain_stamp", lambda: {key: versions[key] for key in ("ptoas", "cann", "bisheng")})
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    return versions


def _func():
    return tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr("global_symbol", "toolchain_probe")


def _publish(cache, key, payload, tmp_path):
    lib = tmp_path / "input.so"
    lib.write_bytes(payload)
    executable = SimpleNamespace(export_library=lambda path, **kwargs: Path(path).write_bytes(payload))
    adapter = SimpleNamespace(
        libpath=str(lib),
        get_kernel_source=lambda: "// host source",
        get_host_source=lambda: "// host source",
        get_exportable_executable=lambda: executable,
    )
    kernel = SimpleNamespace(adapter=adapter, params=[], kernel_source="// device source")
    cache._publish_kernel_to_disk(key, kernel)


@pytest.mark.parametrize("target, backend, tool", CASES)
def test_kernel_lookup_separates_toolchains(monkeypatch, tmp_path, toolchain, target, backend, tool):
    cache, context, _ = _resolve_cache_dispatch(target, None, backend, False)
    monkeypatch.setattr(cache, "_memory_cache", {})
    loaded_paths = []

    def load_kernel(cls, **kwargs):
        path = Path(kwargs["kernel_lib_path"])
        loaded_paths.append(path)
        return SimpleNamespace(payload=path.read_bytes())

    monkeypatch.setattr(JITKernel, "from_database", classmethod(load_kernel))

    def lookup():
        return cache.lookup(func=_func(), out_idx=None, backend_context=context, pass_configs=None, compile_flags=None)

    original_version = toolchain[tool]
    old_key, missing = lookup()
    assert missing is None
    _publish(cache, old_key, b"old toolchain", tmp_path)
    assert lookup()[1].payload == b"old toolchain"

    # Keep the memory entry as well as the disk entry: neither may be reused.
    toolchain[tool] = original_version + " changed build"
    new_key, missing = lookup()
    assert new_key != old_key
    assert missing is None
    assert len(loaded_paths) == 1
    cache._memory_cache.clear()
    assert lookup() == (new_key, None)
    assert len(loaded_paths) == 1
    _publish(cache, new_key, b"new toolchain", tmp_path)
    assert lookup()[1].payload == b"new toolchain"

    # A fresh process using the old toolchain can still reuse its own entry.
    cache._memory_cache.clear()
    toolchain[tool] = original_version
    restored_key, restored = lookup()
    assert restored_key == old_key
    assert restored.payload == b"old toolchain"
    assert loaded_paths[-1] == loaded_paths[0]


@pytest.mark.parametrize("backend", ["tvm_ffi", "cython"])
def test_legacy_kernel_entry_is_not_reused(monkeypatch, tmp_path, toolchain, backend):
    cache, context, _ = _resolve_cache_dispatch("ascend", None, backend, False)
    monkeypatch.setattr(cache, "_memory_cache", {})
    args = dict(func=_func(), out_idx=None, args=(), target=context.target, target_host=context.target_host, execution_backend=backend)
    with monkeypatch.context() as legacy:
        legacy.setattr(cache, "_get_toolchain_key", lambda *args: {})
        old_key = cache._generate_key(**args)
    _publish(cache, old_key, b"legacy toolchain", tmp_path)

    new_key, kernel = cache.lookup(func=args["func"], out_idx=None, backend_context=context, pass_configs=None, compile_flags=None)
    assert new_key != old_key
    assert kernel is None
    assert Path(cache._get_write_cache_path(old_key)).is_dir()


def _tunable(block_size=128):
    return block_size


@pytest.mark.parametrize("target, backend, tool", CASES)
def test_autotune_key_separates_toolchains(toolchain, target, backend, tool):
    tuner = AutoTuner(_tunable, configs=[{"block_size": 128}]).set_compile_args(target=target, execution_backend=backend)

    def key():
        return tuner.generate_cache_key(inspect.signature(_tunable).parameters, {})

    original_version = toolchain[tool]
    old_key = key()
    toolchain[tool] = original_version + " changed build"
    assert key() != old_key
    toolchain[tool] = original_version
    assert key() == old_key


def test_non_ascend_key_does_not_query_toolchain(monkeypatch):
    def unexpected_query():
        pytest.fail("Non-Ascend cache lookups must not query Ascend tools")

    monkeypatch.setattr(bisheng, "get_bisheng_version", unexpected_query)
    monkeypatch.setattr(bisheng, "get_ld_lld_version", unexpected_query)
    monkeypatch.setattr(pto_toolchain, "toolchain_stamp", unexpected_query)
    cache = KernelCache()
    args = dict(func=_func(), out_idx=None, args=(), target=tvm.target.Target("llvm"))
    key = cache._generate_key(**args)
    with monkeypatch.context() as legacy:
        legacy.setattr(cache, "_get_toolchain_key", lambda *args: {})
        assert cache._generate_key(**args) == key


def test_cython_cache_does_not_require_standalone_linker(monkeypatch, toolchain):
    def unexpected_query():
        pytest.fail("The Cython launcher does not invoke the standalone CCE linker")

    monkeypatch.setattr(bisheng, "get_ld_lld_version", unexpected_query)
    cache, context, _ = _resolve_cache_dispatch("ascend", None, "cython", False)
    assert cache._generate_key(func=_func(), out_idx=None, args=(), target=context.target, execution_backend="cython")


@pytest.mark.parametrize("tool", ["bisheng", "ld_lld"])
def test_version_query_failure_precedes_cache_lookup(monkeypatch, toolchain, tool):
    cache, context, _ = _resolve_cache_dispatch("ascend", None, "tvm_ffi", False)

    def failed_query():
        raise RuntimeError("tool version unavailable")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Must not load cached binaries when the toolchain cannot be identified")

    monkeypatch.setattr(bisheng, f"get_{tool}_version", failed_query)
    monkeypatch.setattr(cache, "_load_kernel_from_disk", unexpected_load)
    with pytest.raises(RuntimeError, match="tool version unavailable"):
        cache.lookup(func=_func(), out_idx=None, backend_context=context, pass_configs=None, compile_flags=None)
