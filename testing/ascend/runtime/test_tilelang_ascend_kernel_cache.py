"""Ascend cache artifacts preserve the selected execution backend's ABI."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.cache import _resolve_cache_dispatch


@pytest.mark.parametrize("backend, host_suffix", [("tvm_ffi", ".c"), ("cython", ".asc")])
def test_ascend_cache_source_suffixes(backend, host_suffix):
    cache, context, _ = _resolve_cache_dispatch("ascend", None, backend, False)
    assert context.module.name == "ascend"
    assert context.execution_backend.name == backend
    assert cache.device_kernel_path.endswith(".asc")
    assert cache.host_kernel_path.endswith(host_suffix)


def test_ascend_target_with_pto_execution_uses_pto_cache_dispatch():
    _, context, _ = _resolve_cache_dispatch("ascend", None, "pto", False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


def test_ascend_env_defaults_can_select_pto(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "ascend")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "pto")

    _, context, _ = _resolve_cache_dispatch(None, None, None, False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


@tilelang.testing.requires_ascend
def test_cython_scalar_output_shape_disk_cache(tmp_path, monkeypatch):
    """Cached explicit scalar dimensions work without a second compilation."""
    from tilelang.env import CacheState, env
    from tilelang.jit.kernel import JITKernel

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    cache, _, _ = _resolve_cache_dispatch("ascend", None, "cython", False)
    monkeypatch.setattr(cache, "_memory_cache", {})

    @T.prim_func
    def main(out: T.handle, rows: T.int64, source: T.Tensor((1,), T.int32)):
        output = T.match_buffer(out, (rows,), T.int32)
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(64):
                if i < rows:
                    output[i] = source[0] + i

    source = torch.tensor([7], dtype=torch.int32, device="npu")
    fresh = tilelang.compile(main, target="ascend", execution_backend="cython", out_idx=[0])
    torch.testing.assert_close(fresh(3, source).cpu(), torch.arange(7, 10, dtype=torch.int32))
    cache._memory_cache.clear()

    def fail_compile(*args, **kwargs):
        pytest.fail("Expected disk-cache hit, not recompilation")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", fail_compile)
    restored = tilelang.compile(tvm.ir.load_json(tvm.ir.save_json(main)), target="ascend", execution_backend="cython", out_idx=[0])
    assert restored is not fresh
    for rows in (0, 3, 5):
        torch.testing.assert_close(restored(rows, source).cpu(), torch.arange(7, 7 + rows, dtype=torch.int32))
