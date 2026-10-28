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


@pytest.mark.pto
def test_ascend_target_with_pto_execution_uses_pto_cache_dispatch():
    _, context, _ = _resolve_cache_dispatch("ascend", None, "pto", False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


@pytest.mark.pto
def test_ascend_env_defaults_can_select_pto(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "ascend")
    monkeypatch.setenv("TILELANG_EXECUTION_BACKEND", "pto")

    _, context, _ = _resolve_cache_dispatch(None, None, None, False)

    assert context.module.name == "pto"
    assert context.execution_backend.name == "pto"
    assert "pto" in context.target.keys


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target, backend", [("ascend", "cython"), pytest.param("pto", "pto", marks=pytest.mark.pto)])
def test_scalar_output_shape_disk_cache(target, backend, tmp_path, monkeypatch):
    """Cached scalar and expression dimensions work without a second compilation."""
    from tilelang.env import CacheState, env
    from tilelang.jit.kernel import JITKernel

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    cache, _, _ = _resolve_cache_dispatch(target, None, backend, False)
    monkeypatch.setattr(cache, "_memory_cache", {})

    @T.prim_func
    def main(out: T.handle, rows: T.int64, source: T.Tensor((1,), T.int32), bias: T.int64):
        output = T.match_buffer(out, (rows, rows * 2), T.int32)
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(64):
                if i < rows:
                    for j in T.serial(rows * 2):
                        output[i, j] = source[0] + i + j + T.int32(bias)

    # Distinct scalar arguments may share a TIR name hint.
    params = list(main.params)
    bias = tvm.tirx.Var(params[1].name, params[3].dtype)
    body = tvm.tirx.stmt_functor.substitute(main.body, {params[3]: bias})
    params[3] = bias
    main = tvm.tirx.PrimFunc(params, body, buffer_map=main.buffer_map, attrs=main.attrs)

    source = torch.tensor([7], dtype=torch.int32, device="npu")
    fresh = tilelang.compile(main, target=target, execution_backend=backend, out_idx=[0])
    torch.testing.assert_close(
        fresh(3, source, 11).cpu(), 18 + torch.arange(3, dtype=torch.int32)[:, None] + torch.arange(6, dtype=torch.int32)
    )
    cache._memory_cache.clear()

    def fail_compile(*args, **kwargs):
        pytest.fail("Expected disk-cache hit, not recompilation")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", fail_compile)
    restored = tilelang.compile(tvm.ir.load_json(tvm.ir.save_json(main)), target=target, execution_backend=backend, out_idx=[0])
    assert restored is not fresh
    for rows in (0, 3, 5):
        torch.testing.assert_close(
            restored(rows, source, 11).cpu(),
            18 + torch.arange(rows, dtype=torch.int32)[:, None] + torch.arange(rows * 2, dtype=torch.int32),
        )
