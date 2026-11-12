"""Exercise the ACL host-argument ABI used by the tvm-ffi Ascend runtime."""

from __future__ import annotations

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytest.importorskip("torch_npu")

ELEMENTS = 128
ROWS = T.dynamic("rows")
WIDTH = 128


@T.prim_func
def scalar_args_kernel(
    output: T.Tensor((ELEMENTS,), T.int64),
    left: T.int32,
    wide: T.int64,
    right: T.int32,
):
    with T.Kernel(1), T.SimtVF(threads=ELEMENTS):
        for i in T.Parallel(ELEMENTS):
            output[i] = left + wide + right


@T.prim_func
def dynamic_grid_kernel(
    source: T.Tensor((ROWS, WIDTH), T.bfloat16),
    destination: T.Tensor((ROWS, WIDTH), T.bfloat16),
):
    with T.Kernel(ROWS) as row:
        tile = T.alloc_shared((WIDTH,), T.bfloat16)
        T.copy(source[row, :], tile)
        T.copy(tile, destination[row, :])


@T.prim_func
def conditional_loop_kernel(
    source: T.Tensor((ROWS, WIDTH), T.float32),
    destination: T.Tensor((ROWS, WIDTH), T.float32),
    run: T.int32,
    count: T.int32,
):
    if run > 0:
        for rep in T.serial(count):
            with T.Kernel(ROWS) as row, T.SimtVF(threads=WIDTH):
                for i in T.Parallel(WIDTH):
                    destination[row, i] = source[row, i] + T.Cast(T.float32, rep)


def require_npu() -> None:
    if not torch.npu.is_available():
        pytest.skip("an available Ascend NPU is required")


def test_tvm_ffi_packs_mixed_width_scalars() -> None:
    require_npu()
    kernel = tilelang.compile(
        scalar_args_kernel,
        out_idx=0,
        target="ascend",
        execution_backend="tvm_ffi",
    )

    left = 17
    wide = 1 << 40
    right = -9
    output = kernel(left, wide, right)
    torch.npu.synchronize()

    expected = torch.full_like(output, left + wide + right)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("rows", [1, 8])
def test_tvm_ffi_dynamic_grid(rows: int) -> None:
    require_npu()
    kernel = tilelang.compile(
        dynamic_grid_kernel,
        out_idx=-1,
        target="ascend",
        execution_backend="tvm_ffi",
    )

    source = torch.randn(
        (rows, WIDTH),
        dtype=torch.bfloat16,
        device="npu",
    )
    destination = kernel(source)
    torch.npu.synchronize()

    torch.testing.assert_close(destination, source)


def test_exception_dump_packs_mixed_width_scalars() -> None:
    require_npu()
    kernel = tilelang.compile(
        scalar_args_kernel,
        out_idx=0,
        target="ascend",
        execution_backend="tvm_ffi",
        pass_configs={"tl.ascend_exception_dump": True},
    )
    output = kernel(17, 1 << 40, -9)
    torch.npu.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, (1 << 40) + 8))


@pytest.mark.parametrize("cache_layout", ["single", "secondary", "corrupt-secondary"])
def test_exception_dump_dynamic_shape_disk_cache(tmp_path, monkeypatch, cache_layout):
    """Restore dump modules from ordered roots without rewriting published entries."""
    import hashlib
    import os
    import shutil
    from pathlib import Path

    require_npu()
    from tilelang import tvm
    from tilelang.cache import _resolve_cache_dispatch
    from tilelang.env import CacheState, env
    from tilelang.jit.kernel import JITKernel

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    cache, _, _ = _resolve_cache_dispatch("ascend", None, "tvm_ffi", False)
    monkeypatch.setattr(cache, "_memory_cache", {})
    options = dict(
        out_idx=-1,
        target="ascend",
        execution_backend="tvm_ffi",
        pass_configs={"tl.ascend_exception_dump": True},
    )
    fresh = tilelang.compile(dynamic_grid_kernel, **options)
    source = torch.randn((1, WIDTH), dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(fresh(source).cpu(), source.cpu())
    cache._memory_cache.clear()
    entry = fresh._tilelang_cache_path
    if cache_layout != "single":
        # The new writable root misses; the original root acts as read-only.
        roots = [tmp_path / "new-writes", tmp_path]
        if cache_layout == "corrupt-secondary":
            fallback = tmp_path / "fallback"
            fallback_entry = fallback / Path(entry).relative_to(tmp_path)
            shutil.copytree(entry, fallback_entry)
            roots.append(fallback)
            # A corrupt higher-priority entry must raise, not fall through to
            # the valid copy or recompile, per the upstream immutable-cache policy.
            library = Path(entry) / cache.kernel_lib_path
            data = library.read_bytes()
            library.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        monkeypatch.setattr(env, "TILELANG_CACHE_DIR", os.pathsep.join(map(str, roots)))
    published = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(entry).iterdir() if p.is_file()}

    def fail_compile(*args, **kwargs):
        pytest.fail("Expected exception-dump disk-cache hit, not recompilation")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", fail_compile)
    saved_ir = tvm.ir.load_json(tvm.ir.save_json(dynamic_grid_kernel))
    if cache_layout == "corrupt-secondary":
        with pytest.raises(ValueError, match="integrity verification"):
            tilelang.compile(saved_ir, **options)
        assert published == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(entry).iterdir() if p.is_file()}
        assert not list((tmp_path / "new-writes").rglob(cache.kernel_lib_path))
        return
    restored = tilelang.compile(saved_ir, **options)
    assert restored._tilelang_cache_path == entry
    assert restored is not fresh
    for rows in (2, 8):
        source = torch.randn((rows, WIDTH), dtype=torch.bfloat16, device="npu")
        torch.testing.assert_close(restored(source).cpu(), source.cpu())
    assert published == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(entry).iterdir() if p.is_file()}
    if cache_layout != "single":
        assert not list((tmp_path / "new-writes").rglob(cache.kernel_lib_path))


@pytest.mark.parametrize("enabled", [False, True])
def test_tvm_ffi_exception_dump_preserves_existing_host_control(enabled):
    require_npu()
    kernel = tilelang.compile(
        conditional_loop_kernel,
        target="ascend",
        execution_backend="tvm_ffi",
        pass_configs={"tl.ascend_exception_dump": enabled},
    )
    for rows in (1, 3):
        source = torch.arange(rows * WIDTH, device="npu", dtype=torch.float32).reshape(rows, WIDTH) / 4
        for run, count in ((0, 3), (1, 0), (1, 1), (1, 3)):
            destination = torch.full_like(source, -99)
            kernel(source, destination, run, count)
            torch.npu.synchronize()
            expected = source + count - 1 if run > 0 and count > 0 else torch.full_like(source, -99)
            torch.testing.assert_close(destination, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
