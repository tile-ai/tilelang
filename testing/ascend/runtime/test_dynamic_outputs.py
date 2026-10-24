"""Dynamic output allocation preserves Ascend argument and stream handling."""

import pytest
import torch
import shutil

import tilelang
import tilelang.testing
from tilelang.ascend import language as T


pytest.importorskip("torch_npu")

ROWS = T.dynamic("rows")
WIDTH = 128


@T.prim_func
def dynamic_outputs(
    positive: T.Tensor((ROWS, WIDTH), T.float32),
    source: T.Tensor((ROWS, WIDTH), T.float32),
    negative: T.Tensor((ROWS, WIDTH), T.float32),
    bias: T.int32,
):
    with T.Kernel(ROWS) as row, T.SimtVF(threads=WIDTH):
        for i in T.Parallel(WIDTH):
            positive[row, i] = source[row, i] + T.float32(bias)
            negative[row, i] = source[row, i] - T.float32(bias)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("execution_backend", ["cython", "tvm_ffi"])
@pytest.mark.parametrize("rows", [1, 3])
def test_dynamic_outputs_before_sizing_input(execution_backend, rows):
    kernel = tilelang.compile(dynamic_outputs, out_idx=[0, 2], target="ascend", execution_backend=execution_backend)
    reference = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH)
    source = reference.npu()
    torch.npu.synchronize()

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        prepared = source + 17
        positive, negative = kernel(prepared, 9)
    stream.synchronize()

    assert positive.device == negative.device == source.device
    torch.testing.assert_close(positive.cpu(), reference + 26, atol=0, rtol=0)
    torch.testing.assert_close(negative.cpu(), reference + 8, atol=0, rtol=0)


@T.prim_func
def expression_outputs(source: T.Tensor((ROWS, WIDTH), T.float32), count: T.int32):
    doubled = T.empty((ROWS * 2, WIDTH), T.float32)
    rounded = T.empty((T.ceildiv(ROWS, 2), WIDTH), T.float32)
    scalar = T.empty((count, WIDTH), T.float32)
    scalar_expr = T.empty((count * 2, WIDTH), T.float32)
    with T.Kernel(1), T.SimtVF(threads=WIDTH):
        for row in T.serial(ROWS):
            for i in T.Parallel(WIDTH):
                doubled[row * 2, i] = source[row, i] + 1.0
                doubled[row * 2 + 1, i] = source[row, i] - 1.0
        for row in T.serial(T.ceildiv(ROWS, 2)):
            for i in T.Parallel(WIDTH):
                rounded[row, i] = source[row * 2, i] + 2.0
        for row in T.serial(count):
            for i in T.Parallel(WIDTH):
                scalar[row, i] = source[0, i] + 3.0
                scalar_expr[row * 2, i] = source[0, i] + 4.0
                scalar_expr[row * 2 + 1, i] = source[0, i] + 5.0
    return doubled, rounded, scalar, scalar_expr


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("execution_backend", ["cython", "tvm_ffi", "pto"])
def test_expression_outputs_and_cache(execution_backend, tmp_path, monkeypatch):
    from tilelang.cache import _resolve_cache_dispatch
    from tilelang.env import env
    from tilelang.jit import JITKernel

    if execution_backend == "pto":
        pytest.importorskip("ptodsl")
        if shutil.which("ptoas") is None:
            pytest.skip("ptoas is required for PTO device execution")
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    cache, _, _ = _resolve_cache_dispatch("ascend", None, execution_backend, False)
    monkeypatch.setattr(cache, "_memory_cache", {})
    kernel = tilelang.compile(expression_outputs, target="ascend", execution_backend=execution_backend)
    cache._memory_cache.clear()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("The second compile must restore the disk cache")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", unexpected_compile)
    restored = tilelang.compile(expression_outputs, target="ascend", execution_backend=execution_backend)
    assert restored is not kernel
    for compiled in (kernel, restored):
        for rows, count in [(1, 1), (3, 2), (0, 0)]:
            reference = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH)
            source = reference.npu()
            torch.npu.synchronize()
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                outputs = compiled(source + 17, count)
            stream.synchronize()
            expected = [
                torch.stack((reference + 18, reference + 16), dim=1).reshape(rows * 2, WIDTH),
                reference[::2] + 19,
                (reference[:1] + 20).repeat(count, 1),
                torch.stack((reference[:1] + 21, reference[:1] + 22), dim=1).repeat(count, 1, 1).reshape(count * 2, WIDTH),
            ]
            for actual, wanted in zip(outputs, expected, strict=True):
                assert actual.device == source.device
                torch.testing.assert_close(actual.cpu(), wanted, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
