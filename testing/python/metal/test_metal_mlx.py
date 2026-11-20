"""Optional MLX runtime coverage for functional and in-place kernels."""

import numpy as np
import pytest

import tilelang.testing
from tilelang.metal import language as T
from tilelang.contrib.mlx import compile as compile_mlx

mx = pytest.importorskip("mlx.core")
pytest.importorskip("_tilelang_mlx")


def make_add(n=1003, dtype="float32"):
    @T.prim_func
    def add(c: T.Tensor((n,), dtype), b: T.Tensor((n,), dtype), a: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 256), threads=128) as bx:
            for i in T.Parallel(256):
                j = bx * 256 + i
                if j < n:
                    c[j] = a[j] + b[j]

    return add


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype,n", [("float32", 1003), ("float16", 1024), ("int32", 256)])
def test_mlx_add_and_lazy_chain(dtype, n):
    kernel = compile_mlx(make_add(n, dtype), out_idx=0)
    a = mx.arange(n).astype(getattr(mx, dtype))
    b = mx.ones((n,), dtype=getattr(mx, dtype))
    result = kernel(b, a)
    result = kernel(b, result)
    del a, b
    mx.eval(result)
    np.testing.assert_array_equal(np.array(result), (np.arange(n) + 2).astype(dtype))


@tilelang.testing.requires_metal
def test_mlx_functional_cache_update_and_stream():
    @T.prim_func
    def update(old: T.Tensor((64,), T.float32), delta: T.Tensor((16,), T.float32), updated: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(64):
                if i >= 16 and i < 32:
                    updated[i] = old[i] + delta[i - 16]
                else:
                    updated[i] = old[i]

    kernel = compile_mlx(update, out_idx=-1)
    old = mx.arange(64).astype(mx.float32)
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        result = kernel(old * 2, mx.ones((16,)), stream=stream)
        result = result + 3
    mx.eval(result)
    expected = np.arange(64, dtype=np.float32) * 2 + 3
    expected[16:32] += 1
    np.testing.assert_array_equal(np.array(result), expected)
    np.testing.assert_array_equal(np.array(old), np.arange(64))


@tilelang.testing.requires_metal
def test_mlx_gemm_noncontiguous_inputs():
    @T.prim_func
    def matmul(a: T.Tensor((32, 32), T.float16), b: T.Tensor((32, 32), T.float16), c: T.Tensor((32, 32), T.float32)):
        with T.Kernel(1, threads=128):
            sa = T.alloc_shared((32, 32), T.float16)
            sb = T.alloc_shared((32, 32), T.float16)
            sc = T.alloc_shared((32, 32), T.float32)
            T.copy(a, sa)
            T.copy(b, sb)
            T.clear(sc)
            T.gemm(sa, sb, sc)
            T.copy(sc, c)

    kernel = compile_mlx(matmul, out_idx=2)
    a_np = (np.arange(1024).reshape(32, 32) % 7 - 3).astype(np.float16)
    b_np = (np.arange(1024).reshape(32, 32) % 5 - 2).astype(np.float16)
    result = kernel(mx.array(a_np).T, mx.array(b_np))
    mx.eval(result)
    np.testing.assert_array_equal(np.array(result), a_np.T.astype(np.int64) @ b_np.astype(np.int64))


@tilelang.testing.requires_metal
def test_mlx_input_contract():
    kernel = compile_mlx(make_add(32), out_idx=0)
    with pytest.raises(ValueError, match="Expected 2"):
        kernel(mx.ones((32,)))
    with pytest.raises(ValueError, match="requires shape"):
        kernel(mx.ones((16,)), mx.ones((32,)))
    with pytest.raises(ValueError, match="requires shape"):
        kernel(mx.ones((32,), dtype=mx.float16), mx.ones((32,)))
    with pytest.raises(TypeError, match="mlx.core.array"):
        kernel(np.ones(32), mx.ones((32,)))


@tilelang.testing.requires_metal
def test_mlx_reject_undeclared_mutation_and_dynamic_shapes():
    @T.prim_func
    def inplace(a: T.Tensor((32,), T.float32), b: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                a[i] = a[i] + 1
                b[i] = a[i]

    with pytest.raises(ValueError, match="immutable"):
        compile_mlx(inplace, out_idx=1)

    n = T.dynamic("n")

    @T.prim_func
    def dynamic(a: T.Tensor((n,), T.float32), b: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, 32), threads=32):
            for i in T.Parallel(32):
                b[i] = a[i]

    with pytest.raises(ValueError, match="static tensor shapes"):
        compile_mlx(dynamic, out_idx=1)


@tilelang.testing.requires_metal
def test_mlx_multiple_outputs_and_unused_input():
    @T.prim_func
    def split(
        unused: T.Tensor((32,), T.float32),
        first: T.Tensor((32,), T.float32),
        value: T.Tensor((32,), T.float32),
        second: T.Tensor((32,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                first[i] = value[i] + 1
                second[i] = value[i] * 2

    kernel = compile_mlx(split, out_idx=[3, 1])
    doubled, incremented = kernel(mx.zeros((32,)), mx.arange(32).astype(mx.float32))
    mx.eval(doubled, incremented)
    np.testing.assert_array_equal(np.array(doubled), np.arange(32) * 2)
    np.testing.assert_array_equal(np.array(incremented), np.arange(32) + 1)


@tilelang.testing.requires_metal
def test_mlx_offset_slice():
    kernel = compile_mlx(make_add(1024), out_idx=0)
    a = mx.arange(1025).astype(mx.float32)[1:]
    result = kernel(mx.ones((1024,)), a)
    mx.eval(result)
    np.testing.assert_array_equal(np.array(result), np.arange(1024) + 2)


@tilelang.testing.requires_metal
def test_mlx_reject_multiple_launches_and_scalar_parameters():
    @T.prim_func
    def two_launches(a: T.Tensor((32,), T.float32), b: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                b[i] = a[i]
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                b[i] += 1

    with pytest.raises(ValueError, match="exactly one device kernel"):
        compile_mlx(two_launches, out_idx=1)

    @T.prim_func
    def scalar(a: T.Tensor((32,), T.float32), b: T.Tensor((32,), T.float32), scale: T.float32):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                b[i] = a[i] * scale

    with pytest.raises(ValueError, match="tensor parameters only"):
        compile_mlx(scalar, out_idx=1)


# Mirrors vLLM-metal's GDN scatter contract: distinct valid destination slots,
# negative/padding slots ignored, and the returned state alias carries the write.
def make_state_scatter(dtype="float32"):
    @T.prim_func
    def scatter(pool: T.Tensor((8, 16), dtype), rows: T.Tensor((4, 16), dtype), ids: T.Tensor((4,), T.int32)):
        with T.Kernel(4, threads=32) as bx:
            for j in T.Parallel(16):
                if ids[bx] >= 0 and ids[bx] < 8:
                    pool[ids[bx], j] = rows[bx, j]

    return compile_mlx(scatter, inplace_idx=0)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_mlx_inplace_scatter_alias_and_guards(dtype):
    scatter = make_state_scatter(dtype)
    backing = mx.full((10, 16), -77, dtype=getattr(mx, dtype))
    mx.eval(backing)
    pool = backing[1:9]
    values = np.arange(64).reshape(4, 16).astype(dtype)
    updated = scatter(pool, mx.array(values), mx.array([6, -1, 2, 100], dtype=mx.int32))
    mx.eval(updated)
    expected = np.full((10, 16), -77, dtype=dtype)
    expected[7] = values[0]
    expected[3] = values[2]
    # Observing the backing allocation after completion proves storage aliasing,
    # not merely numerical equivalence of a newly allocated output.
    np.testing.assert_array_equal(np.array(backing), expected)
    np.testing.assert_array_equal(np.array(updated), expected[1:9])


@tilelang.testing.requires_metal
def test_mlx_inplace_cross_stream_chain_and_lifetime():
    scatter = make_state_scatter()
    producer, consumer = mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)
    with mx.stream(producer):
        state = mx.ones((8, 16)) * 3
        rows = mx.arange(64).reshape(4, 16).astype(mx.float32)
        state = scatter(state, rows, mx.array([0, 2, 4, 6], dtype=mx.int32), stream=producer)
    with mx.stream(consumer):
        state = scatter(state, rows + 100, mx.array([1, 3, 5, 7], dtype=mx.int32), stream=consumer)
        result = state * 2 + 1
    del state, rows
    mx.eval(result)
    values = np.arange(64, dtype=np.float32).reshape(4, 16)
    expected = np.empty((8, 16), dtype=np.float32)
    expected[::2], expected[1::2] = values, values + 100
    np.testing.assert_array_equal(np.array(result), expected * 2 + 1)


@tilelang.testing.requires_metal
def test_mlx_inplace_compiled_repeated_updates():
    @T.prim_func
    def step(state: T.Tensor((32,), T.float32), result: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                state[i] += 1
                result[i] = state[i] * 2

    kernel = compile_mlx(step, out_idx=1, inplace_idx=0)

    @mx.compile
    def run(state):
        _, state = kernel(state)
        return kernel(state)

    state = mx.zeros((32,))
    for iteration in range(3):
        result, state = run(state)
        mx.eval(result, state)
        np.testing.assert_array_equal(np.array(state), np.full(32, 2 * (iteration + 1)))
        np.testing.assert_array_equal(np.array(result), np.full(32, 4 * (iteration + 1)))


@tilelang.testing.requires_metal
def test_mlx_inplace_reject_noncontiguous_state_and_overlap():
    scatter = make_state_scatter()
    rows = mx.ones((4, 16))
    ids = mx.array([0, 1, 2, 3], dtype=mx.int32)
    state = mx.arange(128).astype(mx.float32).reshape(16, 8).T
    # Evaluate the transpose before construction, preserving its strided view.
    mx.eval(state)
    with pytest.raises(ValueError, match="row-contiguous"):
        mx.eval(scatter(state, rows, ids))
    state = mx.ones((8, 16))
    mx.eval(state)
    with pytest.raises(ValueError, match="overlaps"):
        mx.eval(scatter(state, state[:4], ids))


@tilelang.testing.requires_metal
def test_mlx_alias_index_validation():
    with pytest.raises(ValueError, match="disjoint"):
        compile_mlx(make_add(32), out_idx=0, inplace_idx=0)
    with pytest.raises(ValueError, match="distinct"):
        compile_mlx(make_add(32), inplace_idx=[0, 0])


@tilelang.testing.requires_metal
@pytest.mark.parametrize("width", [5, 12])
def test_mlx_paged_scatter_preserves_padding(width):
    # Matches vLLM-metal's two pages, three live tokens and one padding row.
    page_stride = 4 * width

    @T.prim_func
    def scatter(
        pool: T.StridedTensor((2, 3, width), (page_stride, width, 1), T.float32),
        rows: T.Tensor((3, width), T.float32),
        ids: T.Tensor((3,), T.int32),
    ):
        with T.Kernel(3, threads=32) as bx:
            for j in T.Parallel(width):
                if ids[bx] >= 0 and ids[bx] < 6:
                    pool[ids[bx] // 3, ids[bx] % 3, j] = rows[bx, j]

    kernel = compile_mlx(scatter, inplace_idx=0)
    backing = mx.full((2 * page_stride,), 99, dtype=mx.float32)
    mx.eval(backing)
    pool = mx.as_strided(backing, (2, 3, width), (page_stride, width, 1))
    rows = mx.arange(3 * width).reshape(3, width).astype(mx.float32)
    updated = kernel(pool, rows, mx.array([4, 1, 5], dtype=mx.int32))
    mx.eval(updated)
    expected = np.full(2 * page_stride, 99, dtype=np.float32)
    for row, slot in enumerate([4, 1, 5]):
        start = (slot // 3) * page_stride + (slot % 3) * width
        expected[start : start + width] = np.arange(row * width, (row + 1) * width)
    np.testing.assert_array_equal(np.array(backing), expected)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("compiled", [False, True])
def test_mlx_consecutive_updates_keep_distinct_outputs(compiled):
    # vLLM-metal checks that two recurrences share state but not their outputs.
    @T.prim_func
    def step(state: T.Tensor((32,), T.float32), result: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                state[i] += 1
                result[i] = state[i] * 2

    kernel = compile_mlx(step, out_idx=1, inplace_idx=0)

    def run(state):
        first, state = kernel(state)
        second, state = kernel(state)
        return first, second, state

    run = mx.compile(run) if compiled else run
    first, second, state = run(mx.zeros((32,)))
    mx.eval(first, second, state)
    np.testing.assert_array_equal(np.array(first), np.full(32, 2))
    np.testing.assert_array_equal(np.array(second), np.full(32, 4))
    np.testing.assert_array_equal(np.array(state), np.full(32, 2))


@tilelang.testing.requires_metal
def test_mlx_multiple_state_aliases_disjoint_views():
    @T.prim_func
    def step(left: T.Tensor((17,), T.float32), right: T.Tensor((17,), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(17):
                left[i] += 1
                right[i] += 10

    kernel = compile_mlx(step, inplace_idx=[1, 0])
    backing = mx.zeros((40,))
    mx.eval(backing)
    right, left = kernel(backing[1:18], backing[21:38])
    mx.eval(right, left)
    expected = np.zeros(40)
    expected[1:18], expected[21:38] = 1, 10
    np.testing.assert_array_equal(np.array(backing), expected)
    np.testing.assert_array_equal(np.array(left), np.ones(17))
    np.testing.assert_array_equal(np.array(right), np.full(17, 10))


@tilelang.testing.requires_metal
@pytest.mark.parametrize("inplace", [False, True])
def test_mlx_singleton_axis_strides(inplace):
    @T.prim_func
    def copy(a: T.Tensor((1, 32), T.float32), b: T.Tensor((1, 32), T.float32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                b[0, i] = a[0, i] + 1

    kernel = compile_mlx(copy, inplace_idx=1) if inplace else compile_mlx(copy, out_idx=1)
    a = mx.arange(32).astype(mx.float32)[None, :]
    backing = mx.zeros((32,))
    mx.eval(backing)
    result = kernel(a, backing[None, :]) if inplace else kernel(a)
    mx.eval(result)
    np.testing.assert_array_equal(np.array(result), (np.arange(32) + 1)[None, :])
    if inplace:
        np.testing.assert_array_equal(np.array(backing), np.arange(32) + 1)


@tilelang.testing.requires_metal
def test_mlx_long_cross_stream_chain():
    scatter = make_state_scatter()
    streams = [mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)]
    state = mx.zeros((8, 16))
    expected = np.zeros((8, 16), dtype=np.float32)
    for iteration in range(64):
        slots = [0, 2, 4, 6] if iteration % 2 == 0 else [1, 3, 5, 7]
        with mx.stream(streams[iteration % 2]):
            values = mx.full((4, 16), iteration, dtype=mx.float32)
            state = scatter(state, values, mx.array(slots, dtype=mx.int32))
        expected[slots] = iteration
    mx.eval(state)
    np.testing.assert_array_equal(np.array(state), expected)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("offset", [1, 3])
def test_mlx_unaligned_state_and_source_views(dtype, offset):
    # vLLM-metal varies both source and destination alignment independently of
    # row widths. Check the entire allocation, not just the returned view.
    scatter = make_state_scatter(dtype)
    backing = mx.full((128 + offset + 7,), -11, dtype=getattr(mx, dtype))
    source = mx.arange(64 + 3).astype(getattr(mx, dtype))
    mx.eval(backing, source)
    state = backing[offset : offset + 128].reshape(8, 16)
    rows = source[3:].reshape(4, 16)
    updated = scatter(state, rows, mx.array([7, 1, -1, 4], dtype=mx.int32))
    mx.eval(updated)
    expected = np.full(128 + offset + 7, -11, dtype=dtype)
    values = np.arange(3, 67, dtype=dtype).reshape(4, 16)
    for row, slot in [(0, 7), (1, 1), (3, 4)]:
        expected[offset + slot * 16 : offset + (slot + 1) * 16] = values[row]
    np.testing.assert_array_equal(np.array(backing), expected)
    np.testing.assert_array_equal(np.array(source), np.arange(67, dtype=dtype))


@tilelang.testing.requires_metal
@pytest.mark.parametrize("singleton_stride", [0, 1])
def test_mlx_declared_singleton_state_strides(singleton_stride):
    @T.prim_func
    def update(state: T.StridedTensor((2, 1, 8), (8, singleton_stride, 1), T.float32)):
        with T.Kernel(1, threads=32):
            for i, j in T.Parallel(2, 8):
                state[i, 0, j] += 3

    kernel = compile_mlx(update, inplace_idx=0)
    backing = mx.arange(18).astype(mx.float32)
    mx.eval(backing)
    result = kernel(backing[1:17].reshape(2, 1, 8))
    mx.eval(result)
    expected = np.arange(18, dtype=np.float32)
    expected[1:17] += 3
    np.testing.assert_array_equal(np.array(backing), expected)


@tilelang.testing.requires_metal
def test_mlx_reject_declared_overlapping_state():
    @T.prim_func
    def update(state: T.StridedTensor((2, 1, 8), (4, 0, 1), T.float32)):
        with T.Kernel(1, threads=32):
            for i, j in T.Parallel(2, 8):
                state[i, 0, j] += 1

    with pytest.raises(ValueError, match="non-overlapping"):
        compile_mlx(update, inplace_idx=0)
