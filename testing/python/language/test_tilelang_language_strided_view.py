"""Strided view shape, dtype and storage contracts (no device required)."""

import pytest
from tilelang import tvm
import tilelang.language as T

BATCH, ROWS, COLS_I8 = 2, 3, 16
COLS_I32 = COLS_I8 // 4


def strided_buffer(shape, dtype, strides, name="A"):
    return tvm.tirx.decl_buffer(shape, dtype, name=name, strides=strides)


@pytest.mark.parametrize("src_dtype,dtype", [("float32", "float16"), ("float16", "float32"), ("float16x2", "float16")])
def test_scalar_view_rejects_different_bit_count(src_dtype, dtype):
    src = T.Tensor((), src_dtype)
    with pytest.raises(ValueError, match="logical bit count"):
        T.view(src, shape=(), dtype=dtype)


@pytest.mark.parametrize("src_dtype,dtype", [("float32", "float32"), ("float32", "int32"), ("float16x2", "int32")])
def test_scalar_view_preserves_bit_count(src_dtype, dtype):
    src = T.Tensor((), src_dtype)
    viewed = T.view(src, shape=(), dtype=dtype)
    assert len(viewed.shape) == 0
    assert str(viewed.dtype) == dtype
    assert viewed.data.same_as(src.data)


@pytest.mark.parametrize(
    "src_shape,src_dtype,shape,dtype",
    [
        ((), "float32", (1,), "float32"),
        ((1,), "float32", (), "float32"),
        ((), "float32", (2,), "float16"),
        ((2,), "float16", (), "float32"),
    ],
)
def test_scalar_view_rank_change(src_shape, src_dtype, shape, dtype):
    src = T.Tensor(src_shape, src_dtype)
    viewed = T.view(src, shape=shape, dtype=dtype)
    assert tuple(int(dim) for dim in viewed.shape) == shape
    assert str(viewed.dtype) == dtype
    assert viewed.data.same_as(src.data)


def test_view_requires_explicit_strides_for_noncontiguous_source():
    stride0 = T.dynamic("stride0", dtype=T.int64)
    stride1 = T.dynamic("stride1", dtype=T.int64)
    src = strided_buffer((BATCH, ROWS, COLS_I8), "int8", (stride0, stride1, 1))
    with pytest.raises(ValueError, match="requires explicit strides"):
        T.view(src, (BATCH, ROWS, COLS_I32), dtype=T.int32)


def test_view_accepts_explicit_strides_for_rank_change():
    stride0 = T.dynamic("stride0", dtype=T.int64)
    stride1 = T.dynamic("stride1", dtype=T.int64)
    src = strided_buffer((BATCH, ROWS, COLS_I8), "int8", (stride0, stride1, 1))
    viewed = T.view(src, (BATCH * ROWS, COLS_I32), dtype=T.int32, strides=(stride1 // 4, 1))

    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(viewed.strides[0], stride1 // 4)
    assert int(viewed.strides[1]) == 1
    assert viewed.data.same_as(src.data)


def test_view_ignores_singleton_dimension_stride():
    src = strided_buffer((1, COLS_I8), "int8", (999, 1))
    viewed = T.view(src, (COLS_I8,))
    assert tuple(int(stride) for stride in viewed.strides) == (1,)


def test_view_rejects_invalid_strided_views():
    src = strided_buffer((BATCH, ROWS, COLS_I8), "int8", (6144, 512, 1))
    with pytest.raises(ValueError, match="requires explicit strides"):
        T.view(src, (BATCH * ROWS, COLS_I8))
    with pytest.raises(ValueError, match="expected 2 strides"):
        T.view(src, (BATCH * ROWS, COLS_I32), dtype=T.int32, strides=(1,))
    with pytest.raises(ValueError, match="logical bit count"):
        T.view(src, (BATCH * ROWS, 100), dtype=T.int32, strides=(100, 1))
    offset_src = tvm.tirx.decl_buffer((16,), "int8", name="offset_src", elem_offset=1)
    with pytest.raises(ValueError, match="non-zero elem_offset"):
        T.view(offset_src, (16,))
