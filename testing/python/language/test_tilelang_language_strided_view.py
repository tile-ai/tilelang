"""Strided view shape, dtype and storage contracts (no device required)."""

import pytest
from tilelang import tvm
import tilelang.language as T
from tvm_tirx import ir
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as native
from tilelang._compat.tensor import tensor_strides

BATCH, ROWS, COLS_I8 = 2, 3, 16
COLS_I32 = COLS_I8 // 4


def strided_buffer(shape, dtype, strides, name="A"):
    return tvm.tirx.decl_buffer(shape, dtype, name=name, strides=strides)


def make_view(source, *args, **kwargs):
    with IRBuilder() as builder:
        with native.function():
            native.func_name_("view_contract")
            native.arg_("source", source)
            viewed = T.view(source, *args, **kwargs)
    (binding,) = builder.get().body.seq
    assert isinstance(binding, ir.Bind) and binding.var.same_as(viewed)
    assert binding.value.op.name == "tirx.decl_tensor"
    ir.assert_structural_equal(binding.value.args[0], source.data)
    return viewed


@pytest.mark.parametrize("src_dtype,dtype", [("float32", "float16"), ("float16", "float32"), ("float16x2", "float16")])
def test_scalar_view_rejects_different_bit_count(src_dtype, dtype):
    src = T.Tensor((), src_dtype)
    with pytest.raises(ValueError, match="logical bit count"):
        T.view(src, shape=(), dtype=dtype)


@pytest.mark.parametrize("src_dtype,dtype", [("float32", "float32"), ("float32", "int32"), ("float16x2", "int32")])
def test_scalar_view_preserves_bit_count(src_dtype, dtype):
    src = T.Tensor((), src_dtype)
    viewed = make_view(src, shape=(), dtype=dtype)
    assert len(viewed.shape) == 0
    assert str(viewed.dtype) == dtype


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
    viewed = make_view(src, shape=shape, dtype=dtype)
    assert tuple(int(dim) for dim in viewed.shape) == shape
    assert str(viewed.dtype) == dtype


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
    viewed = make_view(src, (BATCH * ROWS, COLS_I32), dtype=T.int32, strides=(stride1 // 4, 1))

    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(tensor_strides(viewed)[0], stride1 // 4)
    assert int(tensor_strides(viewed)[1]) == 1


def test_view_ignores_singleton_dimension_stride():
    src = strided_buffer((1, COLS_I8), "int8", (999, 1))
    viewed = make_view(src, (COLS_I8,))
    assert tuple(int(stride) for stride in tensor_strides(viewed)) == (1,)


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
