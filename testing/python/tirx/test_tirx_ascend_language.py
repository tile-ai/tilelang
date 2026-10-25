import os

import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as builder_api

from tilelang._compat.tensor import TensorView

if os.environ.get("TILELANG_LIGHT_IMPORT") == "1":
    pytest.skip("Ascend language construction requires the full frontend", allow_module_level=True)

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_ascend_shape_helpers_follow_native_tensor_sources():
    from tilelang.ascend.language.copy_op import _get_shape

    source = TensorView.create("source", [16, 32], "float16", scope="shared")
    assert [int(extent) for extent in _get_shape(source.tensor)] == [16, 32]
    assert [int(extent) for extent in _get_shape(source.load([2, 3]))] == [16, 32]
    region = ir.TensorRegion(source.tensor, [ir.Range(2, 6), ir.Range(3, 11)], ir.TensorRegionType())
    assert [int(extent) for extent in _get_shape(region)] == [4, 8]


def test_ascend_simd_pair_reads_native_expression_type():
    from tilelang.ascend.language.simd import SimdPair

    value = ir.Var("pair", "float32x64")
    pair = SimdPair(value)
    assert tuple(map(str, pair.dtype)) == ("float32x64", "float32x64")


def test_ascend_mutable_pointer_metadata_uses_tensor_identity():
    from tilelang.ascend.language import simd
    from tilelang.language.frame import is_mutable_scalar

    with IRBuilder() as builder, builder_api.scope():
        source = builder_api.bind_(builder_api.alloc_tensor(ty_args=[tirx.TensorType([128], "float16", scope="shared")]))
        pointer = simd.make_ubuf_ptr(ir.TensorLoad(source, [0]), "float16")
        assert is_mutable_scalar(pointer)
        assert pointer.ty.dtype == ir.PrimType("uint64")
        assert simd._ubuf_ptr_dtype(ir.TensorLoad(pointer, [0])) == "float16"
        pair = simd.vld(ir.TensorLoad(pointer, [0]), post_inc=128)
        assert pair[0].ty == ir.PrimType("float16x128")
        advanced = pair[1]
        assert advanced.ty == ir.PrimType("uint64")
        assert isinstance(advanced.args[0].ty, ir.PtrType)
        builder_api.setitem_(pointer, 0, advanced)
    assert tirx.analysis.check_well_formed(tirx.Function([], builder.get()))
