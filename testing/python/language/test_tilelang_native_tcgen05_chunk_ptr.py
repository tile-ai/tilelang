"""Scale-factor chunk pointers retain element units and the source view origin."""

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang
import tilelang.language as T
from tilelang.language import builtin
from tilelang.utils.language import retrieve_buffer_and_offset


def _nodes(value, node_type):
    result = []
    tvm_ffi.structural_walk(value, (node_type, result.append), order="pre")
    return result


@pytest.mark.parametrize("mask", [1, 3])
def test_chunk_pointer_carries_offset_and_remaining_extent_in_element_units(mask):
    layout = tirx.TileLayout.from_iters(
        [tirx.layout.Iter(3, 512, "m"), tirx.layout.Iter(256, 1, "m")],
        offset={"m": 7},
    )
    source = ir.Var("S", tirx.TensorType((3, 256), "uint32", scope="shared", layout=layout))
    region = ir.TensorRegion(source, [ir.Range.from_min_extent(1, 1), ir.Range.from_min_extent(0, 256)], ir.TensorRegionType())
    tensor, start = retrieve_buffer_and_offset(region)
    analyzer = sym.Analyzer()
    for chunk in range(2):
        pointer = builtin._tcgen05_chunk_ptr(tensor, start + chunk * 128, mask)
        pointer.validate()
        assert pointer.op.name == "tl.ptr_access"
        assert pointer.ty == ir.PtrType(ir.PrimType("uint32"), "shared")
        assert pointer.args[0].op.name == "tirx.tensor_data_ptr"
        assert pointer.args[0].args[0].same_as(source)
        assert int(analyzer.simplify(pointer.args[1])) == 7 + 512 + chunk * 128
        assert int(analyzer.simplify(pointer.args[2])) == 1536 - 512 - chunk * 128
        assert int(pointer.args[3]) == mask


@pytest.mark.parametrize("dtype,itemsize", [("uint8", 1), ("uint32", 4), ("float16", 2)])
def test_chunk_pointer_real_llvm_address_includes_alias_origin_once(dtype, itemsize):
    @T.prim_func
    def sample(A: T.Tensor((2048,), dtype), B: T.Tensor((2,), "uint64")):
        alias = T.decl_buffer((3, 256), dtype, data=A.data, strides=(512, 1), elem_offset=7, align=1)
        first = builtin._tcgen05_chunk_ptr(alias, 512, 1)
        second = builtin._tcgen05_chunk_ptr(alias, 512 + 128, 3)
        B[0] = T.reinterpret("uint64", first) - T.reinterpret("uint64", A.data)
        B[1] = T.reinterpret("uint64", second) - T.reinterpret("uint64", A.data)

    assert tirx.analysis.check_well_formed(sample)
    target = {"kind": "llvm", "mcpu": "generic"}
    kernel = tilelang.compile(sample, target=target, target_host=target, execution_backend="tvm_ffi")
    source = torch.zeros(2048, dtype=getattr(torch, dtype))
    output = torch.zeros(2, dtype=torch.uint64)
    kernel(source, output)
    np.testing.assert_array_equal(output.numpy(), [(7 + 512) * itemsize, (7 + 512 + 128) * itemsize])


@pytest.mark.parametrize("transpose", [False, True])
def test_public_chunked_scale_helper_accepts_stage_slices(transpose):
    @T.prim_func
    def sample():
        source = T.decl_buffer((3, 256), "uint32", scope="shared")
        destination = T.decl_buffer((128, 8), "uint32", scope="shared.tmem")
        if transpose:
            T.tcgen05_sf_warp_transpose(source[1, :])
        else:
            T.tcgen05_cp_warpx4(source[1, :], destination, tmem_col_offset=2)

    assert tirx.analysis.check_well_formed(sample)
    (loop,) = _nodes(sample, ir.For)
    assert int(loop.begin) == 0 and int(loop.end) == 2
    calls = _nodes(sample, ir.Call)
    (pointer,) = [call for call in calls if call.op.name == "tl.ptr_access"]
    assert int(pointer.args[3]) == (3 if transpose else 1)
    expected = "tl.ptx_tcgen05_sf_warp_transpose" if transpose else "tl.ptx_tcgen05_cp_warpx4"
    assert len([call for call in calls if call.op.name == expected]) == 1
    for chunk in range(2):
        analyzer = sym.Analyzer()
        constraints = sym.ConstraintContext()
        constraints.append(sym.BindingConstraint(loop.var, ir.const(chunk)))
        analyzer.sync_constraint_context(constraints)
        assert int(analyzer.simplify(pointer.args[1])) == 256 + chunk * 128


@pytest.mark.parametrize("constructor", [T.tcgen05_before_thread_sync, T.tcgen05_after_thread_sync])
def test_fixed_void_fence_constructor_matches_injected_native_call(constructor):
    result = constructor()
    result.validate()
    assert result.ty == ir.PrimType("void")
    assert not result.ty_args
    assert not result.args
    assert ir.assert_structural_equal(result, ir.Call(result.op, [], ty=ir.PrimType("void"), loc=result.loc)) is None
