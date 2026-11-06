"""Barrier macro parameters denote the selected slot, not its loaded value."""

from tilelang.language import _tensor_spec

import pytest
from tvm_tirx import ir, tirx

import tilelang.language as T

from tilelang.testing.ir import collect_nodes


@pytest.mark.parametrize("macro_depth", [0, 1, 2])
@pytest.mark.parametrize("arrive_2cta", [False, True])
def test_tcgen05_arrive_preserves_macro_slot_pointer(macro_depth, arrive_2cta):
    @T.macro
    def arrive(slot):
        T.tcgen05_mma_arrive(slot, arrive_2cta=arrive_2cta)

    @T.macro
    def nested(slot):
        arrive(slot)

    @T.prim_func
    def kernel(storage: T.Tensor((16,), "uint64", scope="shared.barrier")):
        barrier = T.decl_buffer(
            (2,),
            "uint64",
            data=storage.data,
            scope="shared.barrier",
            strides=(2,),
            elem_offset=5,
        )
        if macro_depth == 0:
            T.tcgen05_mma_arrive(barrier[1], arrive_2cta=arrive_2cta)
        elif macro_depth == 1:
            arrive(barrier[1])
        else:
            nested(barrier[1])

    (arrival,) = [call for call in collect_nodes(kernel, ir.Call) if call.op.name == "tl.tcgen05_mma_arrive"]
    definitions = {binding.var: binding.value for binding in collect_nodes(kernel, ir.Bind)}
    pointer = arrival.args[0]
    while isinstance(pointer, ir.Var) and pointer in definitions:
        pointer = definitions[pointer]
    assert isinstance(pointer.ty, ir.PtrType)
    assert pointer.op.name == "tl.access_ptr"
    assert str(pointer.ty.scope) == "shared.barrier"
    assert str(pointer.ty.element_type.dtype) == "uint64"
    load, _, mask = pointer.args
    assert isinstance(load, ir.TensorLoad)
    assert int(load.indices[0]) == 1
    assert int(_tensor_spec.tensor_offset(load.source)) == 5
    assert [int(stride) for stride in _tensor_spec.tensor_strides(load.source)] == [2]
    assert int(mask) == 3
    assert tirx.analysis.check_well_formed(kernel)


def test_tcgen05_arrive_keeps_explicit_typed_pointer():
    pointer = ir.Var("pointer", ir.PtrType(ir.PrimType("uint64"), "shared.barrier"))
    call = T.tcgen05_mma_arrive(pointer)
    assert call.args[0].same_as(pointer)


def test_wait_accepts_macro_barrier_slot():
    @T.macro
    def wait(slot):
        T.mbarrier_wait_parity(slot, 0)

    @T.prim_func
    def kernel():
        barrier = T.alloc_barrier([1, 1])
        wait(barrier[1])

    (wait_call,) = [call for call in collect_nodes(kernel, ir.Call) if call.op.name == "tl.mbarrier_wait_parity"]
    assert isinstance(wait_call.args[0], ir.TensorLoad)
    assert int(wait_call.args[0].indices[0]) == 1
    assert tirx.analysis.check_well_formed(kernel)
