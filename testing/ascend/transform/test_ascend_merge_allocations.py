"""The common allocator consumes pairwise compatibility without assuming transitivity."""

from tvm_tirx import ir
from testing.ascend._ir import allocate, allocations, attribute, attribute_key, launch_thread, tensor

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import tirx
from testing.ascend._ir import nodes
from testing.ascend._memory_ir import element_offset


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_packer_respects_complete_allocation_conflicts(disable_reuse):
    a, b, c = [tensor((size,), "int32", name=name, scope="shared.dyn") for name, size in [("a", 64), ("b", 32), ("c", 32)]]
    # a is compatible with either b or c; b and c must remain disjoint.
    contract = {a: [b, c], b: [a], c: [a]}
    body = ir.SeqStmt(
        [
            *[allocate(buffer) for buffer in (a, b, c)],
            *[ir.TensorStore(buffer, [0], index) for index, buffer in enumerate((a, b, c))],
        ]
    )
    before = tvm_tirx.IRModule({"main": tirx.Function([], attribute(contract, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(before)
    assert not [node for node in nodes(after, ir.RegionStmt) if attribute_key(node) == "tl.buffer_alias_map"]
    offsets = {int(store.value): int(element_offset(after, store)) for store in nodes(after, ir.TensorStore)}
    assert offsets[1] + 32 <= offsets[2] or offsets[2] + 32 <= offsets[1]
    if disable_reuse:
        for left, left_size, right, right_size in [(0, 64, 1, 32), (0, 64, 2, 32)]:
            assert offsets[left] + left_size <= offsets[right] or offsets[right] + right_size <= offsets[left]
    else:
        assert any(offsets[0] < offsets[index] + 32 and offsets[index] < offsets[0] + 64 for index in (1, 2))


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_subcore_ub_allocations_have_independent_address_spaces(disable_reuse):
    common, dq, epi = [tensor((size,), "int32", name=name, scope="shared.dyn") for name, size in [("common", 32), ("dq", 128), ("epi", 64)]]
    buffers = (common, dq, epi)
    body = ir.SeqStmt(
        [
            *[allocate(buffer) for buffer in buffers],
            *[ir.TensorStore(buffer, [0], index) for index, buffer in enumerate(buffers)],
        ]
    )
    body = attribute({buffer: [] for buffer in buffers}, "tl.buffer_alias_map", 1, body)
    body = attribute({dq: 0, epi: 1}, "tl.aiv_storage_map", 1, body)
    thread = ir.Var("block", "int32")
    body = launch_thread(thread, "blockIdx.x", 1, body)
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(tvm_tirx.IRModule({"main": tirx.Function([], body)}))
    offsets = {int(store.value): int(element_offset(after, store)) for store in nodes(after, ir.TensorStore)}
    assert set(offsets) == {0, 1, 2}
    (allocation,) = allocations(after)
    allocated_bytes = int(tvm_tirx.sym.Analyzer().simplify(allocation.var.ty.shape[0])) * allocation.var.ty.dtype.dtype.bits // 8
    # Private AIV storage shares numeric addresses, while the common buffer
    # must remain disjoint on both cores. Placement order is not the contract.
    assert allocated_bytes == (32 + max(128, 64)) * 4
    for index, size in enumerate((32, 128, 64)):
        assert offsets[index] >= 0 and (offsets[index] + size) * 4 <= allocated_bytes
        if index:
            assert offsets[0] + 32 <= offsets[index] or offsets[index] + size <= offsets[0]
    assert offsets[1] < offsets[2] + 64 and offsets[2] < offsets[1] + 128


def test_ss_allocations_remain_disjoint_and_aligned_even_with_alias_contract():
    # The original sequential allocator used only the SS scope's 32-byte
    # alignment. Encode that raw-IR precondition explicitly in the native type;
    # the native default of 64 bytes is covered independently below.
    a, b = [tensor((2,), "uint32", name=name, scope="shared.ss", align=32) for name in ("a", "b")]
    contract = {a: [b], b: [a]}
    body = ir.SeqStmt(
        [
            allocate(a),
            allocate(b),
            ir.TensorStore(a, [0], ir.const(1, "uint32")),
            ir.TensorStore(b, [0], ir.const(2, "uint32")),
        ]
    )
    body = launch_thread(ir.Var("bx", "int32"), "blockIdx.x", 1, body)
    before = tvm_tirx.IRModule({"main": tirx.Function([], attribute(contract, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=False)(before)
    stores = nodes(after, ir.TensorStore)
    offsets = [int(element_offset(after, store)) * 4 for store in stores]
    assert sorted(offsets) == [0, 32]
    bindings = allocations(after)
    assert len(bindings) == 1


def test_ss_native_default_alignment_is_preserved_by_merge():
    a, b = [tensor((2,), "uint32", name=name, scope="shared.ss") for name in ("a", "b")]
    assert a.ty.data_alignment == b.ty.data_alignment == 64
    body = ir.SeqStmt(
        [
            allocate(a),
            allocate(b),
            ir.TensorStore(a, [0], ir.const(1, "uint32")),
            ir.TensorStore(b, [0], ir.const(2, "uint32")),
        ]
    )
    body = launch_thread(ir.Var("bx", "int32"), "blockIdx.x", 1, body)
    before = tvm_tirx.IRModule({"main": tirx.Function([], attribute({a: [b], b: [a]}, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=False)(before)
    offsets = [int(element_offset(after, store)) * 4 for store in nodes(after, ir.TensorStore)]
    assert sorted(offsets) == [0, 64]
    (arena,) = allocations(after)
    assert arena.var.ty.data_alignment == 64
