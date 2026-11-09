"""The common allocator consumes pairwise compatibility without assuming transitivity."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_packer_respects_complete_allocation_conflicts(disable_reuse):
    a, b, c = [tirx.decl_buffer((size,), "int32", name=name, scope="shared.dyn") for name, size in [("a", 64), ("b", 32), ("c", 32)]]
    # a is compatible with either b or c; b and c must remain disjoint.
    contract = {a.data: [b.data, c.data], b.data: [a.data], c.data: [a.data]}
    body = tirx.SeqStmt(
        [
            *[tirx.AllocBuffer(buffer) for buffer in (a, b, c)],
            *[tirx.BufferStore(buffer, index, [0]) for index, buffer in enumerate((a, b, c))],
        ]
    )
    before = tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt(contract, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(before)
    assert not [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]
    offsets = {int(store.value): int(tvm.arith.Analyzer().simplify(store.indices[0])) for store in nodes(after, tirx.BufferStore)}
    assert offsets[1] + 32 <= offsets[2] or offsets[2] + 32 <= offsets[1]
    if disable_reuse:
        for left, left_size, right, right_size in [(0, 64, 1, 32), (0, 64, 2, 32)]:
            assert offsets[left] + left_size <= offsets[right] or offsets[right] + right_size <= offsets[left]
    else:
        assert any(offsets[0] < offsets[index] + 32 and offsets[index] < offsets[0] + 64 for index in (1, 2))


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_subcore_ub_allocations_have_independent_address_spaces(disable_reuse):
    common, dq, epi = [
        tirx.decl_buffer((size,), "int32", name=name, scope="shared.dyn") for name, size in [("common", 32), ("dq", 128), ("epi", 64)]
    ]
    buffers = (common, dq, epi)
    body = tirx.SeqStmt(
        [
            *[tirx.AllocBuffer(buffer) for buffer in buffers],
            *[tirx.BufferStore(buffer, index, [0]) for index, buffer in enumerate(buffers)],
        ]
    )
    body = tirx.AttrStmt({buffer.data: [] for buffer in buffers}, "tl.buffer_alias_map", 1, body)
    body = tirx.AttrStmt({dq.data: 0, epi.data: 1}, "tl.aiv_storage_map", 1, body)
    thread = tirx.IterVar(tvm.ir.Range(0, 1), tirx.Var("block", "int32"), tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.AttrStmt(thread, "thread_extent", 1, body)
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(tvm.IRModule({"main": tirx.PrimFunc([], body)}))
    offsets = {int(store.value): int(tvm.arith.Analyzer().simplify(store.indices[0])) for store in nodes(after, tirx.BufferStore)}
    assert set(offsets) == {0, 1, 2}
    (allocation,) = nodes(after, tirx.AllocBuffer)
    allocated_bytes = int(tvm.arith.Analyzer().simplify(allocation.buffer.shape[0])) * tvm.DataType(allocation.buffer.dtype).bits // 8
    # Private AIV storage shares numeric addresses, while the common buffer
    # must remain disjoint on both cores. Placement order is not the contract.
    assert allocated_bytes == (32 + max(128, 64)) * 4
    for index, size in enumerate((32, 128, 64)):
        assert offsets[index] >= 0 and (offsets[index] + size) * 4 <= allocated_bytes
        if index:
            assert offsets[0] + 32 <= offsets[index] or offsets[index] + size <= offsets[0]
    assert offsets[1] < offsets[2] + 64 and offsets[2] < offsets[1] + 128


def test_ss_allocations_remain_disjoint_and_aligned_even_with_alias_contract():
    a, b = [tirx.decl_buffer((2,), "uint32", name=name, scope="shared.ss") for name in ("a", "b")]
    contract = {a.data: [b.data], b.data: [a.data]}
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(a),
            tirx.AllocBuffer(b),
            tirx.BufferStore(a, tirx.const(1, "uint32"), [0]),
            tirx.BufferStore(b, tirx.const(2, "uint32"), [0]),
        ]
    )
    body = tirx.AttrStmt(tirx.IterVar(None, tirx.Var("bx", "int32"), tirx.IterVar.ThreadIndex, "blockIdx.x"), "thread_extent", 1, body)
    before = tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt(contract, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=False)(before)
    stores = nodes(after, tirx.BufferStore)
    offsets = [int(tvm.arith.Analyzer().simplify(store.indices[0])) * 4 for store in stores]
    assert sorted(offsets) == [0, 32]
    allocations = nodes(after, tirx.AllocBuffer)
    assert len(allocations) == 1
