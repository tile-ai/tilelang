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
