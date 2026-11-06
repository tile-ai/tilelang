"""Explicit unrolling expands host-side loops and preserves vector-function loops."""

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import nodes, sblock, execution_blocks, tensor


def _loop(body, *, kind=ir.ForKind.UNROLLED, explicit=True, trip_count=2, step=None):
    var = ir.Var("i", "int32")
    annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    step = ir.const(1) if step is None else step
    # The old explicit-unroll fixture's extent counted body copies, including
    # non-unit steps. Native For instead carries the exclusive upper bound.
    return ir.For(var, ir.const(0), trip_count * step, step, kind, body(var), annotations=annotations)


def _apply(body, output):
    before = tvm_tirx.IRModule({"main": tirx.Function([output], body)})
    assert tirx.analysis.check_well_formed(before["main"])
    after = transform.UnrollLoopSkipVF()(before)
    assert tirx.analysis.check_well_formed(after["main"])
    return after


@pytest.mark.parametrize(
    "kind,explicit,block,expanded",
    [
        (ir.ForKind.UNROLLED, True, None, True),
        (ir.ForKind.UNROLLED, False, None, False),
        (ir.ForKind.UNROLLED, None, None, False),
        (ir.ForKind.DEFAULT, True, None, False),
        (ir.ForKind.UNROLLED, True, "ordinary", True),
        (ir.ForKind.UNROLLED, True, "SIMD_VF", False),
        (ir.ForKind.UNROLLED, True, "SIMT_VF", False),
    ],
)
def test_only_explicit_unrolled_loops_outside_vf_expand(kind, explicit, block, expanded):
    out = tensor((2,), "int32", name="out")
    loop = _loop(lambda i: ir.TensorStore(out, [i], i), kind=kind, explicit=explicit)
    before = sblock(block, loop) if block else loop
    after = _apply(before, out)
    if expanded:
        assert not nodes(after, ir.For)
        assert [int(store.value) for store in nodes(after, ir.TensorStore)] == [0, 1]
    else:
        (retained,) = nodes(after, ir.For)
        tvm_tirx.ir.assert_structural_equal(retained, loop)


def test_explicit_inner_loop_expands_inside_a_retained_loop():
    out = tensor((2,), "int32", name="out")
    inner = _loop(lambda i: ir.TensorStore(out, [i], i))
    outer = _loop(lambda _: inner, explicit=False)
    after = _apply(outer, out)
    (retained,) = nodes(after, ir.For)
    assert retained.var.same_as(outer.var)
    assert [int(store.value) for store in nodes(after, ir.TensorStore)] == [0, 1]


def test_non_unit_step_values_are_preserved():
    out = tensor((8,), "int32", name="out")
    before = _loop(lambda i: ir.TensorStore(out, [i], i), trip_count=4, step=ir.IntImm("int32", 2))
    after = _apply(before, out)
    stores = nodes(after, ir.TensorStore)
    assert [int(store.value) for store in stores] == [0, 2, 4, 6]
    assert [int(store.indices[0]) for store in stores] == [0, 2, 4, 6]


def test_expansion_around_vf_preserves_its_inner_loop():
    out = tensor((2,), "int32", name="out")
    inner = _loop(lambda i: ir.TensorStore(out, [i], i))
    vf = sblock("SIMD_VF", inner)
    after = _apply(_loop(lambda _: vf), out)
    blocks = execution_blocks(after, "SIMD_VF")
    assert len(blocks) == 2
    for block in blocks:
        tvm_tirx.ir.assert_structural_equal(block.body, ir.SeqStmt([inner]), map_free_vars=True)
    assert len(nodes(after, ir.For)) == 2
