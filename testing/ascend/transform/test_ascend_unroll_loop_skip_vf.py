"""Explicit unrolling expands host-side loops and preserves vector-function loops."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _loop(body, *, kind=tirx.ForKind.UNROLLED, explicit=True, extent=2, step=None):
    var = tirx.Var("i", "int32")
    annotations = {} if explicit is None else {"pragma_unroll_explicit": explicit}
    return tirx.For(var, 0, extent, kind, body(var), annotations=annotations, step=step)


def _apply(body, output):
    before = tvm.IRModule({"main": tirx.PrimFunc([output.data], body, buffer_map={output.data: output})})
    return transform.UnrollLoopSkipVF()(before)


@pytest.mark.parametrize(
    "kind,explicit,block,expanded",
    [
        (tirx.ForKind.UNROLLED, True, None, True),
        (tirx.ForKind.UNROLLED, False, None, False),
        (tirx.ForKind.UNROLLED, None, None, False),
        (tirx.ForKind.SERIAL, True, None, False),
        (tirx.ForKind.UNROLLED, True, "ordinary", True),
        (tirx.ForKind.UNROLLED, True, "SIMD_VF", False),
        (tirx.ForKind.UNROLLED, True, "SIMT_VF", False),
    ],
)
def test_only_explicit_unrolled_loops_outside_vf_expand(kind, explicit, block, expanded):
    out = tirx.decl_buffer((2,), "int32", name="out")
    loop = _loop(lambda i: tirx.BufferStore(out, i, [i]), kind=kind, explicit=explicit)
    before = tirx.SBlock([], [], [], block, loop) if block else loop
    after = _apply(before, out)
    if expanded:
        assert not nodes(after, tirx.For)
        assert [int(store.value) for store in nodes(after, tirx.BufferStore)] == [0, 1]
    else:
        (retained,) = nodes(after, tirx.For)
        tvm.ir.assert_structural_equal(retained, loop)


def test_explicit_inner_loop_expands_inside_a_retained_loop():
    out = tirx.decl_buffer((2,), "int32", name="out")
    inner = _loop(lambda i: tirx.BufferStore(out, i, [i]))
    outer = _loop(lambda _: inner, explicit=False)
    after = _apply(outer, out)
    (retained,) = nodes(after, tirx.For)
    assert retained.loop_var.same_as(outer.loop_var)
    assert [int(store.value) for store in nodes(after, tirx.BufferStore)] == [0, 1]


def test_non_unit_step_values_are_preserved():
    out = tirx.decl_buffer((8,), "int32", name="out")
    before = _loop(lambda i: tirx.BufferStore(out, i, [i]), extent=4, step=tirx.IntImm("int32", 2))
    after = _apply(before, out)
    stores = nodes(after, tirx.BufferStore)
    assert [int(store.value) for store in stores] == [0, 2, 4, 6]
    assert [int(store.indices[0]) for store in stores] == [0, 2, 4, 6]


def test_expansion_around_vf_preserves_its_inner_loop():
    out = tirx.decl_buffer((2,), "int32", name="out")
    inner = _loop(lambda i: tirx.BufferStore(out, i, [i]))
    vf = tirx.SBlock([], [], [], "SIMD_VF", inner)
    after = _apply(_loop(lambda _: vf), out)
    blocks = [b for b in nodes(after, tirx.SBlock) if b.name_hint == "SIMD_VF"]
    assert len(blocks) == 2
    for block in blocks:
        tvm.ir.assert_structural_equal(block.body, inner, map_free_vars=True)
    assert len(nodes(after, tirx.For)) == 2
