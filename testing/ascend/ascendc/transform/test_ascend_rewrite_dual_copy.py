"""Dual-copy rewriting splits software regions and preserves hardware copies."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, nodes


def _copy(src_scope, dst_scope, src_shape, dst_shape, *, split=1, double=False):
    src = tirx.decl_buffer(src_shape, "float32", name="src", scope=src_scope)
    dst = tirx.decl_buffer(dst_shape, "float32", name="dst", scope=dst_scope)

    def region(buffer, mask):
        return tirx.Call("handle", tvm.ir.Op.get("tl.region"), [buffer[tuple(0 for _ in buffer.shape)], mask, *buffer.shape])

    annotations = {"dual_dst_ctl": tirx.IntImm("int32", split), "unit_flag_ctrl": tirx.IntImm("int32", 3)}
    if double:
        annotations["double"] = tirx.IntImm("int32", 1)
    return tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.ascend_copy"), [region(src, 1), region(dst, 2)], annotations))


def _module(body):
    return tvm.IRModule({"main": tirx.PrimFunc([], body)})


def _cthread(body, extent=2):
    sid = tirx.Var("sid", "int32")
    thread = tirx.IterVar(tvm.ir.Range(0, extent), sid, tirx.IterVar.ThreadIndex, "cthread")
    return tirx.AttrStmt(thread, "thread_extent", extent, body)


@pytest.mark.parametrize(
    "src_scope, dst_scope, src_shape, dst_shape, double, split, axis",
    [
        ("global", "shared.dyn", (128,), (64,), False, 1, 0),
        ("shared.dyn", "global", (64,), (128,), True, 1, 0),
        ("shared.dyn", "shared.l1", (64,), (128,), True, 1, 0),
        ("global", "shared.dyn", (64, 64), (32, 64), False, 1, 0),
        ("shared.dyn", "global", (64, 32), (64, 64), True, 2, 1),
    ],
    ids=["gm-to-ub-1d", "ub-to-gm-1d", "ub-to-l1-1d", "split-m", "split-n"],
)
@pytest.mark.parametrize("manual", [False, True], ids=["automatic", "manual-vector"])
def test_software_copy_splits_the_full_region(src_scope, dst_scope, src_shape, dst_shape, double, split, axis, manual):
    body = _cthread(_copy(src_scope, dst_scope, src_shape, dst_shape, split=split, double=double))
    sid = body.node.var
    if manual:
        body = tirx.SBlock([], [], [], "VECTOR", body)
    before = _module(body)
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": not manual}):
        after = transform.RewriteDualCopy()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    full, half = (copy.args[1], copy.args[0]) if double else (copy.args[0], copy.args[1])
    extent = half.args[axis + 2]
    expected_index = sid * extent
    tvm.ir.assert_structural_equal(full.args[0].indices[axis], expected_index)
    tvm.ir.assert_structural_equal(full.args[axis + 2], extent)
    assert all(int(index) == 0 for dimension, index in enumerate(full.args[0].indices) if dimension != axis)
    assert not any(key in copy.annotations for key in ("double", "dual_dst_ctl", "unit_flag_ctrl"))


def test_automatic_copy_synthesizes_one_subcore_scope():
    copy = _copy("global", "shared.dyn", (128,), (64,))
    root = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "tilelang_root", tirx.SeqStmt([copy, copy])))
    after = transform.RewriteDualCopy()(_module(root))
    scopes = [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "thread_extent"]
    assert len(scopes) == 1 and int(scopes[0].value) == 2
    assert len(calls(scopes[0].body, "tl.tileop.ascend_copy")) == 2


def test_hardware_copy_retains_dual_destination_and_unit_flags():
    before = _module(_copy("shared.l0c", "shared.dyn", (64, 64), (32, 64)))
    after = transform.RewriteDualCopy()(before)
    tvm.ir.assert_structural_equal(after, before)


@pytest.mark.parametrize(
    "src_scope, dst_scope, src_shape, dst_shape, split, message",
    [
        ("global", "shared.l1", (64, 64), (32, 64), 1, "supports only L0C->UB"),
        ("shared.l0c", "shared.dyn", (127, 64), (63, 64), 1, "exact 2:1 extent ratio"),
        ("shared.l0c", "shared.dyn", (64, 48), (64, 24), 2, "multiple of 32"),
        ("shared.l0c", "shared.dyn", (128,), (64,), 1, "at least two-dimensional"),
        ("global", "shared.dyn", (127,), (64,), 1, "exact 2:1 extent ratio"),
    ],
    ids=["unsupported-path", "odd-m", "unaligned-n", "hardware-1d", "odd-software-region"],
)
def test_invalid_copy_regions(src_scope, dst_scope, src_shape, dst_shape, split, message):
    before = _module(_cthread(_copy(src_scope, dst_scope, src_shape, dst_shape, split=split)))
    with pytest.raises(ValueError, match=message):
        transform.RewriteDualCopy()(before)


@pytest.mark.parametrize("mixed", [False, True], ids=["unscoped", "mixed-kernel"])
def test_manual_copy_requires_vector_scope(mixed):
    body = _copy("global", "shared.dyn", (128,), (64,))
    if mixed:
        body = _cthread(body)
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}), pytest.raises(ValueError, match=r"explicit T\.Vector"):
        transform.RewriteDualCopy()(_module(body))
