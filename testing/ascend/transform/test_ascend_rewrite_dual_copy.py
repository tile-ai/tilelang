"""Dual-copy rewriting splits software regions and preserves hardware copies."""

from tvm_tirx import ir
import tvm_ffi
from testing.ascend._ir import allocate, call_annotations, launch_thread, sblock, tensor

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import tirx
from testing.ascend._ir import calls, nodes


def _copy(src_scope, dst_scope, src_shape, dst_shape, *, split=1, double=False):
    src = tensor(src_shape, "float32", name="src", scope=src_scope)
    dst = tensor(dst_shape, "float32", name="dst", scope=dst_scope)

    def region(buffer, mask):
        return ir.TensorRegion(buffer, [ir.Range.from_min_extent(0, extent) for extent in buffer.ty.shape], ir.TensorRegionType())

    annotations = {"dual_dst_ctl": ir.IntImm("int32", split), "unit_flag_ctrl": ir.IntImm("int32", 3)}
    if double:
        annotations["double"] = ir.IntImm("int32", 1)
    return ir.Evaluate(
        tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
            ir.PrimType("void"), tvm_tirx.ir.Op.get("tl.tileop.ascend_copy"), [region(src, 1), region(dst, 2)], annotations, ir.UnknownLoc()
        )
    )


def _module(body):
    used = list(dict.fromkeys(var for var in nodes(body, ir.Var) if tirx.is_tensor_var(var)))
    params = [var for var in used if str(var.ty.scope) == "global"]
    local = [allocate(var) for var in used if str(var.ty.scope) != "global"]
    return tvm_tirx.IRModule({"main": tirx.Function(params, [*local, body])})


def _cthread(body, extent=2):
    sid = ir.Var("sid", "int32")
    thread = sid
    return launch_thread(thread, "cthread", extent, body)


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
    sid = body.body_params[0]
    if manual:
        body = sblock("VECTOR", body)
    before = _module(body)
    with tvm_tirx.transform.PassContext(config={"tl.enable_auto_schedule": not manual}):
        after = transform.RewriteDualCopy()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    full, half = (copy.args[1], copy.args[0]) if double else (copy.args[0], copy.args[1])
    extent = half.region[axis].extent
    expected_index = sid * extent
    tvm_tirx.ir.assert_structural_equal(full.region[axis].begin, expected_index)
    tvm_tirx.ir.assert_structural_equal(full.region[axis].extent, extent)
    assert all(int(axis_region.begin) == 0 for dimension, axis_region in enumerate(full.region) if dimension != axis)
    assert not any(key in call_annotations(copy) for key in ("double", "dual_dst_ctl", "unit_flag_ctrl"))


def test_automatic_copy_synthesizes_one_subcore_scope():
    copy = _copy("global", "shared.dyn", (128,), (64,))
    root = launch_thread(ir.Var("block", "int32"), "blockIdx.x", 1, [copy, copy])
    after = transform.RewriteDualCopy()(_module(root))
    scopes = [node for node in nodes(after, ir.RegionStmt) if node.op.name == "tirx.launch_thread" and node.args[0].value == "cthread"]
    assert len(scopes) == 1 and int(scopes[0].args[1]) == 2
    assert len(calls(scopes[0].body, "tl.tileop.ascend_copy")) == 2


def test_hardware_copy_retains_dual_destination_and_unit_flags():
    before = _module(_copy("shared.l0c", "shared.dyn", (64, 64), (32, 64)))
    after = transform.RewriteDualCopy()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


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
    with tvm_tirx.transform.PassContext(config={"tl.enable_auto_schedule": False}), pytest.raises(ValueError, match=r"explicit T\.Vector"):
        transform.RewriteDualCopy()(_module(body))
