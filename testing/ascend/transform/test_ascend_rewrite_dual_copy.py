"""Dual-copy rewriting splits software regions and preserves hardware copies."""

import pytest
from tvm_tirx import ir
from tilelang import tvm
from tilelang.ascend import transform
from tilelang._compat import tirx
from testing.ascend._ir import calls, nodes, kernel


def _copy(src_scope, dst_scope, src_shape, dst_shape, *, split=1, double=False):
    src = tirx.decl_buffer(src_shape, "float32", name="src", scope=src_scope)
    dst = tirx.decl_buffer(dst_shape, "float32", name="dst", scope=dst_scope)

    def region(buffer, mask):
        return ir.TensorRegion(buffer, [ir.Range(0, extent) for extent in buffer.shape], ir.TensorRegionType())

    annotations = {"dual_dst_ctl": tirx.IntImm("int32", split), "unit_flag_ctrl": tirx.IntImm("int32", 3)}
    if double:
        annotations["double"] = tirx.IntImm("int32", 1)
    return ir.Evaluate(ir.Call("tl.tileop.ascend_copy", [region(src, 1), region(dst, 2)], attrs=annotations))


def _module(body):
    tensors = {region.source for region in nodes(body, ir.TensorRegion)}
    return kernel(
        body,
        buffers=[tensor for tensor in tensors if tensor.scope().name != "global"],
        params=[tensor for tensor in tensors if tensor.scope().name == "global"],
    )


def _cthread(body, extent=2):
    sid = tirx.Var("sid", "int32")
    return ir.RegionStmt("tl.launch_grid", ["cthread", extent], [sid], {}, body)


def _region(expression):
    if isinstance(expression, ir.TensorRegion):
        return expression
    assert expression.op.name == "tl.region"
    load = expression.args[0]
    return ir.TensorRegion(
        load.source,
        [ir.Range.from_min_extent(index, extent) for index, extent in zip(load.indices, expression.args[2:], strict=True)],
        ir.TensorRegionType(),
    )


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
        body = ir.RegionStmt("tl.scope", [], [], {"name_hint": "VECTOR"}, body)
    before = _module(body)
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": not manual}):
        after = transform.RewriteDualCopy()(before)
    (copy,) = calls(after, "tl.tileop.ascend_copy")
    full, half = (copy.args[1], copy.args[0]) if double else (copy.args[0], copy.args[1])
    full, half = _region(full), _region(half)
    extent = half.region[axis].extent
    expected_index = sid * extent
    tvm.ir.assert_structural_equal(full.region[axis].begin, expected_index)
    tvm.ir.assert_structural_equal(full.region[axis].extent, extent)
    assert all(int(region.begin) == 0 for dimension, region in enumerate(full.region) if dimension != axis)
    assert not any(key in copy.attrs for key in ("double", "dual_dst_ctl", "unit_flag_ctrl"))


def test_automatic_copy_synthesizes_one_subcore_scope():
    copy = _copy("global", "shared.dyn", (128,), (64,))
    root = ir.SeqStmt([copy, copy])
    after = transform.RewriteDualCopy()(_module(root))
    scopes = [node for node in nodes(after, ir.RegionStmt) if node.op.name == "tl.launch_grid" and str(node.args[0].value) == "cthread"]
    assert len(scopes) == 1 and int(scopes[0].args[1]) == 2
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
