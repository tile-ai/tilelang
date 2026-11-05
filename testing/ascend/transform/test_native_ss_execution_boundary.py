"""Statement metadata is preserved without becoming executable SS storage."""

import json

import pytest
import tvm_ffi
from testing.ascend._ir import (
    allocate,
    allocated_tensors,
    kernel,
    nodes,
    sblocks,
    tensor,
)
from testing.ascend.auto_schedule._scheduled_ir import unit
from tilelang.ascend import transform

from tvm_tirx import ir, tirx


def with_template(region, template):
    attrs = dict(region.attrs.items())
    attrs["pragma_ss_template"] = template
    return ir.RegionStmt(region.op, region.args, region.body_params, attrs, region.body, region.result_vars)


def snapshot(value, path):
    path.with_suffix(".json").write_text(ir.save_json(value))
    try:
        path.with_suffix(".py").write_text(value.script())
    except RuntimeError as error:
        # The native printer cannot represent statement-valued dictionaries;
        # its limitation does not affect JSON serialization or verification.
        path.with_suffix(".printer-error.txt").write_text(str(error))


def checked_lower(before, tmp_path):
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    snapshot(before, tmp_path / "before")
    after = transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    snapshot(after, tmp_path / "after")
    return after


@pytest.mark.parametrize("scope", ["shared.ss", "shared.ss.dyn"])
def test_native_region_retains_its_statement_template_exactly(scope, tmp_path):
    live = tensor((2,), "uint32", name="live", scope=scope, align=32)
    hidden = tensor((2,), "uint32", name="template_only", scope=scope, align=32)
    template = ir.SeqStmt([allocate(hidden), ir.TensorStore(hidden, [1], ir.const(73, "uint32"))])
    template_json = ir.save_json(template)
    carried = with_template(
        ir.RegionStmt("tirx.device_scope", [], [], {}, [ir.TensorStore(live, [0], ir.const(19, "uint32"))]),
        template,
    )
    before = kernel([unit(carried, core=1)], buffers=[live], allocation_stage=0)
    after = checked_lower(before, tmp_path)
    (root,) = sblocks(after, "tilelang_root")
    prefix = [stmt.var for stmt in root.body.seq if isinstance(stmt, ir.Bind)]
    templates = [region.attrs["pragma_ss_template"] for region in nodes(root, ir.RegionStmt) if "pragma_ss_template" in region.attrs]
    (tmp_path / "observed.json").write_text(
        json.dumps(
            {
                "prefix": [str(var.name) for var in prefix],
                "template_count": len(templates),
                "template_json_unchanged": [ir.save_json(value) == template_json for value in templates],
            },
            indent=2,
        )
        + "\n"
    )
    assert len(prefix) == 1 and prefix[0].same_as(live)
    assert len(templates) == 1
    assert ir.save_json(templates[0]) == template_json
    assert ir.save_json(template) == template_json


@pytest.mark.parametrize("scope", ["shared.ss", "shared.ss.dyn"])
def test_unexecuted_template_cannot_consume_ss_pool_capacity(scope, tmp_path):
    live = tensor((512,), "uint32", name="live", scope=scope, align=32)
    hidden = tensor((1024,), "uint32", name="template_only", scope=scope, align=32)
    marked = unit(ir.TensorStore(live, [0], ir.const(19, "uint32")), core=1)
    before = kernel([with_template(marked, allocate(hidden))], buffers=[live], allocation_stage=0)
    after = checked_lower(before, tmp_path)
    analyzed = transform.InferBufferAliases()(after)
    pooled = transform.MergeUBAllocations()(analyzed)
    assert tirx.analysis.check_well_formed(pooled)
    snapshot(pooled, tmp_path / "pooled")
    arenas = [var for var in allocated_tensors(pooled) if var.ty.scope.name in ("shared.ss", "shared.ss.dyn")]
    sizes = [int(var.ty.dtype.dtype.bits * var.ty.dtype.dtype.lanes // 8) for var in arenas]
    for index, arena in enumerate(arenas):
        for extent in arena.ty.shape:
            sizes[index] *= int(extent)
    (tmp_path / "observed.json").write_text(json.dumps({"arena_sizes_bytes": sizes, "architectural_capacity_bytes": 3072}, indent=2) + "\n")
    device = ir.IRModule({"main": pooled["main"].with_attr("calling_conv", int(ir.CallingConv.DEVICE_KERNEL_LAUNCH))})
    snapshot(device, tmp_path / "device")
    emitted = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(device, device["main"].attrs["target"])
    source = emitted.inspect_source()
    (tmp_path / "kernel.cc").write_text(source)
    assert source.count("(volatile __ssbuf__ uint32_t *)0") == 1
    assert sizes == [2048]
    assert sum(sizes) <= 3072
