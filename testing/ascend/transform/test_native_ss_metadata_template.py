"""A statement carried by Region metadata is not executed storage."""

import json

from testing.ascend._ir import allocate, kernel, kernel_body, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit
from tilelang.ascend import transform
from tvm_tirx import ir, tirx


def test_region_attrs_statement_template_is_not_an_executed_ss_owner(tmp_path):
    live = tensor((2,), "uint32", name="live", scope="shared.ss", align=32)
    template_only = tensor((2,), "uint32", name="template_only", scope="shared.ss", align=32)
    marked = unit(ir.TensorStore(live, [0], ir.const(19, "uint32")), core=1)
    fields = dict(marked.attrs.items())
    fields["pragma_ss_template"] = allocate(template_only)
    metadata = ir.RegionStmt(marked.op, marked.args, marked.body_params, fields, marked.body, marked.result_vars)
    before = kernel([metadata], buffers=[live], allocation_stage=0)
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    (tmp_path / "before.json").write_text(saved)
    try:
        (tmp_path / "before.py").write_text(before.script())
    except RuntimeError as error:
        (tmp_path / "before-printer-error.txt").write_text(str(error))
    after = transform.LowerScheduledTIR()(before)
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    (tmp_path / "after.json").write_text(ir.save_json(after))
    (tmp_path / "after.py").write_text(after.script())
    root = kernel_body(after)
    executed = [stmt.var for stmt in root.seq if isinstance(stmt, ir.Bind)]
    (tmp_path / "observed.json").write_text(json.dumps({"executed_prefix_owners": [str(var.name) for var in executed]}, indent=2) + "\n")
    assert len(executed) == 1 and executed[0].same_as(live)
