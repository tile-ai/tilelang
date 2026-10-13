"""The reuse contract survives scheduled lowering and is consumed by packing."""

import tilelang
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import nodes
from testing.ascend.auto_schedule._reuse_utils import _make_cyclic_scalar_program


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @tvm.ir.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert any("tl.buffer_alias_map" in block.annotations for block in nodes(snapshots["tl.InsertSync"], tirx.SBlock))
    assert any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.LowerScheduledTIR"], tirx.AttrStmt))
    assert not any(node.attr_key == "tl.buffer_alias_map" for node in nodes(snapshots["tl.MergeUBAllocations"], tirx.AttrStmt))
