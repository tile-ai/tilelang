"""Implicit scalar completion follows physical stages, including nested readers."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir

from testing.ascend._ir import annotation, annotations, calls, kernel, serial, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize("versions", [2, 3])
@pytest.mark.parametrize("nested_reader", [False, True])
@pytest.mark.parametrize("staggered", [False, True])
def test_unmaterialized_scalar_dependencies_do_not_shorten_physical_lifetimes(versions, nested_reader, staggered):
    x, y, out_x, out_y = [tensor((8,), "int32", name=name) for name in ("x", "y", "out_x", "out_y")]
    a, b = [tensor((8,), "int32", name=name, scope="shared.dyn") for name in ("a", "b")]
    k, j = ir.Var("k", "int32"), ir.Var("j", "int32")
    reader_stage, b_stage = (1, 4) if staggered else (0, 0)
    reader = ir.TensorStore(out_x, [k], ir.TensorLoad(a, [0]))
    if nested_reader:
        reader = unit(serial(j, 0, 1, [unit(reader)]), core=None, stage=reader_stage)
    else:
        reader = unit(reader, stage=reader_stage)
    body = [
        unit(ir.TensorStore(a, [0], ir.TensorLoad(x, [k]))),
        reader,
        unit(ir.TensorStore(b, [0], ir.TensorLoad(y, [k])), stage=b_stage),
        unit(ir.TensorStore(out_y, [k], ir.TensorLoad(b, [0])), stage=b_stage),
    ]
    loop = ir.For(
        k,
        ir.const(0),
        ir.const(8),
        ir.const(1),
        ir.ForKind.DEFAULT,
        [*body, annotation({"multi_buffer_eligible": [a, b]})],
        annotations={"num_stages": ir.const(5 if staggered else 1), "enable_offset": ir.const(staggered)},
    )
    before = kernel(
        unit(loop, core=None),
        buffers=[a, b],
        params=[x, y, out_x, out_y],
        annotations={"tl.buffer_versions_map": {a: versions}},
        allocation_stage=0,
        allocation_core=5,
    )
    assert "tirx.noalias" not in before["main"].attrs
    frozen = ir.save_json(before)
    after = transform.InsertSync()(before)
    assert ir.save_json(before) == frozen
    for operation in ("tl.ascend_set_flag", "tl.ascend_wait_flag", "tl.ascend_pipe_barrier"):
        assert not calls(after, operation)
    aliases = {}
    for fields in annotations(after):
        aliases.update(fields.get("tl.buffer_alias_map", {}))
    can_alias = any(peer.same_as(b) for peer in aliases.get(a, []))
    # With staggered stages, a(6) has been written but its stage-1 reader
    # has not run when b(2) writes. With equal stages each reader completes
    # before the next allocation is written on the scalar pipe.
    assert can_alias == (not staggered)
