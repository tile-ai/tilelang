"""Loop eligibility metadata retains its lexical position without a hardware epoch."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir, tirx

from test_native_ascend_core_passes import _apply, _costed, _metadata
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_schedule_units import _attributes, _bind_target, _fields
from test_native_reducer_contracts import allocate, annotation, integer, objects, tensor


def _program(stage, tail, *, guarded=False):
    storage = tensor("storage", [1], "int32", "shared.dyn")
    output = tensor("output", [4], "int32", "global")
    index = ir.Var("i", "int32")
    flag = ir.Var("flag", "bool")
    producer = _costed(ir.TensorStore(storage, [integer(0)], index), stage=0)
    read = index if tail is False else ir.TensorLoad(storage, [integer(0)])
    consumer = _costed(ir.TensorStore(output, [index], read), stage=stage)
    body = [ir.If(flag, [producer], None) if guarded else producer, consumer]
    if tail is not None:
        body.append(annotation({"multi_buffer_eligible": [storage] if tail else []}))
    loop = ir.For(
        index,
        integer(0),
        integer(4),
        integer(1),
        ir.ForKind.DEFAULT,
        body,
        {"num_stages": integer(stage + 1), "enable_offset": True, "multi_buffer_eligible": [] if tail is False else [storage]},
    )
    params = [output, flag] if guarded else [output]
    value = _bind_target(
        _kernel(params, [_costed(allocate(storage)), _unit(loop, stage=-1)], {storage: 1 if tail is False else stage + 1}, count=1)
    )
    return value, storage, output


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("tail", [None, False, True], ids=["native-field", "empty-carrier", "storage-carrier"])
def test_trailing_eligibility_preserves_requested_stages_and_definitions(stage, tail):
    value, storage, output = _program(stage, tail)
    result = _apply(transform.AutoSchedule(), value)
    (owner,) = objects(result, ir.For)
    stages = {}
    for unit in _attributes(owner, "tl.schedule_unit"):
        for store in objects(unit, ir.TensorStore):
            stages[store.dest] = int(_fields(unit)["node"]["stage"])
    assert stages[storage] == 0
    assert stages[output] == stage
    if tail is not None:
        assert isinstance(owner.body.seq[-1], ir.Evaluate)
        (eligibility,) = [fields["multi_buffer_eligible"] for fields in _metadata(owner) if "multi_buffer_eligible" in fields]
        assert len(eligibility) == int(tail)
        if tail:
            assert eligibility[0].same_as(storage)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)


@pytest.mark.parametrize("tail", [False, True], ids=["empty-carrier", "storage-carrier"])
def test_trailing_metadata_does_not_release_conditional_stage_boundary(tail):
    value, _, _ = _program(1, tail, guarded=True)
    with pytest.raises(RuntimeError, match="Manual schedule constraints are infeasible"):
        _apply(transform.AutoSchedule(), value)
