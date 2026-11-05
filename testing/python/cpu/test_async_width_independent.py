"""Independent effect and nested-vector boundaries for the async-copy producer."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from testing.python.cpu.test_async_width_contract import (
    CopyProgram,
    check,
    cp_calls,
    existing_native_operator_bridge,  # noqa: F401 — imported autouse fixture
    integer,
)


@pytest.mark.parametrize("effect_position", ["source_index", "predicate"])
def test_effectful_operand_retains_each_original_evaluation(effect_position, tmp_path):
    program = CopyProgram()
    if effect_position == "source_index":
        offset = ir.Call("tirx.call_extern", [ir.StringImm("next_copy_offset"), program.offset], ty="int32")
        body = program.copy(source_index=offset + program.lane)
    else:
        predicate = ir.Call("tirx.call_extern", [ir.StringImm("next_copy_predicate"), program.offset], ty="bool")
        body = program.copy(predicate=predicate)
    # The complete native function must remain structurally identical: no
    # per-lane effect may be moved into a single aggregated transaction.
    check(program, [program.loop([body])], tmp_path, expected_count=None)


@pytest.mark.parametrize("enable_vectorize", [False, True])
def test_nested_vector_context_never_multiplies_a_legal_call_past_16_bytes(enable_vectorize, tmp_path):
    program = CopyProgram()
    batch = ir.Var("batch", "int32")
    index = batch * 8 + program.lane
    inner = program.loop([program.copy(source_index=index, shared_index=index)])
    outer = ir.For(batch, integer(0), integer(4), integer(1), ir.ForKind.VECTORIZED, [inner], {})
    injected = check(program, [outer], tmp_path, expected_count=8)
    function = injected.with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(enable_vectorize)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    calls = cp_calls(result)
    assert calls
    for call in calls:
        call.validate()
        assert int(call.args[2]) * 2 in (4, 8, 16)
    (tmp_path / "nested-vector-result.json").write_text(ir.save_json(result))


def test_runtime_alignment_guard_only_observes_pointer_bits(tmp_path):
    program = CopyProgram(source_alignment=1)
    result = check(program, [program.loop()], tmp_path, expected_count=8, explicit=True)
    conditional = result.body[0]
    assert isinstance(conditional, ir.If)
    pointers = []
    tvm_ffi.structural_walk(conditional.condition, (ir.Call, pointers.append))
    accesses = [call for call in pointers if isinstance(call.op, ir.Op) and call.op.name == "tl.access_ptr"]
    assert len(accesses) == 2
    assert all(int(call.args[1]) == 0 and int(call.args[2]) == 0 for call in accesses)
    assert int(result.params[0].ty.data_alignment) == 1
