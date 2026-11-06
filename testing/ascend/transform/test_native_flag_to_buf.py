"""Flag compaction preserves ranges, lexical proofs, metadata and mutex ownership."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx


def integer(value):
    return ir.IntImm("int32", value)


def intrinsic(name, *args, metadata=None):
    attrs = None
    args = list(args)
    if metadata is not None:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(metadata)
        attrs = {"schema_version": 1, "operand_count": len(args), "annotation_schema": schema}
        args.append(payload)
    return ir.Call("tl.ascend_" + name, args, ty_args=[ir.PrimType("void")], attrs=attrs)


def flag(kind, event, index, metadata=None):
    return ir.Evaluate(
        intrinsic(kind + "_flag", ir.StringImm(event), integer(index) if isinstance(index, int) else index, metadata=metadata)
    )


def calls(value, name):
    result = []
    tvm_ffi.structural_walk(
        value, (ir.Call, lambda call: result.append(call) if isinstance(call.op, ir.Op) and call.op.name == name else None)
    )
    return result


def apply(body, params=()):
    before = tirx.Function(list(params), body)
    assert tirx.analysis.check_well_formed(before)
    saved = ir.save_json(before)
    transform = tvm_ffi.get_global_func("tl.transform.RewriteFlagToBuf")()
    after = transform(ir.IRModule({"main": before}))["main"]
    assert ir.save_json(before) == saved
    assert tirx.analysis.check_well_formed(after)
    all_calls = []
    tvm_ffi.structural_walk(after, (ir.Call, all_calls.append))
    for call in all_calls:
        call.validate()
    ir.assert_structural_equal(transform(ir.IRModule({"main": after}))["main"], after)
    return after


def test_sparse_slots_compact_without_spilling():
    after = apply([flag(kind, "MTE2_V", index) for index in (100, 102) for kind in ("set", "wait")])
    assert [int(call.args[1].value) for call in calls(after, "tl.ascend_set_flag")] == [0, 1]
    assert [int(call.args[1].value) for call in calls(after, "tl.ascend_wait_flag")] == [0, 1]
    assert not calls(after, "tl.ascend_get_buf")


def test_ninth_sync_point_spills_with_matching_pipe_modes_and_metadata():
    tag = ir.Var("tag", "int32")
    after = apply([flag(kind, "MTE2_V", index, {"trace": tag}) for index in range(9) for kind in ("set", "wait")], [tag])
    assert len(calls(after, "tl.ascend_set_flag")) == 8
    locks = calls(after, "tl.ascend_get_buf")
    unlocks = calls(after, "tl.ascend_rls_buf")
    assert len(locks) == len(unlocks) == 2
    for pair in (locks, unlocks):
        assert [call.args[0].value for call in pair] == ["PIPE_MTE2", "PIPE_V"]
        assert [int(call.args[1].value) for call in pair] == [0, 0]
        assert [int(call.args[2].value) for call in pair] == [1, 0]
        for call in pair:
            assert int(call.attrs["operand_count"]) == 3
            decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"])
            assert decoded["trace"].same_as(tag)


def test_indivisible_dynamic_blocks_use_knapsack():
    i = ir.Var("i", "int32")
    id_a, id_b = i % 5 + 30, i % 4 + 10
    body = [flag(kind, "MTE2_V", value) for value in (id_a, id_b) for kind in ("set", "wait")]
    after = apply([ir.For(i, integer(0), integer(20), integer(1), ir.ForKind.DEFAULT, body)])
    (kept,) = calls(after, "tl.ascend_set_flag")
    (spilled_set, spilled_wait) = calls(after, "tl.ascend_get_buf")
    analyzer = __import__("tvm_tirx", fromlist=["sym"]).sym.Analyzer()
    assert analyzer.can_prove_equal(kept.args[1], i % 5)
    assert analyzer.can_prove_equal(spilled_set.args[1], i % 4)
    ir.assert_structural_equal(spilled_set.args[1], spilled_wait.args[1])


def test_gemm_reservation_is_respected_across_event_pairs():
    pointers = [ir.Var(name, ir.PtrType(ir.PrimType("float16"))) for name in ("c", "a", "b")]
    gemm = ir.Evaluate(
        intrinsic("gemm_l1", *pointers, *[integer(16)] * 4, integer(0), integer(1), ir.StringImm("float16"), integer(5), integer(0))
    )
    after = apply([gemm, *[flag("set", event, index) for event in ("MTE2_V", "V_MTE3") for index in range(9)]], pointers)
    assert [int(call.args[1].value) for call in calls(after, "tl.ascend_get_buf")] == [7, 8]


def test_flat_bind_and_branch_constraints_are_replayed_at_each_shared_call():
    i = ir.Var("i", "int32")
    alias = ir.Var("alias", "int32")
    shared = flag("set", "MTE2_V", alias)
    body = [ir.Bind(alias, i), ir.If(alias < 8, [shared], [shared])]
    after = apply([ir.For(i, integer(0), integer(16), integer(1), ir.ForKind.DEFAULT, body)])
    assert len(calls(after, "tl.ascend_set_flag")) == 1
    (spill,) = calls(after, "tl.ascend_get_buf")
    analyzer = __import__("tvm_tirx", fromlist=["sym"]).sym.Analyzer()
    assert analyzer.can_prove_equal(spill.args[1], alias - 8)


@pytest.mark.parametrize("value", [-1, -8])
def test_negative_event_ids_reject(value):
    with pytest.raises(Exception, match="negative event_id"):
        apply([flag("set", "MTE2_V", value)])


def test_mutex_pool_overflow_is_rejected():
    with pytest.raises(Exception, match="shared pool"):
        apply([flag("set", "MTE2_V", index) for index in range(41)])


def test_wide_dynamic_range_rejects_without_int32_knapsack_truncation():
    i = ir.Var("i", "uint32")
    with pytest.raises(Exception, match="shared pool"):
        apply([flag("set", "MTE2_V", i)], [i])
