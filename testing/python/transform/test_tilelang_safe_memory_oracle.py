"""Reject valid IR mutations that lose the original safe-access obligations."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
from testing.python.transform import _safe_memory_oracle as oracle
from testing.python.transform import test_tilelang_transform_legalize_safe_memory_access as fixtures


def _validate_distinct(reference, mutant):
    assert not tvm_ffi.structural_equal(reference, mutant)
    for function in (reference, mutant):
        assert tirx.analysis.check_well_formed(function)
        tvm_ffi.structural_walk(function, (ir.Call, lambda call: call.validate()))


@pytest.mark.parametrize("access_type", ["r", "w", "rw"])
@pytest.mark.parametrize("field,argument", [("mask", 2), ("extent", 1)])
def test_extern_pointer_metadata_is_observable(access_type, field, argument):
    _, reference = fixtures.call_extern_access_ptr_mask_legalize(access_type)
    unchanged = ir.save_json(reference)

    def mutate(call):
        if not isinstance(call.op, ir.Op) or call.op.name != "tl.access_ptr":
            return call
        args = list(call.args)
        old = int(args[argument])
        new = (2 if old == 1 else 1) if field == "mask" else old + 1
        args[argument] = ir.const(new, args[argument].ty.dtype)
        return ir.Call(call.op, args, ty=call.ty, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc)

    mutant = tvm_ffi.structural_map(reference, [(ir.Call, mutate)])
    assert ir.save_json(reference) == unchanged
    _validate_distinct(reference, mutant)
    oracle.assert_same_memory_effects(reference, reference)
    with pytest.raises(AssertionError):
        oracle.assert_same_memory_effects(mutant, reference)


@pytest.mark.parametrize("removed", ["tl.ptx_commit_group", "tl.ptx_wait_group"])
def test_nonzero_padding_preserves_async_operations(removed):
    before, reference = fixtures.cp_async_access_ptr_nonzero_safe_value_legalize()
    actual = tilelang.transform.LegalizeSafeMemoryAccess()(ir.IRModule({"main": before}))["main"]

    def mutate(statement):
        if isinstance(statement.value, ir.Call) and statement.value.op.name == removed:
            return ir.Evaluate(ir.const(0), loc=statement.loc)
        return statement

    mutant = tvm_ffi.structural_map(actual, [(ir.Evaluate, mutate)])
    _validate_distinct(actual, mutant)
    oracle.assert_nonzero_async_padding(actual, reference)
    with pytest.raises(AssertionError, match="synchronization effects"):
        oracle.assert_nonzero_async_padding(mutant, reference)


@pytest.mark.parametrize("mutation", ["wait_operand", "operation_order"])
def test_nonzero_padding_preserves_async_arguments_and_order(mutation):
    before, reference = fixtures.cp_async_access_ptr_nonzero_safe_value_legalize()
    actual = tilelang.transform.LegalizeSafeMemoryAccess()(ir.IRModule({"main": before}))["main"]

    def change_wait(call):
        if call.op.name != "tl.ptx_wait_group":
            return call
        args = list(call.args)
        args[0] = ir.const(1, args[0].ty.dtype)
        return ir.Call(call.op, args, ty=call.ty, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc)

    def reorder(sequence):
        statements = list(sequence.seq)
        positions = [
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, ir.Evaluate)
            and isinstance(statement.value, ir.Call)
            and statement.value.op.name in ("tl.ptx_commit_group", "tl.ptx_wait_group")
        ]
        if len(positions) != 2:
            return sequence
        first, second = positions
        statements[first], statements[second] = statements[second], statements[first]
        return ir.SeqStmt(statements, loc=sequence.loc)

    callbacks = [(ir.Call, change_wait)] if mutation == "wait_operand" else [(ir.SeqStmt, reorder)]
    mutant = tvm_ffi.structural_map(actual, callbacks)
    _validate_distinct(actual, mutant)
    oracle.assert_nonzero_async_padding(actual, reference)
    with pytest.raises(AssertionError, match="synchronization effects"):
        oracle.assert_nonzero_async_padding(mutant, reference)


def test_copy_source_address_cannot_alias_a_periodic_initializer():
    _, reference = fixtures.vectorize_access_legalize()
    reference = fixtures._materialize_launch(reference)
    source = reference.params[0]

    def mutate(load):
        if not load.source.same_as(source):
            return load
        indices = list(load.indices)
        indices[-1] = ir.prim.FloorMod(indices[-1], ir.const(29, "int32"))
        return ir.TensorLoad(load.source, indices, loc=load.loc)

    mutant = tvm_ffi.structural_map(reference, [(ir.TensorLoad, mutate)])
    _validate_distinct(reference, mutant)
    oracle.assert_same_memory_effects(reference, reference)
    with pytest.raises(AssertionError, match="memory bytes or observable effects"):
        oracle.assert_same_memory_effects(mutant, reference)
