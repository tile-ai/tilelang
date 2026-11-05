"""Version padding must preserve every element of a native pointer span."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx

_spec = importlib.util.spec_from_file_location("allocation_helpers", Path(__file__).with_name("test_native_ascend_allocations.py"))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def _copy_function(pointer_kind, guarded):
    storage = helpers.tensor("storage", (2, 3), "int32")
    output = helpers.tensor("output", (6,), "int32", "global")
    count = ir.Var("count", "int32")
    body = [helpers.allocate(storage), helpers.annotation({"tl.buffer_version": {storage: 2}})]
    body.extend(
        ir.TensorStore(storage, [version, index], ir.const(100 + version * 3 + index)) for version in range(2) for index in range(3)
    )
    source = helpers.pointer(storage, [0, 0], extent=count, mask=1)
    if pointer_kind == "raw":
        source = ir.Call("tl.ptr_access", [storage.data, ir.const(0), count, ir.const(1)])
    copy = ir.Evaluate(
        tirx.call_extern("memcpy", output.data, source, ir.prim.Cast("uint64", count) * ir.const(4, "uint64"), ty=ir.PtrType("void"))
    )
    if guarded:
        body.append(ir.If((count >= 0) & (count <= 3), [copy], []))
    else:
        body.append(copy)
    return tirx.Function([output, count], body)


def _rewrite(function):
    saved = ir.save_json(function)
    assert tirx.analysis.check_well_formed(function)
    result = tvm_ffi.get_global_func("tl.transform.RewriteAscendBufferVersionLayout")()(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == saved
    assert tirx.analysis.check_well_formed(result)
    return result


@pytest.mark.parametrize("pointer_kind", ["tensor", "raw"])
def test_unproven_dynamic_span_preserves_valid_legacy_calls(pointer_kind, tmp_path):
    # Dynamic spans have the same precondition as constant spans: they must
    # remain within one version slot. A caller can guarantee that bound even
    # when the Function does not carry an explicit guard.
    result = _rewrite(_copy_function(pointer_kind, guarded=False))
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(ir.IRModule({"main": result}))
    execute = helpers.compile_cpu(module["main"], tmp_path)
    for amount in [0, 1, 3]:
        output = tvm.runtime.tensor(np.full(6, -1, dtype="int32"))
        execute(output, amount)
        expected = np.array([100 + index if index < amount else -1 for index in range(6)], dtype="int32")
        np.testing.assert_array_equal(output.numpy(), expected)


@pytest.mark.parametrize("pointer_kind", ["tensor", "raw"])
def test_proven_dynamic_span_executes_without_changing_values(pointer_kind, tmp_path):
    result = _rewrite(_copy_function(pointer_kind, guarded=True))
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(ir.IRModule({"main": result}))
    execute = helpers.compile_cpu(module["main"], tmp_path)
    for amount in [0, 1, 3, 4]:
        output = tvm.runtime.tensor(np.full(6, -1, dtype="int32"))
        execute(output, amount)
        copied = amount if amount <= 3 else 0
        expected = np.array([100 + index if index < copied else -1 for index in range(6)], dtype="int32")
        np.testing.assert_array_equal(output.numpy(), expected)


def test_span_proof_does_not_escape_a_branch():
    function = _copy_function("tensor", guarded=True)
    statements = list(function.body.seq)
    count = function.params[1]
    # The earlier count <= 3 fact must not hide a proved crossing in a later
    # independent branch. Constant and proved-dynamic crossings still fail.
    statements.append(ir.If(count >= 4, [statements[-1].then_case.seq[0]], []))
    with pytest.raises(ValueError, match="Access pointer crosses"):
        _rewrite(function.with_body(statements))
