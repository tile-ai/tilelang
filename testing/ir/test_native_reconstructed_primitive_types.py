"""Native type consumers retain value semantics after JSON reconstruction."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT


def _reconstruct(value, mode):
    if mode == "json":
        result = ir.load_json(ir.save_json(value))
    elif mode == "renew":
        result = tirx.renew_def(value)
    else:
        result = value
    ir.assert_structural_equal(result, value)
    return result


def _source(function):
    return str(tvm_ffi.get_global_func("tl.codegen.GenerateSourceHost")(ir.IRModule({"entry": function}), True))


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
def test_source_entry_primitive_return_survives_reconstruction(mode):
    calls = []
    name = "tl.testing.reconstructed_source_entry_" + mode
    tvm_ffi.register_global_func(name, lambda: calls.append("called"), override=True)
    function = tirx.Function(
        [],
        [ir.Evaluate(ir.Call("tirx.call_packed", [ir.StringImm(name)], ty="void")), ir.Return(ir.const(0, "int32"))],
        ir.PrimType("int32"),
        ir.DictAttrs({"global_symbol": "entry", "target": Target("c"), "tl.source_entry": 1}),
    )
    function = _reconstruct(function, mode)
    assert tirx.analysis.check_well_formed(function)
    source = _source(function)
    assert ".cast<void>()" not in source
    generator = LibraryGenerator(Target("c"))
    generator.assign_pass_configs({})
    generator.assign_compile_flags([])
    generator.update_lib_code(NATIVE_SOURCE_INIT + source)
    generator.compile_lib()
    library = generator.load_lib()
    library.entry.argtypes = []
    library.entry.restype = ctypes.c_int32
    assert library.entry() == 0
    assert calls == ["called"]


@pytest.mark.parametrize("mode", ["direct", "json"])
@pytest.mark.parametrize("dtype", ["int64", "uint32", "bool", "float32"])
def test_source_entry_requires_exact_int32_error_type(mode, dtype):
    function = tirx.Function(
        [],
        [ir.Return(ir.const(0, dtype))],
        ir.PrimType(dtype),
        ir.DictAttrs({"global_symbol": "entry", "target": Target("c"), "tl.source_entry": 1}),
    )
    with pytest.raises(TypeError, match="source adapter entry must return an int32 error code"):
        _source(_reconstruct(function, mode))


def _finalize_compute(module):
    passes = tilelang.transform.SplitHostDevice().passes
    (finalize,) = [value for value in passes if value.info.name == "tl.FinalizeCPUComputeCalls"]
    return finalize(module)


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint32"])
def test_compute_error_guard_uses_exact_reconstructed_int32(mode, dtype):
    compute, host = ir.GlobalVar("compute"), ir.GlobalVar("host")
    status = ir.Var("status", dtype)
    callee = tirx.Function(
        [],
        [ir.Return(ir.const(0, dtype))],
        ir.PrimType(dtype),
        ir.DictAttrs({"target": Target("c"), "tirx.is_global_func": True, "global_symbol": "compute_export"}),
    )
    caller = tirx.Function(
        [],
        [
            ir.Bind(status, ir.Call(compute, [], ty=dtype)),
            ir.AssertStmt(status == ir.const(0, dtype), ir.StringImm("RuntimeError"), [ir.StringImm("Error executing compute kernel")]),
        ],
        ir.NoneType(),
        ir.DictAttrs({"target": Target("c", "c"), "global_symbol": "host"}),
    )
    module = ir.IRModule({compute: callee, host: caller})
    if mode == "renew":
        module = ir.IRModule({key: tirx.renew_def(value) for key, value in module.functions.items()})
    else:
        module = _reconstruct(module, mode)
    assert tirx.analysis.check_well_formed(module)
    output = _finalize_compute(module)
    assert tirx.analysis.check_well_formed(output)
    calls, assertions = [], []
    tvm_ffi.structural_walk(output["host"], (ir.Call, calls.append))
    tvm_ffi.structural_walk(output["host"], (ir.AssertStmt, assertions.append))
    externs = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_extern"]
    assert len(externs) == 1 and externs[0].args[0].value == "compute_export"
    throws = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.throw_last_error"]
    assert len(throws) == (1 if dtype == "int32" else 0)
    assert len(assertions) == (0 if dtype == "int32" else 1)


def _hint_function(polarity):
    a = ir.Var("A", tirx.TensorType([8], "int32"))
    b = ir.Var("B", tirx.TensorType([8], "int32"))
    marker = ir.Call("tl.conflict_hint", [a, b, ir.const(-2, "int32"), ir.const(-1, "int32"), ir.StringImm(""), polarity], ty="int32")
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
        {"node": None, "key": "tl.sblock", "value": {"name": "tilelang_root", "no_realize": True}}
    )
    region = ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [ir.Evaluate(marker)])
    return tirx.Function([a, b], [region])


@pytest.mark.parametrize("mode", ["direct", "json", "renew"])
@pytest.mark.parametrize("polarity", [False, True])
def test_conflict_polarity_value_survives_reconstruction(mode, polarity):
    function = _reconstruct(_hint_function(ir.const(polarity, "bool")), mode)
    assert tirx.analysis.check_well_formed(function)
    result = tvm_ffi.get_global_func("tl.transform.NormalizeConflictHints")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    calls = []
    tvm_ffi.structural_walk(result, (ir.Call, calls.append))
    assert not any(isinstance(call.op, ir.Op) and call.op.name == "tl.conflict_hint" for call in calls)
    carriers = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tl.annotation"]
    entries = []
    for call in carriers:
        fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
        entries.extend(fields.get("tl.root_conflict_hints", []))
    assert len(entries) == 1 and bool(entries[0][3]) == polarity


@pytest.mark.parametrize("mode", ["direct", "json"])
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_conflict_polarity_rejects_non_boolean_reconstruction(mode, dtype):
    function = _reconstruct(_hint_function(ir.const(1, dtype)), mode)
    with pytest.raises(ValueError, match="constant boolean polarity"):
        tvm_ffi.get_global_func("tl.transform.NormalizeConflictHints")()(ir.IRModule({"main": function}))
