"""Named parameter lookups retain the old C storage-reference contract."""

from tilelang.language import _annotation_codec
from tilelang.language import _construction
from tvm_tirx.tirx.script import ir_builder as _builder

import ctypes
import json
import subprocess

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang.language as T


@pytest.mark.parametrize("as_expression", [False, True])
def test_lookup_param_has_one_literal_name_and_a_native_pointer_result(as_expression):
    name = ir.StringImm("weights") if as_expression else "weights"
    loc = ir.SourceLoc(ir.SourceName("parameter.py"), 8, 2, 8, 28)
    result = T.lookup_param(name, span=loc)
    assert isinstance(result, ir.Call)
    assert result.op.name == "tl.lookup_param"
    assert result.ty == ir.PtrType("void")
    assert result.loc.same_as(loc)
    assert len(result.args) == 1 and result.args[0].value == "weights"
    if as_expression:
        assert result.args[0].same_as(name)
    assert result.attrs is None and len(result.ty_args) == 0
    assert int(result.op.get_attr("TCallEffectKind")) == int(ir.CallEffectKind.UpdateState)
    result.validate()
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(result)), result)


@pytest.mark.parametrize("name", [17, ("weights",), ir.Var("name", ir.StringType()), ir.Var("pointer", ir.PtrType("void"))])
def test_lookup_param_rejects_nonliteral_names(name):
    with pytest.raises(TypeError):
        T.lookup_param(name)


@pytest.mark.parametrize("arguments", [[], [ir.StringImm("p0"), ir.StringImm("p1")]])
def test_raw_lookup_param_rejects_wrong_arity(arguments):
    with pytest.raises(ValueError):
        ir.Call("tl.lookup_param", arguments).validate()


@pytest.mark.parametrize("options", [{"attrs": {}}, {"attrs": {"unused": 1}}, {"ty_args": [ir.PrimType("int32")]}])
def test_lookup_param_rejects_unconsumed_attributes_and_type_arguments(options):
    with pytest.raises(ValueError):
        ir.Call("tl.lookup_param", [ir.StringImm("p0")], **options).validate()


@pytest.mark.parametrize("result_type", [ir.PrimType("int32"), ir.PtrType("int32"), ir.PtrType("void", "shared")])
def test_lookup_param_rejects_an_explicit_wrong_result_type(result_type):
    with pytest.raises(TypeError):
        ir.Call("tl.lookup_param", [ir.StringImm("p0")], ty=result_type).validate()


def test_lookup_param_uses_the_active_source_location():
    @T.prim_func
    def source():
        T.evaluate(T.lookup_param("weights"))

    calls = []
    tvm_ffi.structural_walk(source, (ir.Call, calls.append))
    result = next(call for call in calls if call.op.name == "tl.lookup_param")
    assert isinstance(result.loc, ir.SourceLoc)
    assert tirx.analysis.check_well_formed(source)


def lookup_function(name, values):
    storage = "int32_t __tvm_param__" + name + "[] = {" + ", ".join(map(str, values)) + "};"
    with IRBuilder() as builder, _construction.prim_func():
        T.func_name("lookup_" + name)
        T.func_ret(ir.PtrType("void"))
        with _annotation_codec.attr(0, "pragma_import_c", storage):
            _builder.emit_(ir.Return(T.lookup_param(name)))
    function = builder.get()
    assert tirx.analysis.check_well_formed(function)
    return function


@pytest.mark.parametrize("backend", ["tilelang_c_host", "tilelang_c"])
def test_c_codegen_resolves_real_named_storage(backend, tmp_path):
    expected = {"p0": [13, 29, -17], "weights": [5, -11, 37]}
    module = tvm_tirx.IRModule({"lookup_" + name: lookup_function(name, values) for name, values in expected.items()})
    generated = tvm_ffi.get_global_func("target.build." + backend)(module, tvm_tirx.target.Target("c"))
    source = generated.inspect_source("c")
    for name in expected:
        assert "return __tvm_param__" + name + ";" in source
        assert "__tvm_param__" + name + "(" not in source
    (tmp_path / "generated.c").write_text(source)
    (tmp_path / "function.json").write_text(ir.save_json(module))
    if backend != "tilelang_c_host":
        return

    from tvm_ffi.libinfo import find_include_path as ffi_include
    from tvm_tirx.libinfo import find_include_path as tirx_include

    # Compile the complete emitter output. The pragma provides real named
    # arrays; the lookup must return their addresses without allocating data.
    library = tmp_path / "lookup.so"
    command = ["cc", "-std=c11", "-fPIC", "-shared", str(tmp_path / "generated.c"), "-o", str(library)]
    command.extend(["-I" + tirx_include(), "-I" + ffi_include()])
    compiled = subprocess.run(command, capture_output=True, text=True, check=False)
    (tmp_path / "compile.json").write_text(json.dumps({"command": command, "returncode": compiled.returncode}, indent=2))
    (tmp_path / "compile.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stderr
    runtime = ctypes.CDLL(str(library))
    observations = []
    for name, values in expected.items():
        lookup = getattr(runtime, "lookup_" + name)
        lookup.argtypes = []
        lookup.restype = ctypes.c_void_p
        storage = (ctypes.c_int32 * len(values)).in_dll(runtime, "__tvm_param__" + name)
        address = lookup()
        assert address == ctypes.addressof(storage)
        view = (ctypes.c_int32 * len(values)).from_address(address)
        assert list(view) == values
        storage[1] = 83
        second_address = lookup()
        assert second_address == address
        assert (ctypes.c_int32 * len(values)).from_address(second_address)[1] == 83
        observations.append({"name": name, "initial_values": values, "address": address, "mutated_middle_value": view[1]})
    assert observations[0]["address"] != observations[1]["address"]
    (tmp_path / "runtime.json").write_text(json.dumps({"c_compilations": 1, "entry_calls": 4, "observations": observations}, indent=2))
