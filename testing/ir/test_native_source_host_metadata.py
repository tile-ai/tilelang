"""Retained source host graphs own definitions independently of packed IR."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.jit.adapter.source_host import SourceHostProgram, source_host_functions
from tilelang.utils.pass_diff import _get_script


def source_graph():
    target = Target("c", Target({"kind": "llvm", "mcpu": "generic"}))
    entry, helper = ir.GlobalVar("main"), ir.GlobalVar("helper")
    n, value = ir.Var("n", "int32"), ir.Var("value", "int32")
    leaf = tirx.Function([value], [ir.Return(value + 3)], value.ty, ir.DictAttrs({"target": target}))
    function = tirx.Function(
        [n],
        [ir.Evaluate(ir.Call(helper, [n], ty=n.ty)), ir.Return(ir.Tuple([]))],
        ir.TupleType([]),
        ir.DictAttrs({"global_symbol": "main", "target": target}),
    )
    return function, ir.IRModule({entry: function, helper: leaf})


@pytest.mark.parametrize("method", ["direct", "json", "diagnostic_json"])
def test_retained_host_module_preserves_definitions_and_helper_calls(method):
    function, original = source_graph()
    before = ir.save_json(original)
    packed = tvm_ffi.get_global_func("tl.transform.MakePackedAPI")()(original)
    assert ir.save_json(original) == before
    assert int(packed.attrs["tl.source_host_schema"]) == 2
    retained = packed.attrs["tl.source_host_module"]
    assert isinstance(retained, ir.IRModule)
    for key in retained.functions:
        assert not key.same_as(original.get_global_var(key.name_hint))
    restored = packed
    if method == "json":
        restored = ir.load_json(ir.save_json(packed))
    elif method == "diagnostic_json":
        # Full native metadata is serialized as JSON. The pinned printer
        # cannot express an IRModule inside another module's attributes; the
        # production diagnostic path must report that limit and retain it.
        snapshot = _get_script(packed)
        marker = "# Snapshot format: native IR JSON (not TVMScript)\n"
        assert snapshot.startswith("# TVMScript capture failed (")
        assert marker in snapshot
        restored = ir.load_json(snapshot.split(marker, 1)[1])
    ir.assert_structural_equal(packed, restored)
    ir.assert_structural_equal(restored, packed)
    assert tvm_ffi.structural_hash(packed) == tvm_ffi.structural_hash(restored)
    functions = source_host_functions(restored)
    retained = restored.attrs["tl.source_host_module"]
    # JSON also retains sharing with the outer module, not just a separately
    # equivalent copy of the nested graph.
    assert restored["helper"].params[0].same_as(retained["helper"].params[0])
    calls = []
    for saved in functions.values():
        tvm_ffi.structural_walk(saved.body, (ir.Call, calls.append))
    internal = [call for call in calls if isinstance(call.op, ir.GlobalVar)]
    assert len(internal) == 1 and internal[0].op in functions
    assert tirx.analysis.check_well_formed(SourceHostProgram(function, restored).raw_module())


@pytest.mark.parametrize("schema", [0, 1, 3])
def test_stale_source_host_schema_is_not_accepted_as_a_current_cache(schema):
    _, original = source_graph()
    module = original.with_attr("tl.source_host_schema", schema)
    module = module.with_attr("tl.source_host_module", original)
    with pytest.raises(ValueError, match="retained by MakePackedAPI"):
        source_host_functions(module)


def test_source_host_schema_requires_an_owned_module():
    _, original = source_graph()
    malformed = original.with_attr("tl.source_host_schema", 2)
    malformed = malformed.with_attr("tl.source_host_module", original.functions)
    with pytest.raises(ValueError, match="Malformed native source-host module"):
        source_host_functions(malformed)
