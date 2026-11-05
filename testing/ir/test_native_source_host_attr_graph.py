"""Source-host packing preserves attribute references to owned Globals and parameters."""

import pytest
import tvm_ffi
from tilelang.jit.adapter.source_host import SourceHostProgram, source_host_functions
from tilelang.utils.pass_diff import _get_script

from tvm_tirx import ir, tirx
from tvm_tirx.target import Target


@pytest.mark.parametrize("roundtrip", ["direct", "json", "diagnostic_json"])
def test_function_attrs_refer_to_owned_globals_and_parameter_definitions(roundtrip):
    target = Target("c", Target({"kind": "llvm", "mcpu": "generic"}))
    entry, helper = ir.GlobalVar("main"), ir.GlobalVar("helper")
    n, value = ir.Var("n", "int32"), ir.Var("value", "int32")
    leaf = tirx.Function([value], [ir.Return(value + 3)], value.ty, ir.DictAttrs({"target": target}))
    data_call = ir.Call(helper, [n], ty=n.ty)
    function = tirx.Function(
        [n],
        [ir.Evaluate(data_call), ir.Return(ir.Tuple([]))],
        ir.TupleType([]),
        ir.DictAttrs(
            {
                "global_symbol": "main",
                "target": target,
                "review.calls": [data_call],
                "review.helper": helper,
            }
        ),
    )
    original = ir.IRModule({entry: function, helper: leaf})
    original_json = ir.save_json(original)
    packed = tvm_ffi.get_global_func("tl.transform.MakePackedAPI")()(original)
    assert ir.save_json(original) == original_json
    retained = packed.attrs["tl.source_host_module"]
    assert not retained.get_global_var("main").same_as(entry)
    assert not retained.get_global_var("helper").same_as(helper)
    if roundtrip == "json":
        restored = ir.load_json(ir.save_json(packed))
    elif roundtrip == "diagnostic_json":
        snapshot = _get_script(packed)
        marker = "# Snapshot format: native IR JSON (not TVMScript)\n"
        assert snapshot.startswith("# TVMScript capture failed (")
        assert marker in snapshot
        restored = ir.load_json(snapshot.split(marker, 1)[1])
    else:
        restored = packed
    ir.assert_structural_equal(packed, restored)
    ir.assert_structural_equal(restored, packed)
    assert tvm_ffi.structural_hash(packed) == tvm_ffi.structural_hash(restored)
    retained = restored.attrs["tl.source_host_module"]
    caller = retained["main"]
    assert caller.attrs["review.helper"].same_as(retained.get_global_var("helper"))
    metadata_call = caller.attrs["review.calls"][0]
    assert metadata_call.op.same_as(retained.get_global_var("helper"))
    assert metadata_call.args[0].same_as(caller.params[0])
    assert restored["main"].attrs["review.calls"][0].args[0].same_as(caller.params[0])
    assert restored["helper"].params[0].same_as(retained["helper"].params[0])
    # Metadata must not synthesize a second executable call.
    assert len(caller.body.seq) == 2
    functions = source_host_functions(restored)
    assert len(functions) == 2
    assert tirx.analysis.check_well_formed(SourceHostProgram(function, restored).raw_module())
