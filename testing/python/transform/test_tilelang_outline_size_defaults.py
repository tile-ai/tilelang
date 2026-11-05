"""Host/device outlining retains size defaults by current variable identity."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

import tilelang


def variables(object_):
    result = []
    tvm_ffi.structural_walk(object_, (ir.Var, lambda variable: result.append(variable)))
    return result


def defaults(function):
    return list(tvm_ffi.get_global_func("tl.CollectSizeVarDefaults")(function))


def make_module():
    size = tvm_ffi.get_global_func("tl.MarkSizeVarDefault")(ir.Var("n", "int32"))
    scalar = ir.Var("n", "int32")
    source = ir.Var("A", tirx.TensorType([prim.max(size, 0)], "int32", align=4))
    output = ir.Var("B", tirx.TensorType([2], "int32", align=4))
    target = Target("c").with_host(Target("c"))
    function = (
        tirx.Function(
            [source, output, scalar],
            [ir.TensorStore(output, [0], size), ir.TensorStore(output, [1], scalar)],
        )
        .with_attr("global_symbol", "compute")
        .with_attr("target", target)
    )
    function = tvm_ffi.get_global_func("tl.AttachSizeVarDefaults")(function)

    def declaration():
        tensor = ir.Var("A", tirx.TensorType([ir.Var("n", "int32")], "int32"))
        return tirx.Function([tensor], [], attrs=ir.DictAttrs({"declaration": True, "target": target}))

    return ir.IRModule({"main": function, "compute_kernel": declaration(), "other": declaration()})


def check_carriers(module):
    host = module["main"]
    host_size = variables(host.params[0].ty.shape)
    assert len(host_size) == 1
    assert len(defaults(host)) == 1
    assert defaults(host)[0].same_as(host_size[0])
    assert not defaults(host)[0].same_as(host.params[2])
    assert host_size[0].name == host.params[2].name

    device_entries = [
        (symbol, function) for symbol, function in module.functions.items() if symbol.name_hint not in ("main", "compute_kernel", "other")
    ]
    assert len(device_entries) == 1
    device_symbol, device = device_entries[0]
    stores = []
    tvm_ffi.structural_walk(device.body, (ir.TensorStore, lambda store: stores.append(store)))
    stored = {int(store.indices[0]): store.value for store in stores}
    assert len(defaults(device)) == 1
    assert defaults(device)[0].same_as(stored[0])
    assert not defaults(device)[0].same_as(stored[1])
    assert any(defaults(device)[0].same_as(parameter) for parameter in device.params)

    calls = []
    tvm_ffi.structural_walk(host.body, (ir.Call, lambda call: calls.append(call)))
    assert any(isinstance(call.op, ir.GlobalVar) and call.op.same_as(device_symbol) for call in calls)
    for name in ("compute_kernel", "other"):
        assert module[name].attrs["declaration"]
        assert len(module[name].body.seq) == 0
        assert defaults(module[name]) == []
    # The native verifier explicitly requires definitions. Declarations stay
    # in the original module for name supply and linking, and are checked
    # above for their preserved declaration contract.
    definitions = ir.IRModule(
        {symbol: function for symbol, function in module.functions.items() if not function.attrs.get("declaration", False)}
    )
    assert tirx.analysis.check_well_formed(definitions)


@pytest.mark.parametrize("serialized", [False, True])
def test_outlining_preserves_host_and_device_default_operands(serialized):
    source = make_module()
    if serialized:
        source = ir.load_json(ir.save_json(source))
    original = ir.save_json(source)
    annotated = tilelang.transform.AnnotateDeviceRegions()(source)
    outlined = tilelang.transform.OutlineDeviceRegions()(annotated)
    assert ir.save_json(source) == original
    check_carriers(outlined)
    repeated = tilelang.transform.OutlineDeviceRegions()(outlined)
    assert len(repeated.functions) == len(outlined.functions)
    check_carriers(repeated)
    assert tvm_ffi.structural_equal(repeated, outlined)
