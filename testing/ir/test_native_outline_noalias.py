"""A source alias opt-out follows only its own newly extracted kernels."""

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang


def _host(name, noalias, kernels, loc=None):
    source = ir.Var("source", tirx.TensorType([1], "int32", layout="default"))
    output = ir.Var("output", source.ty)
    device = Target("c")
    body = [
        ir.RegionStmt(
            "tirx.device_scope",
            [],
            [],
            {"target": device},
            [ir.TensorStore(output, [0], ir.TensorLoad(source, [0]) + index)],
        )
        for index in range(kernels)
    ]
    function = tirx.Function([source, output], body, ir.PrimType("void"), loc=ir.UnknownLoc() if loc is None else loc).with_attr(
        {"global_symbol": name, "target": device.with_host(device)}
    )
    return function if noalias is None else function.with_attr("tirx.noalias", noalias)


def _device(name, noalias):
    return tirx.Function([], [ir.Evaluate(0)], ir.PrimType("void")).with_attr(
        {"global_symbol": name, "target": Target("c"), "tirx.is_global_func": True, "tirx.noalias": noalias}
    )


def _calls(function):
    result = []
    tvm_ffi.structural_walk(
        function, (ir.Call, lambda call: result.append(call) if isinstance(call.op, ir.GlobalVar) else None), order="pre"
    )
    return result


def test_explicit_noalias_opt_out_is_scoped_to_each_host_and_preserves_existing_devices():
    # This existing name also forces the native name supply to avoid a prefix
    # collision. Alias policy must follow the returned GlobalVar identities.
    existing_true = _device("may_alias_kernel", True)
    existing_false = _device("existing_false", False)
    module = ir.IRModule(
        {
            "may_alias": _host("may_alias", False, 2),
            "disjoint": _host("disjoint", True, 1),
            "unspecified": _host("unspecified", None, 1),
            "may_alias_kernel": existing_true,
            "existing_false": existing_false,
        }
    )
    before = ir.save_json(module)
    result = tilelang.transform.OutlineDeviceRegions()(module)
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    generated = []
    for name, expected, count in [("may_alias", False, 2), ("disjoint", True, 1), ("unspecified", True, 1)]:
        calls = _calls(result[name])
        assert len(calls) == count
        for call in calls:
            assert bool(result[call.op].attrs["tirx.noalias"]) is expected
            assert not tirx.analysis.undefined_vars(result[call.op])
            generated.append(call.op)
    assert len({symbol.name_hint for symbol in generated}) == 4
    assert all(not symbol.same_as(result.get_global_var("may_alias_kernel")) for symbol in generated)
    ir.assert_structural_equal(result["may_alias_kernel"], existing_true)
    ir.assert_structural_equal(result["existing_false"], existing_false)
    ir.assert_structural_equal(tilelang.transform.OutlineDeviceRegions()(result), result)


def test_outlining_preserves_source_locations_without_copying_lowering_flags():
    location = ir.SourceLoc(ir.SourceName("outline.py"), 27, 0, 30, 12)
    module = ir.IRModule({"main": _host("main", True, 2, loc=location)})
    before = ir.save_json(module)
    result = tilelang.transform.OutlineDeviceRegions()(module)
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    calls = _calls(result["main"])
    assert len(calls) == 2
    for call in calls:
        assert result[call.op].loc.same_as(location)
    ir.assert_structural_equal(tilelang.transform.OutlineDeviceRegions()(result), result)
