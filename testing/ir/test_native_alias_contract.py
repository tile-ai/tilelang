"""Alias exceptions follow native storage identities across split signatures."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.transform as tltr
from tvm_tirx.tirx.compilation_pipeline import finalize_host_passes


def annotation(values):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))


def tensor(name):
    return ir.Var(name, tirx.TensorType([1], "int32", layout="default", data_alignment=4))


def nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def source(*, alias_kind="body", same_names=False, noalias=True):
    a, b, c = (tensor("arg" if same_names else name) for name in ("z", "a", "m"))
    body = []
    selected = [a, b]
    if alias_kind == "view":
        view = tensor("unrelated_view_name")
        body.append(ir.Bind(view, ir.Call("tirx.decl_tensor", [a.data], ty_args=[view.ty])))
        selected = [view, b]
    if alias_kind in ("body", "view"):
        body.append(annotation({"tl.non_restrict_params": selected}))
    body += [
        ir.TensorStore(a, [0], ir.IntImm("int32", 1)),
        ir.TensorStore(b, [0], ir.IntImm("int32", 2)),
        ir.TensorStore(c, [0], ir.TensorLoad(a, [0])),
    ]
    function = tirx.Function([a, b, c], body, ir.NoneType()).with_attr("global_symbol", "main").with_attr("tirx.noalias", noalias)
    if alias_kind == "function":
        function = function.with_attr("tl.non_restrict_params", [a.data, b.data])
    return function


def lower(function, target="c"):
    target = tvm_tirx.target.Target({"kind": target, **({"mcpu": "generic"} if target == "llvm" else {})}, host=target)
    module = ir.IRModule({"main": function})
    for transform in (
        tltr.NormalizeAliasAnnotations(),
        tirx.transform.BindTarget(target),
        tirx.transform.LowerTensorLayouts(),
        tirx.transform.FlattenTensor(),
        tltr.AnnotateDeviceRegions(),
        tltr.SplitHostDevice(),
        tltr.HoistNonRestrictParams(finalize=True),
    ):
        module = transform(module)
    assert tirx.analysis.check_well_formed(module)
    assert not any(isinstance(call.op, ir.Op) and call.op.name == "tl.annotation" for call in nodes(module, ir.Call))
    return module


def compile_lowered(module):
    # This module has already been split. Rebinding every function to a new
    # host-bearing target would incorrectly pack the private compute ABI.
    module = tltr.MakePackedAPI()(module)
    module = finalize_host_passes()(module)
    target = tvm_tirx.target.Target("c")
    executable = tvm_tirx.codegen.Executable(tvm_ffi.get_global_func("target.build.tilelang_c_host")(module, target))
    executable.jit(cc="c++", options=["-std=c++17"])
    return executable


@pytest.mark.parametrize("alias_kind", ["body", "function", "view"])
@pytest.mark.parametrize("same_names", [False, True])
def test_pointer_identity_after_flatten_split_and_reload(alias_kind, same_names):
    module = lower(source(alias_kind=alias_kind, same_names=same_names))
    device = next(function for function in module.functions.values() if function.attrs.get("tirx.is_global_func", False))
    marked = list(device.attrs["tl.non_restrict_params"])
    indices = list(map(int, device.attrs["tl.may_alias_param_indices"]))
    assert len(marked) == len(indices) == 2
    assert all(device.params[index].same_as(value) for index, value in zip(indices, marked))
    assert all(isinstance(parameter.ty, ir.PtrType) for parameter in marked)
    reloaded = ir.load_json(ir.save_json(module))
    assert tirx.analysis.check_well_formed(reloaded)
    shared = np.array([99], dtype="int32")
    output = np.array([-1], dtype="int32")
    executable = compile_lowered(reloaded)
    executable(tvm_ffi.from_dlpack(shared), tvm_ffi.from_dlpack(shared), tvm_ffi.from_dlpack(output))
    np.testing.assert_array_equal(shared, [2])
    np.testing.assert_array_equal(output, [2])


def test_explicit_noalias_false_survives_split():
    module = lower(source(alias_kind="none", noalias=False))
    device = next(function for function in module.functions.values() if function.attrs.get("tirx.is_global_func", False))
    assert not device.attrs["tirx.noalias"]
    shared, output = np.array([99], dtype="int32"), np.array([-1], dtype="int32")
    executable = compile_lowered(module)
    executable(tvm_ffi.from_dlpack(shared), tvm_ffi.from_dlpack(shared), tvm_ffi.from_dlpack(output))
    np.testing.assert_array_equal(output, [2])


@pytest.mark.parametrize("indices,error", [([-1], ValueError), ([3], ValueError), ([0], TypeError)])
def test_invalid_final_alias_indices_are_rejected(indices, error):
    scalar = ir.Var("n", "int32")
    pointer = ir.Var("p", ir.PtrType("int32", "global"))
    function = tirx.Function([scalar, pointer], [ir.Return(ir.IntImm("int32", 0))], ir.PrimType("int32")).with_attr(
        "global_symbol", "entry"
    )
    function = function.with_attr("tirx.noalias", True).with_attr("tl.may_alias_param_indices", indices)
    module = ir.IRModule({"entry": function})
    with pytest.raises(error, match="may_alias_param_indices"):
        # This metadata belongs to TileLang's emitter. Native TIRx emitters
        # deliberately do not interpret downstream alias exceptions.
        tvm_ffi.get_global_func("target.build.tilelang_c_host")(module, tvm_tirx.target.Target("c"))
