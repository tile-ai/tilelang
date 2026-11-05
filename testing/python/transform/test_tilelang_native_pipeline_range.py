"""A conservative index range must not erase the storage identity of a tensor."""

import numpy as np
import pytest
from tvm_tirx import ir, runtime, target, tirx

import tilelang
from tilelang.testing.packed import compile_packed_function


def make_staged_function(dtype, extent, begin):
    size = max(1, extent)
    source = ir.Var("source", tirx.TensorType((size,), "int32"))
    output = ir.Var("output", tirx.TensorType((size,), "int32"))
    temporary = ir.Var("temporary", tirx.TensorType((1,), "int32", scope="local"))
    index = ir.Var("index", dtype)
    offset = index - ir.const(begin, dtype)
    loop = ir.For(
        index,
        ir.const(begin, dtype),
        ir.const(begin + extent, dtype),
        ir.const(1, dtype),
        ir.ForKind.DEFAULT,
        [
            ir.TensorStore(temporary, [0], ir.TensorLoad(source, [offset]) + 3),
            ir.TensorStore(output, [offset], ir.TensorLoad(temporary, [0]) * 2),
        ],
        annotations={"tl_pipeline_order": [0, 1], "tl_pipeline_stage": [0, 2]},
    )
    return tirx.Function(
        [source, output],
        [ir.Bind(temporary, ir.Call("tirx.alloc_tensor", [], ty_args=[temporary.ty])), loop],
    ).with_attr({"global_symbol": "main", "tirx.noalias": True, "target": target.Target({"kind": "llvm", "mcpu": "generic"})})


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("extent", [0, 1, 7])
@pytest.mark.parametrize("begin", [0, 3])
def test_manual_pipeline_known_tensor_preserves_numeric_result(dtype, extent, begin):
    function = make_staged_function(dtype, extent, begin)
    assert tirx.analysis.check_well_formed(function)
    snapshot = ir.save_json(function)
    module = ir.IRModule({"main": function})
    planned = tilelang.transform.PipelinePlanning()(module)
    assert tirx.analysis.check_well_formed(planned)
    planned_loop = planned["main"].body.seq[-1]
    assert list(planned_loop.annotations["software_pipeline_stage"]) == [0, 2]
    assert list(planned_loop.annotations["software_pipeline_order"]) == [0, 1]
    injected = tilelang.transform.InjectSoftwarePipeline()(planned)
    assert tirx.analysis.check_well_formed(injected)
    assert not tirx.analysis.undefined_vars(injected["main"])
    assert ir.save_json(function) == snapshot

    data = np.arange(max(1, extent), dtype="int32") * 5 - 11
    expected = np.full_like(data, -99)
    expected[:extent] = (data[:extent] + 3) * 2
    # Both before and after pass execution use the same independently declared
    # arithmetic oracle. A zero-trip loop must retain the sentinel output.
    for candidate in (function, injected["main"]):
        executable = compile_packed_function(candidate, c_codegen="compute")
        actual = runtime.tensor(np.full_like(data, -99))
        executable(runtime.tensor(data), actual)
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("kind", ["raw_pointer", "opaque_global_state"])
def test_manual_pipeline_unknown_storage_still_rejected(kind):
    index = ir.Var("index", "int32")
    if kind == "raw_pointer":
        destination = ir.Var("destination", ir.PtrType("int32", scope="shared"))
        source = ir.Var("source", ir.PtrType("int32", scope="global"))
        params = [destination, source]
        operation = ir.Call("tl.ptx_cp_async", [destination, source, ir.const(1)])
    else:
        params = []
        operation = tirx.call_extern("peer_unknown_global_state", ty=ir.PrimType("void"))
    loop = ir.For(
        index,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        [ir.Evaluate(operation)],
        annotations={"tl_pipeline_order": [0], "tl_pipeline_stage": [0]},
    )
    function = tirx.Function(params, [loop]).with_attr("target", target.Target({"kind": "cuda", "arch": "sm_90"}))
    assert tirx.analysis.check_well_formed(function)
    with pytest.raises(ValueError, match="explicit tensor storage identity"):
        tilelang.transform.PipelinePlanning()(ir.IRModule({"main": function}))
