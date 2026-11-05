"""Owning clamp packing keeps explicit types and scalar broadcast semantics."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def _objects(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="post")
    return result


@pytest.mark.parametrize("explicit_type_argument", [False, True])
@pytest.mark.parametrize("scalar_bounds", [False, True])
def test_clamp_vectorization_retypes_result_and_preserves_values(explicit_type_argument, scalar_bounds):
    tensors = [ir.Var(name, tirx.TensorType([4], "float32", data_alignment=16)) for name in ["a", "lo", "hi", "out"]]
    a, lo, hi, out = tensors
    lane = ir.Var("lane", "int32")
    operands = [ir.TensorLoad(a, [lane])]
    if scalar_bounds:
        operands += [ir.const(-1.0, "float32"), ir.const(1.0, "float32")]
    else:
        operands += [ir.TensorLoad(lo, [lane]), ir.TensorLoad(hi, [lane])]
    scalar = ir.PrimType("float32")
    call = ir.Call("tl.clamp", operands, ty=scalar, ty_args=[scalar] if explicit_type_argument else [])
    body = ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, [ir.TensorStore(out, [lane], call)])
    source = tirx.Function(tensors, body, ir.NoneType()).with_attr("tirx.noalias", True)
    before = ir.save_json(source)
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": source}))["main"]
    assert ir.save_json(source) == before
    assert tirx.analysis.check_well_formed(result)
    assert not _objects(result, ir.For)
    calls = [call for call in _objects(result, ir.Call) if call.op.same_as(ir.Op.get("tl.clamp"))]
    assert len(calls) == 1
    vector = calls[0]
    assert vector.ty == ir.PrimType("float32x4")
    assert all(argument.ty == vector.ty for argument in vector.args)
    assert list(vector.ty_args) == ([vector.ty] if explicit_type_argument else [])
    vector.validate()

    from tilelang.testing.packed import compile_packed_function

    result = result.with_attr("target", tvm_tirx.target.Target("c")).with_attr("global_symbol", "main")
    compiled = compile_packed_function(result, target="c", c_codegen="compute")
    cases = [
        ([-2.0, -0.0, 0.5, 2.0], [-1.0] * 4, [1.0] * 4),
        ([np.nan, 0.5, 0.5, 0.5], [-1.0, np.nan, -1.0, 2.0], [1.0, 1.0, np.nan, -2.0]),
    ]
    for data, lower, upper in cases:
        arrays = [np.asarray(values, dtype="float32") for values in [data, lower, upper]]
        expected = np.minimum(np.maximum(arrays[0], -1 if scalar_bounds else arrays[1]), 1 if scalar_bounds else arrays[2])
        inputs = [tvm_tirx.runtime.tensor(array) for array in arrays]
        output = tvm_tirx.runtime.tensor(np.full(4, 99, dtype="float32"))
        compiled(*inputs, output)
        np.testing.assert_array_equal(output.numpy(), expected)
