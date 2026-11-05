"""A vector tensor element and a vector access have independent lane counts."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.transform as tr
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("padding", [None, -7.0, (-3.0, 5.0)])
def test_out_of_bounds_vector_elements_preserve_padding_lane_order(padding):
    source = ir.Var("source", tirx.TensorType([2], "float32x2", data_alignment=4))
    output = ir.Var("output", tirx.TensorType([4], "float32", data_alignment=4))
    index = ir.Var("index", "int32")
    statements = []
    if padding is not None:
        value = (
            ir.prim.Shuffle([ir.FloatImm("float32", x) for x in padding], [0, 1])
            if isinstance(padding, tuple)
            else ir.FloatImm("float32", padding)
        )
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"safe_value_map": {source: value}})
        statements.append(
            ir.Evaluate(
                ir.Call(
                    "tl.annotation",
                    [payload],
                    attrs={
                        "schema_version": 1,
                        "annotation_schema": schema,
                    },
                )
            )
        )
    statements += [
        ir.Bind(index, ir.IntImm("int32", 5)),
        ir.TensorStore(output, [ir.prim.Ramp(0, 1, 4)], ir.TensorLoad(source, [ir.prim.Ramp(index, 1, 2)])),
    ]
    function = tirx.Function([source, output], statements, ir.NoneType()).with_attr("global_symbol", "main")
    mod = tr.LegalizeSafeMemoryAccess()(ir.IRModule({"main": function}))
    mod = tirx.transform.StmtSimplify()(mod)
    loads = []
    tvm_ffi.structural_walk(mod, (ir.TensorLoad, loads.append), order="pre")
    assert not loads  # The proven out-of-bounds access must not execute.
    # The eliminated source has a vector DLPack dtype. Drop only this unused
    # parameter to execute the generated fallback with a standard NumPy oracle.
    fallback = tirx.Function([output], mod["main"].body, ir.NoneType()).with_attr("global_symbol", "main")
    assert tirx.analysis.check_well_formed(fallback)
    executable = compile_packed_function(fallback, c_codegen="compute")
    actual = np.full(4, 97.0, dtype="float32")
    executable(tvm_ffi.from_dlpack(actual))
    expected = [0.0] * 4 if padding is None else list(padding) * 2 if isinstance(padding, tuple) else [padding] * 4
    np.testing.assert_array_equal(actual, expected)
