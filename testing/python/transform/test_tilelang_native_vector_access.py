"""Vector memory dependencies include every lane, before and after unswitching."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, sym, tirx

import tilelang
from tilelang.testing.packed import compile_packed_function


def _index(case):
    if case == "scalar":
        return ir.const(3), [3]
    if case == "ramp":
        return ir.prim.Ramp(3, 2, 4), [3, 5, 7, 9]
    if case == "descending":
        return ir.prim.Ramp(12, -3, 4), [12, 9, 6, 3]
    if case == "broadcast":
        return ir.prim.Broadcast(ir.const(7), 4), [7] * 4
    if case == "cast":
        return ir.prim.Cast("int64x4", ir.prim.Ramp(3, 2, 4)), [3, 5, 7, 9]
    if case == "wrap":
        # Narrow after a supported Ramp; the fixed-width lane oracle is unchanged.
        ramp = ir.prim.Ramp(ir.const(254, "int32"), ir.const(1, "int32"), 4)
        return ir.prim.Cast("uint8x4", ramp), [254, 255, 0, 1]
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["scalar", "ramp", "descending", "broadcast", "cast", "wrap"])
def test_access_region_covers_all_lanes(case):
    index, expected = _index(case)
    tensor = ir.Var("A", tirx.TensorType((256,), "int32"))
    function = tirx.Function([tensor], [ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index]))])
    original = ir.save_json(function)
    accesses = tvm_ffi.get_global_func("tl.testing.CollectPipelineAccesses")(function)
    assert len(accesses["reads"]) == len(accesses["writes"]) == 1
    analyzer = sym.Analyzer()
    for region in [accesses["reads"][0], accesses["writes"][0]]:
        assert region.source.same_as(tensor)
        bounds = region.region[0]
        assert bounds.begin.ty.dtype.lanes == bounds.extent.ty.dtype.lanes == 1
        begin = int(analyzer.simplify(bounds.begin))
        end = begin + int(analyzer.simplify(bounds.extent))
        assert all(begin <= lane < end for lane in expected)
        if case in ("scalar", "ramp", "descending", "broadcast"):
            assert (begin, end) == (min(expected), max(expected) + 1)
        if case == "wrap":
            assert accesses["unknown"] and (begin, end) == (0, 256)
    assert ir.save_json(function) == original


def test_unswitching_preserves_guard_written_by_last_vector_lane():
    source = ir.Var("A", tirx.TensorType((8,), "int32"))
    output = ir.Var("B", tirx.TensorType((4,), "int32"))
    loop = ir.Var("i", "int32")
    body = [
        ir.If(
            ir.TensorLoad(source, [7]) == 0,
            [ir.TensorStore(output, [ir.prim.Ramp(0, 1, 4)], ir.prim.Broadcast(loop + 10, 4))],
            None,
        ),
        ir.TensorStore(source, [ir.prim.Ramp(4, 1, 4)], ir.prim.Broadcast(loop + 1, 4)),
    ]
    function = tirx.Function([source, output], [ir.For(loop, 0, 2, 1, ir.ForKind.DEFAULT, body)]).with_attr("global_symbol", "main")
    module = ir.IRModule({"main": function})
    original = ir.save_json(module)
    transformed = tilelang.transform.LoopUnswitching()(module)
    assert tirx.analysis.check_well_formed(transformed)
    assert ir.save_json(module) == original
    executable = compile_packed_function(transformed, c_codegen="compute")
    source_data = runtime.tensor(np.zeros(8, dtype="int32"))
    output_data = runtime.tensor(np.zeros(4, dtype="int32"))
    executable(source_data, output_data)
    np.testing.assert_array_equal(source_data.numpy(), np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype="int32"))
    np.testing.assert_array_equal(output_data.numpy(), np.full(4, 10, dtype="int32"))
