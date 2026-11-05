"""Evaluate the analysis Range using native C compute, independently of IntSet."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx

import tilelang  # noqa: F401  -- register the real TileLang collector
from tilelang.testing.packed import compile_packed_function


def _typed_ramp(base, stride, lanes, dtype):
    if ir.PrimType(dtype).dtype.bits < 32:
        # Native Ramp supports 32/64-bit elements. Narrow the completed lane
        # value, retaining the exact fixed-width wrap of the original fixture.
        wide = ir.prim.Ramp(ir.prim.Cast("int32", base), ir.const(stride, "int32"), lanes)
        return ir.prim.Cast(f"{dtype}x{lanes}", wide)
    return ir.prim.Ramp(base, ir.const(stride, dtype), lanes)


def _evaluate(bounds):
    out = ir.Var("out", tirx.TensorType((3,), "int64"))
    exprs = [bounds.begin, bounds.extent, bounds.begin + bounds.extent]
    function = tirx.Function([out], [ir.TensorStore(out, [i], ir.prim.Cast("int64", expr)) for i, expr in enumerate(exprs)]).with_attr(
        "global_symbol", "main"
    )
    executable = compile_packed_function(function, c_codegen="compute")
    actual = runtime.tensor(np.zeros(3, dtype="int64"))
    executable(actual)
    return actual.numpy().tolist()


CASES = [
    ("int8", 0, 127, 2, [0, 127]),
    ("int8", 124, 1, 4, [124, 125, 126, 127]),
    ("int8", -1, 127, 2, [-1, 126]),
    # The legal native Ramp32 + lane cast keeps all 256 values in int8. Its
    # analysis range uses int64, so the exclusive end 128 is representable.
    ("int8", -128, 1, 256, list(range(-128, 128))),
    ("uint8", 255, 0, 1, [255]),
    ("uint8", 255, 0, 4, [255] * 4),
    ("int32", 0, (1 << 31) - 1, 2, [0, (1 << 31) - 1]),
    ("uint32", (1 << 32) - 1, 0, 4, [(1 << 32) - 1] * 4),
    ("int64", 12, -3, 4, [12, 9, 6, 3]),
]


@pytest.mark.parametrize("dtype,base,stride,lanes,expected", CASES)
def test_cpu_analysis_range_covers_nonwrapping_lanes(dtype, base, stride, lanes, expected):
    index = ir.const(base, dtype)
    if lanes != 1:
        index = ir.prim.Broadcast(index, lanes) if stride == 0 else _typed_ramp(index, stride, lanes, dtype)
    tensor = ir.Var("A", tirx.TensorType((256,), "int32"))
    function = tirx.Function([tensor], [ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index]))])
    analysis = tvm_ffi.get_global_func("tl.testing.CollectPipelineAccesses")(function)
    bounds = analysis["reads"][0].region[0]
    observed = _evaluate(bounds)
    print(
        {
            "case": [dtype, base, stride, lanes],
            "unknown": analysis["unknown"],
            "begin": bounds.begin.script(),
            "extent": bounds.extent.script(),
            "actual": observed,
            "expected_lanes": expected,
        }
    )
    begin, extent, end = observed
    assert extent > 0 and end == begin + extent
    assert all(begin <= lane < end for lane in expected)
    if dtype == "int8" and lanes == 256:
        assert not analysis["unknown"]
        assert observed == [-128, 256, 128]


@pytest.mark.parametrize(
    "dtype,base,stride,lanes",
    [
        ("int8", -128, -127, 2),
        ("int8", 127, 1, 2),
        ("int8", -127, -4, 4),
        ("int32", (1 << 31) - 1, 1, 4),
        ("uint8", 254, 1, 4),
        ("int64", 0, (1 << 63) - 1, 2),
        ("int64", (1 << 63) - 4, 1, 4),
        ("int64", (1 << 63) - 1, 0, 1),
    ],
)
def test_unrepresentable_range_conservatively_marks_unknown(dtype, base, stride, lanes):
    index = ir.const(base, dtype)
    if lanes != 1:
        index = _typed_ramp(index, stride, lanes, dtype)
    # The large-index cases are analysis-only negative tests, with no actual
    # tensor memory access. They cannot justify a finite disjointness proof.
    tensor = ir.Var("A", tirx.TensorType((256,), "int32"))
    function = tirx.Function([tensor], [ir.TensorStore(tensor, [index], ir.TensorLoad(tensor, [index]))])
    analysis = tvm_ffi.get_global_func("tl.testing.CollectPipelineAccesses")(function)
    print({"case": [dtype, base, stride, lanes], "unknown": analysis["unknown"], "region": analysis["reads"][0].script()})
    assert analysis["unknown"]
