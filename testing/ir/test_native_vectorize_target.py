"""Layout and vectorization use each Function's target without ambient state."""

from contextlib import nullcontext

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_fragment_tail import target
from test_native_reducer_contracts import allocate, integer, objects, region, tensor, tileop
from test_native_webgpu_ops import _function


def _program(kind, operation):
    thread = ir.Var("worker", "int32")
    shape = [8, 32] if operation == "transpose" else [1024]
    output_shape = [32, 8] if operation == "transpose" else shape
    source, output = tensor("A", shape, "float32", "global"), tensor("B", output_shape, "float32", "global")
    if operation == "parallel":
        index = ir.Var("i", "int32")
        body = [
            ir.For(
                index,
                integer(0),
                integer(1024),
                integer(1),
                ir.ForKind.PARALLEL,
                [ir.TensorStore(output, [index], ir.TensorLoad(source, [index]))],
            )
        ]
    elif operation in {"local", "local.var", "local.fragment"}:
        size = 257 if operation == "local.fragment" else 1
        private = tensor("F", [size], "float32", operation)
        output = tensor("B", [257 if operation == "local.fragment" else 128], "float32", "global")
        destination = (
            region(output)
            if operation == "local.fragment"
            else ir.TensorRegion(output, [ir.Range(thread, thread + 1)], ir.TensorRegionType())
        )
        body = [
            allocate(private),
            tileop("fill", region(private), ir.FloatImm("float32", 3.5)),
            tileop("copy", region(private), destination),
        ]
    elif operation == "fill":
        body = [tileop("fill", region(output), ir.FloatImm("float32", 2))]
    else:
        body = [tileop(operation, region(source), region(output))]
    return _function([source, output], body, thread=thread).with_attr("target", target(kind))


@pytest.mark.parametrize("kind", ["cuda", "webgpu", "metal"])
@pytest.mark.parametrize("operation", ["copy", "fill", "transpose", "parallel", "local", "local.var", "local.fragment"])
def test_public_layout_lowering_uses_function_target_with_absent_or_conflicting_context(kind, operation):
    value = _program(kind, operation)
    original = ir.save_json(value)
    expected = None
    for ambient in [None, target(kind), target("webgpu" if kind == "cuda" else "cuda")]:
        context = nullcontext() if ambient is None else ambient
        with context:
            current = tvm_tirx.target.Target.current()
            if ambient is None:
                assert current is None
            module = ir.IRModule({"main": value})
            inferred = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)
            lowered = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(inferred)
            if expected is None:
                expected = (inferred, lowered)
            else:
                ir.assert_structural_equal(expected[0], inferred)
                ir.assert_structural_equal(expected[1], lowered)
            assert tirx.analysis.check_well_formed(lowered)
            assert not tirx.analysis.undefined_vars(lowered["main"])
            assert objects(lowered, ir.TensorStore)
        assert tvm_tirx.target.Target.current() is None
    assert ir.save_json(value) == original
