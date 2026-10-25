import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def legalize(body, params):
    function = tirx.Function(params, body, ret_type=ir.TupleType([])).with_attr("global_symbol", "negative_indices")
    result = tvm_ffi.get_global_func("tl.transform.LegalizeNegativeIndex")()(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    return result


@pytest.mark.parametrize("nonunit_step", [False, True])
def test_negative_indices_use_native_loop_domains(nonunit_step):
    source = TensorView.create("source", [8], "int32")
    output = TensorView.create("output", [4], "int32")
    index = ir.Var("index", "int32")
    if nonunit_step:
        body = ir.For(index, -8, 0, 2, ir.ForKind.DEFAULT, output.store((index + 8) // 2, source.load(index)))
    else:
        body = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, output.store(index, source.load(index - 4)))
    executable = tirx.compile(legalize(body, [source.tensor, output.tensor]), target="c")
    actual = runtime.empty((4,), "int32")
    executable(runtime.tensor(np.arange(8, dtype="int32")), actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(0, 8, 2) if nonunit_step else np.arange(4, 8))


def test_negative_indices_use_branch_constraints_without_leaking_them():
    source = TensorView.create("source", [8], "int32")
    output = TensorView.create("output", [1], "int32")
    index = ir.Var("index", "int32")
    body = ir.If(index < 0, output.store(0, source.load(index)), output.store(0, source.load(index)))
    executable = tirx.compile(legalize(body, [source.tensor, output.tensor, index]), target="c")
    actual = runtime.empty((1,), "int32")
    for offset in (-8, -3, -1, 0, 2, 7):
        executable(runtime.tensor(np.arange(8, dtype="int32")), actual, offset)
        np.testing.assert_array_equal(actual.numpy(), [offset % 8])


def test_native_tensor_regions_keep_negative_slice_extents():
    source = TensorView.create("source", [8], "float32")
    output = TensorView.create("output", [3], "float32")
    region = source.region([ir.Range(-4, -1)])
    body = ir.Evaluate(ir.Call("tl.tileop.copy", [region, output.tensor]))
    function = legalize(body, [source.tensor, output.tensor])
    regions = []
    stmt_functor.post_order_visit(function, lambda node: regions.append(node) if isinstance(node, ir.TensorRegion) else None)
    assert len(regions) == 1
    assert int(regions[0].region[0].begin) == 4
    assert int(regions[0].region[0].extent) == 3


@pytest.mark.parametrize("base,stride,expected", [(-2, 1, [6, 7, 0, 1]), (1, -1, [1, 0, 7, 6]), (-6, 1, [2, 3, 4, 5])])
@pytest.mark.gpu
def test_vector_negative_indices_preserve_each_lane(base, stride, expected):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA is required for vector gather execution")
    source = TensorView.create("source", [8], "int32")
    output = TensorView.create("output", [4], "int32")
    body = output.store(prim.Ramp(0, 1, 4), source.load(prim.Ramp(base, stride, 4)))
    thread = ir.Var("thread", "int32")
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, body)
    function = legalize(body, [source.tensor, output.tensor])
    target = Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    executable = tirx.compile(function, target=target)
    actual = runtime.empty((4,), "int32", device=device)
    executable(runtime.tensor(np.arange(8, dtype="int32"), device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
