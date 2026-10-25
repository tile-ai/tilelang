import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as builder_api

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func("tl." + name)(*arguments)


def nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def validate(body):
    function = tirx.Function([], body)
    assert tirx.analysis.check_well_formed(function)
    ir.assert_structural_equal(function, ir.load_json(ir.save_json(function)))
    ir.assert_structural_equal(function, tirx.renew_def(function))
    return function


@pytest.mark.parametrize(
    "kind,arguments,name",
    [
        ("Cube", [], "CUBE"),
        ("Vector", [2], "VECTOR"),
        ("SimdVF", [11, 3], "SIMD_VF"),
        ("SimtVF", [[ir.const(32), ir.const(2), ir.const(1)], 17, 4], "SIMT_VF"),
    ],
)
def test_ascend_frames_keep_allocations_inside_native_scopes(kind, arguments, name):
    with IRBuilder() as builder:
        frame = native(kind, *arguments)
        with frame:
            scratch = TensorView.create("scratch", [1], "int32", scope="local")
            builder_api.emit_(scratch.allocate())
            value = frame.thread_vars[0] if kind == "SimtVF" else frame.sid_var if kind == "Vector" else ir.const(1)
            builder_api.emit_(scratch.store(0, value))
    function = validate(builder.get())
    region = [region for region in nodes(function, ir.RegionStmt) if region.attrs.get("name_hint") == name][0]
    assert len(nodes(region.body, ir.Bind)) == 1
    assert not nodes(function, ir.For)
    if kind == "SimtVF":
        launches = [region for region in nodes(region, ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
        assert len(launches) == 3
        assert int(region.attrs["tl.vf_latency"]) == 17
        assert int(region.attrs["tl.vf_source_index"]) == 4
    if kind == "SimdVF":
        assert int(region.attrs["tl.vf_latency"]) == 11
        assert int(region.attrs["tl.vf_source_index"]) == 3


def test_nested_ascend_frames_do_not_steal_parent_or_sibling_storage():
    with IRBuilder() as builder:
        with native("Cube"):
            outer = TensorView.create("outer", [1], "int32", scope="local")
            builder_api.emit_(outer.allocate())
            with native("SimdVF", 0, 0):
                inner = TensorView.create("inner", [1], "int32", scope="local")
                builder_api.emit_(inner.allocate())
                builder_api.emit_(inner.store(0, 5))
            builder_api.emit_(outer.store(0, 3))
        with native("SimdVF", 0, 1):
            sibling = TensorView.create("sibling", [1], "int32", scope="local")
            builder_api.emit_(sibling.allocate())
            builder_api.emit_(sibling.store(0, 7))
    function = validate(builder.get())
    scopes = [region for region in nodes(function, ir.RegionStmt) if region.attrs.get("name_hint") == "SIMD_VF"]
    assert [nodes(region, ir.Bind)[0].var.name for region in scopes] == ["inner", "sibling"]


@pytest.mark.parametrize("vector_count", [1, 2])
def test_mixed_kernel_uses_explicit_grid_and_vector_launches(vector_count):
    with IRBuilder() as builder:
        frame = native("MixedKernelLaunch", [ir.const(8)], ir.const(vector_count), {})
        with frame:
            builder_api.evaluate(frame.grid_vars[0] + frame.grid_vars[1])
    function = validate(builder.get())
    module = native("transform.MaterializeKernelLaunch", True, False, None, None, ["cthread"])(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    launches = [region for region in nodes(module, ir.RegionStmt) if region.op.name == "tirx.launch_thread"]
    assert {str(region.args[0].value) for region in launches} == {"blockIdx.x", "cthread"}


@pytest.mark.parametrize(
    "kind,arguments",
    [
        ("MixedKernelLaunch", [[ir.const(2), ir.const(3)], ir.const(2), {}]),
        ("MixedKernelLaunch", [[ir.const(2)], ir.const(3), {}]),
        ("Vector", [0]),
        ("Vector", [3]),
        ("SimtVF", [[ir.const(32)], 0, 0]),
        ("SimtVF", [[ir.const(32), ir.const(0), ir.const(1)], 0, 0]),
        ("SimdVF", [-1, 0]),
    ],
)
def test_ascend_frames_reject_invalid_launch_configuration(kind, arguments):
    with pytest.raises(ValueError):
        native(kind, *arguments)
