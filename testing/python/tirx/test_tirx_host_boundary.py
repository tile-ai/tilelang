import numpy as np
import pytest
import tvm_ffi

from tvm_tirx import ir, runtime, target, tirx
from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def transform(name, function, *args):
    result = tvm_ffi.get_global_func(f"tl.transform.{name}")(*args)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    ir.assert_structural_equal(result, tirx.renew_def(result))
    return result


def scope(body, attrs=None, operation="tirx.scope", args=None, params=None):
    return ir.RegionStmt(ir.Op.get(operation), args or [], params or [], ir.DictAttrs(attrs or {}), body)


def collect(function, node_type):
    nodes = []
    stmt_functor.post_order_visit(function, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


def test_nonrestrict_metadata_survives_parameter_pruning():
    unused = TensorView.create("unused", [1], "int32")
    source = TensorView.create("source", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    function = tirx.Function(
        [unused.tensor, source.tensor, output.tensor],
        scope([output.store(0, source.load(0))], {"tl.non_restrict_params": [source.tensor, output.data, source.tensor]}),
    ).with_attr("tl.readonly_param_indices", [ir.const(0), ir.const(1)])
    result = transform("HoistNonRestrictParams", function)
    assert [int(index) for index in result.attrs["tl.non_restrict_param_indices"]] == [1, 2]
    assert all("tl.non_restrict_params" not in region.attrs for region in collect(result, ir.RegionStmt))
    result = transform("Simplify", result, True)
    assert len(result.params) == 2
    assert [int(index) for index in result.attrs["tl.non_restrict_param_indices"]] == [0, 1]
    assert [int(index) for index in result.attrs["tl.readonly_param_indices"]] == [0]


def test_nonrestrict_parameter_name_fallback_and_ambiguity():
    source = TensorView.create("source", [1], "int32")
    output = TensorView.create("output", [1], "int32")
    placeholder = ir.Var("source_handle", source.tensor.ty)
    function = tirx.Function(
        [source.tensor, output.tensor], scope(output.store(0, source.load(0)), {"tl.non_restrict_params": [placeholder]})
    )
    result = transform("HoistNonRestrictParams", function)
    assert [int(index) for index in result.attrs["tl.non_restrict_param_indices"]] == [0]
    ambiguous = tirx.Function(
        [source.tensor, placeholder, output.tensor],
        scope(output.store(0, source.load(0)), {"tl.non_restrict_params": [ir.Var("source", source.tensor.ty)]}),
    )
    with pytest.raises(ValueError, match="Ambiguous non-restrict"):
        transform("HoistNonRestrictParams", ambiguous)


@pytest.mark.parametrize("kind", ["cuda", "c"])
def test_device_regions_are_idempotent(kind):
    index = ir.Var("block", "int32")
    output = TensorView.create("output", [2], "int32")
    body = scope(output.store(index, index), operation="tirx.launch_thread", args=[ir.StringImm("blockIdx.x"), ir.const(2)], params=[index])
    function = tirx.Function([output.tensor], body).with_attr("target", target.Target({"kind": kind, "host": "llvm"}))
    result = transform("AnnotateDeviceRegions", function)
    ir.assert_structural_equal(result, transform("AnnotateDeviceRegions", result))
    regions = [region for region in collect(result, ir.RegionStmt) if region.op.name == "tirx.device_scope"]
    assert len(regions) == 1
    assert regions[0].attrs["target"].kind.name == kind
    assert regions[0].attrs["target"].host is None


def test_explicit_device_target_and_separate_kernels_are_preserved():
    output = TensorView.create("output", [2], "int32")
    explicit = scope(output.store(0, 3), {"target": target.Target("c")}, operation="tirx.device_scope")
    index = ir.Var("block", "int32")
    implicit = scope(output.store(index, 5), operation="tirx.launch_thread", args=[ir.StringImm("blockIdx.x"), ir.const(2)], params=[index])
    function = tirx.Function([output.tensor], [explicit, implicit]).with_attr("target", target.Target({"kind": "cuda", "host": "llvm"}))
    result = transform("AnnotateDeviceRegions", function)
    kinds = [region.attrs["target"].kind.name for region in collect(result, ir.RegionStmt) if region.op.name == "tirx.device_scope"]
    assert kinds == ["c", "cuda"]


def test_global_allocations_keep_host_condition_and_alias_capture():
    enabled = ir.Var("enabled", "bool")
    output = TensorView.create("output", [1], "int32")
    allocation = TensorView.create("temporary", [2], "int32")
    alias = TensorView.create("alias", [1], "int32", elem_offset=1)
    device = scope(
        [allocation.allocate(), alias.declare(allocation.data), alias.store(0, 19), output.store(0, alias.load(0))],
        {"target": target.Target("c")},
        operation="tirx.device_scope",
    )
    function = tirx.Function([enabled, output.tensor], ir.If(enabled, device, None))
    result = transform("HoistGlobalBufferAllocations", function)
    assert isinstance(result.body.seq[0], ir.If)
    branch = result.body.seq[0].then_case
    assert isinstance(branch.seq[0], ir.Bind)
    assert branch.seq[0].var.same_as(allocation.tensor)
    assert branch.seq[1].op.name == "tirx.device_scope"
    ir.assert_structural_equal(result, transform("HoistGlobalBufferAllocations", result))
    executable_function = stmt_functor.ir_transform(
        result, None, lambda node: node.body if isinstance(node, ir.RegionStmt) and node.op.name == "tirx.device_scope" else None
    )
    executable_function = executable_function.with_attr("global_symbol", "conditional_allocation")
    executable = tirx.compile(executable_function, target="llvm")
    for condition, expected in [(False, 7), (True, 19)]:
        actual = runtime.tensor(np.array([7], dtype="int32"))
        executable(condition, actual)
        np.testing.assert_array_equal(actual.numpy(), [expected])


def test_global_allocation_with_host_dynamic_shape():
    size = ir.Var("size", "int32")
    allocation = TensorView.create("temporary", [size], "int32")
    function = tirx.Function([size], scope([allocation.allocate(), allocation.store(0, 1)], operation="tirx.device_scope"))
    result = transform("HoistGlobalBufferAllocations", function)
    assert isinstance(result.body.seq[0], ir.Bind)
    assert result.body.seq[1].op.name == "tirx.device_scope"


def test_global_allocation_rejects_device_local_shape():
    index = ir.Var("block", "int32")
    allocation = TensorView.create("temporary", [index + 1], "int32")
    function = tirx.Function(
        [],
        scope(
            [allocation.allocate(), allocation.store(0, 1)],
            operation="tirx.launch_thread",
            args=[ir.StringImm("blockIdx.x"), ir.const(2)],
            params=[index],
        ),
    )
    with pytest.raises(ValueError, match="depends on device-local variable"):
        transform("HoistGlobalBufferAllocations", function)


def split(function):
    module = tvm_ffi.get_global_func("tl.transform.SplitHostDevice")()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return module


def test_split_remaps_metadata_by_capture_not_parameter_order():
    source = TensorView.create("z_source", [1], "int32")
    output = TensorView.create("a_output", [1], "int32")
    function = tirx.Function([source.tensor, output.tensor], output.store(0, source.load(0)))
    function = function.with_attr("target", target.Target("c", host="c"))
    function = function.with_attr("tl.non_restrict_param_indices", [ir.const(0)])
    function = function.with_attr("tl.readonly_param_indices", [ir.const(0)])
    function = function.with_attr("tl.smem_alignment_map", {"shared": ir.const(128)})
    module = split(function)
    kernel = module["main_kernel"]
    assert [param.name for param in kernel.params] == ["a_output_ptr", "z_source_ptr"]
    assert [int(index) for index in kernel.attrs["tl.non_restrict_param_indices"]] == [1]
    assert [int(index) for index in kernel.attrs["tl.readonly_param_indices"]] == [1]
    assert int(kernel.attrs["tl.smem_alignment_map"]["shared"]) == 128
    assert not kernel.attrs["tirx.noalias"]


def test_split_remaps_nonrestrict_host_aliases():
    source = TensorView.create("source", [2], "int32")
    alias = TensorView.create("alias", [1], "int32", elem_offset=1)
    output = TensorView.create("output", [1], "int32")
    body = scope(output.store(0, alias.load(0)), {"target": target.Target("cuda")}, operation="tirx.device_scope")
    function = tirx.Function([source.tensor, output.tensor], [alias.declare(source.data), body])
    function = function.with_attr("target", target.Target("cuda", host="llvm"))
    function = function.with_attr("tl.non_restrict_param_indices", [ir.const(0)])
    module = split(function)
    kernel = module["main_kernel"]
    assert kernel.params[0].name == "alias_ptr"
    assert [int(index) for index in kernel.attrs["tl.non_restrict_param_indices"]] == [0]


@pytest.mark.parametrize("kind", ["c", "cuda"])
def test_split_lifts_host_evaluable_device_assumptions(kind):
    size = ir.Var("size", "int32")
    output = TensorView.create("output", [1], "int32")
    body = [ir.Evaluate(ir.Call("prim.assume", [size > 0])), output.store(0, size + 7)]
    if kind == "cuda":
        thread = ir.Var("thread", "int32")
        body = scope(body, operation="tirx.launch_thread", args=[ir.StringImm("threadIdx.x"), ir.const(1)], params=[thread])
        device = runtime.cuda(0)
        if not device.exist:
            pytest.skip("A CUDA device is required")
        device_target = target.Target(
            {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
            host="c",
        )
    else:
        device = runtime.cpu()
        device_target = target.Target("c", host="c")
    function = tirx.Function([size, output.tensor], body).with_attr("target", device_target).with_attr("global_symbol", "assumed_kernel")
    function = transform("InjectAssumes", function)
    module = split(function)
    assert len(collect(module["main"], ir.AssertStmt)) >= 1
    assert not collect(module["assumed_kernel_kernel"], ir.AssertStmt)
    executable = tirx.compile(module, target=device_target)
    actual = runtime.empty((1,), "int32", device=device)
    executable(5, actual)
    np.testing.assert_array_equal(actual.numpy(), [12])
    with pytest.raises(RuntimeError, match="Assume"):
        executable(0, actual)


def test_split_does_not_hoist_conditional_or_thread_dependent_assumptions():
    size = ir.Var("size", "int32")
    thread = ir.Var("thread", "int32")
    first = ir.Evaluate(ir.Call("prim.assume", [thread >= 0]))
    second = ir.If(thread == 0, ir.Evaluate(ir.Call("prim.assume", [size > 0])), None)
    body = scope([first, second], operation="tirx.launch_thread", args=[ir.StringImm("threadIdx.x"), ir.const(2)], params=[thread])
    function = tirx.Function([size], body).with_attr("target", target.Target("cuda", host="llvm"))
    module = split(transform("InjectAssumes", function))
    assert not collect(module["main"], ir.AssertStmt)
    assert not any("tl.assume_requires_runtime_check" in region.attrs for region in collect(module, ir.RegionStmt))


@pytest.mark.parametrize("parallel", [False, True])
def test_cpu_launch_metadata_survives_materialization(parallel):
    from tvm_tirx.script.ir_builder import IRBuilder
    from tvm_tirx.tirx.script import ir_builder as builder

    with IRBuilder() as context:
        frame = tvm_ffi.get_global_func("tl.KernelLaunch")([ir.const(2), ir.const(3)], None, {"tl.cpu_num_threads": 4})
        with frame:
            builder.evaluate(sum(frame.grid_vars))
    function = tirx.Function([], context.get())
    module = ir.IRModule({"main": function})
    with ir.transform.PassContext(config={"tl.cpu_parallel": parallel}):
        module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUKernelLaunch")()(module)
    module = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, False, None, None, None)(module)
    assert tirx.analysis.check_well_formed(module)
    loops = list(reversed(collect(module, ir.For)))
    assert len(loops) == 2
    assert int(loops[0].annotations["tl.cpu_num_threads"]) == 4
    assert "tl.cpu_num_threads" not in loops[1].annotations
    for axis, loop in enumerate(loops):
        if parallel:
            assert int(loop.annotations["tl.cpu_grid_dim"]) == axis
        else:
            assert "tl.cpu_grid_dim" not in loop.annotations


@pytest.mark.gpu
@pytest.mark.parametrize("annotation_site", ["function", "region"])
def test_cluster_launch_uses_native_runtime_and_preserves_argument_order(annotation_site):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    device_target = target.Target(
        {"kind": "cuda", "arch": f"sm_{device.compute_version.replace('.', '')}", "backend_config": {"cuda": {"compiler": "nvcc"}}},
        host="c",
    )
    output = TensorView.create("output", [4], "int32")
    block, thread = ir.Var("block", "int32"), ir.Var("thread", "int32")
    body = scope(
        ir.If(thread == 0, output.store(block, block + 9), None),
        operation="tirx.launch_thread",
        args=[ir.StringImm("threadIdx.x"), ir.const(32)],
        params=[thread],
    )
    dimensions = [ir.const(2), ir.const(1), ir.const(1)]
    if annotation_site == "region":
        body = scope(body, {"cluster_dims": dimensions})
    body = scope(body, operation="tirx.launch_thread", args=[ir.StringImm("blockIdx.x"), ir.const(4)], params=[block])
    function = tirx.Function([output.tensor], body).with_attr("target", device_target).with_attr("global_symbol", "cluster_test")
    if annotation_site == "function":
        function = function.with_attr("cluster_dims", dimensions)
    module = split(function)
    lower = tvm_ffi.get_global_func("tl.transform.LowerDeviceKernelLaunch")()
    module = lower(module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, lower(module))
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    tags = list(module["cluster_test_kernel"].attrs["tirx.kernel_launch_params"])
    assert tags[:3] == ["clusterCtaIdx.x", "clusterCtaIdx.y", "clusterCtaIdx.z"]
    calls = [call for call in collect(module["main"], ir.Call) if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_ffi_kernel"]
    assert len(calls) == 1
    assert [int(argument) for argument in list(calls[0].args)[2:5]] == [2, 1, 1]
    actual = runtime.empty((4,), "int32", device=device)
    executable = tirx.compile(module, target=device_target)
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [9, 10, 11, 12])


@pytest.mark.parametrize("dimensions", [[0, 1, 1], [-1, 1, 1], [1, 1]])
def test_invalid_cluster_dimensions_are_rejected(dimensions):
    thread = ir.Var("thread", "int32")
    body = scope(ir.Evaluate(thread), operation="tirx.launch_thread", args=[ir.StringImm("threadIdx.x"), ir.const(32)], params=[thread])
    function = (
        tirx.Function([], body)
        .with_attr("target", target.Target("cuda", host="c"))
        .with_attr("cluster_dims", [ir.const(value) for value in dimensions])
    )
    with pytest.raises(ValueError, match="(cluster|clusters)"):
        tvm_ffi.get_global_func("tl.transform.LowerDeviceKernelLaunch")()(split(function))
