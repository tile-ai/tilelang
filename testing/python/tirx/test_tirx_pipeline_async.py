import re

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx, transform
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def collect(root, node_type):
    nodes = []
    stmt_functor.post_order_visit(root, lambda node: nodes.append(node) if isinstance(node, node_type) else None)
    return nodes


@pytest.mark.parametrize("pattern", ["closed", "missing_commit", "missing_wait", "nonzero_wait", "intervening_read", "outside", "explicit"])
def test_normalize_only_closed_pipeline_cp_async_groups(pattern):
    source = TensorView.create("source", [5, 4], "int32")
    output = TensorView.create("output", [5], "int32")
    shared = TensorView.create("shared", [4], "int32", scope="shared")
    iteration = ir.Var("iteration", "int32")
    source_ptr = ir.Call("tl.access_ptr", [source.load([iteration, 0]), 4, 1])
    shared_ptr = ir.Call("tl.access_ptr", [shared.load(0), 4, 2])
    copy = ir.Evaluate(ir.Call("tl.ptx_cp_async", [shared_ptr, source_ptr, 4], ty="void"))
    commit = ir.Evaluate(ir.Call("tirx.ptx.cp_async_commit_group", ["async", "commit_group", ""]))
    wait = ir.Evaluate(ir.Call("tirx.ptx.cp_async_wait_group", [1 if pattern == "nonzero_wait" else 0, "async", "wait_group", ""]))
    consumer = output.store(iteration, shared.load(0))
    statements = [copy]
    if pattern != "missing_commit":
        statements.append(commit)
    if pattern == "intervening_read":
        statements.append(consumer)
    if pattern != "missing_wait":
        statements.append(wait)
    statements.append(consumer)
    annotations = {"num_stages": 2}
    if pattern == "explicit":
        annotations.update(software_pipeline_stage=[0, 0, 0, 1], software_pipeline_order=[0, 1, 2, 3])
    loop = ir.For(iteration, 0, 5, 1, ir.ForKind.DEFAULT, statements, annotations=annotations)
    body = [shared.allocate(), loop]
    if pattern == "outside":
        body.append(commit)
    function = tirx.Function([source.tensor, output.tensor], body).with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))
    before = ir.IRModule({"main": function})
    after = tvm_ffi.get_global_func("tl.transform.NormalizePipelineCPAsync")()(before)
    assert tirx.analysis.check_well_formed(after)
    if pattern == "closed":
        calls = collect(after, ir.Call)
        assert any(call.op.name == "tl.ptx_cp_async" for call in calls)
        assert not any(call.op.name.startswith("tirx.ptx.cp_async_") for call in calls)
        planned = tvm_ffi.get_global_func("tl.transform.PipelinePlanning")()(after)
        lowered = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(planned)
        controls = [call.op.name for call in collect(lowered, ir.Call)]
        assert "tirx.ptx.cp_async_commit_group" in controls
        assert "tirx.ptx.cp_async_wait_group" in controls
    else:
        ir.assert_structural_equal(after, before)


def grouped_pipeline(distance=2, *, explicit=False, target="cuda", enabled=True):
    thread = ir.Var("thread", "int32")
    iteration = ir.Var("iteration", "int32")
    inputs = [TensorView.create(f"input_{group}", [5, 32], "float32") for group in range(3)]
    outputs = [TensorView.create(f"output_{group}", [5, 32], "float32") for group in range(3)]
    shared = [TensorView.create(f"shared_{group}", [32], "float32", scope="shared") for group in range(3)]
    body = []
    for source, destination, staging in zip(inputs, outputs, shared):
        body.extend([staging.store(thread, source.load([iteration, thread])), destination.store([iteration, thread], staging.load(thread))])
    attrs = {"software_pipeline_stage": [0, distance] * 3, "software_pipeline_order": [1, 0, 3, 2, 5, 4]}
    if explicit:
        attrs.update(
            software_pipeline_async_stages=[0],
            software_pipeline_async_producers=[1, 0] * 3,
            software_pipeline_async_producer_groups=[0, -1, 1, -1, 2, -1],
        )
    loop = ir.For(iteration, 0, 5, 1, ir.ForKind.DEFAULT, body, annotations=attrs)
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [thread],
        None,
        [*[tensor.allocate() for tensor in shared], loop],
    )
    function = tirx.Function([tensor.tensor for tensor in [*inputs, *outputs]], body, ret_type=ir.TupleType([]))
    if target is not None:
        function = function.with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}) if target == "cuda" else Target(target))
    with transform.PassContext(config={"tl.enable_async_copy": enabled}):
        module = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    return module


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("distance", [1, 2, 3])
def test_native_pipeline_waits_follow_physical_commit_groups(distance, explicit):
    module = grouped_pipeline(distance, explicit=explicit)
    loops = [loop for loop in collect(module, ir.For) if loop.annotations.get("tl.pipeline_injected")]
    assert len(loops) == 1
    calls = collect(loops[0], ir.Call)
    commits = [call for call in calls if call.op.name == "tirx.ptx.cp_async_commit_group"]
    waits = [int(call.args[0]) for call in calls if call.op.name == "tirx.ptx.cp_async_wait_group"]
    assert len(commits) == 3
    expected = min(distance * 3 - 1, 7)
    assert waits == ([expected, 7] * 3 if (distance + 1) * 3 > 8 else [expected] * 3)
    all_waits = [int(call.args[0]) for call in collect(module, ir.Call) if call.op.name == "tirx.ptx.cp_async_wait_group"]
    assert all_waits == [*waits, 0]
    scopes = [node for node in collect(module, ir.RegionStmt) if node.attrs.get("tl.pipeline_cp_async")]
    assert len(scopes) == 3


@pytest.mark.parametrize("target", [None, "c"])
def test_native_explicit_async_pipeline_requires_compatible_target(target):
    with pytest.raises(ValueError, match="CUDA target.*cp.async"):
        grouped_pipeline(explicit=True, target=target)


def test_native_disabled_automatic_pipeline_uses_synchronous_copies():
    module = grouped_pipeline(enabled=False)
    assert not [call for call in collect(module, ir.Call) if call.op.name.startswith("tirx.ptx.cp_async")]


@pytest.mark.parametrize("operation", ["cp_async_commit_group", "cp_async_wait_group", "cp_async_wait_all"])
def test_native_pipeline_rejects_untracked_manual_groups(operation):
    source = TensorView.create("source", [5], "int32")
    shared = TensorView.create("shared", [1], "int32", scope="shared")
    output = TensorView.create("output", [5], "int32")
    iteration = ir.Var("iteration", "int32")
    arguments = [ir.StringImm("async"), ir.StringImm(operation.removeprefix("cp_async_")), ir.StringImm("")]
    if operation == "cp_async_wait_group":
        arguments.insert(0, ir.const(0))
    control = ir.Evaluate(ir.Call(f"tirx.ptx.{operation}", arguments))
    loop = ir.For(
        iteration,
        0,
        5,
        1,
        ir.ForKind.DEFAULT,
        [
            shared.store(0, source.load(iteration)),
            control,
            output.store(iteration, shared.load(0)),
        ],
        annotations={"software_pipeline_stage": [0, 0, 1], "software_pipeline_order": [0, 1, 2]},
    )
    function = tirx.Function([source.tensor, output.tensor], [shared.allocate(), loop], ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))
    with pytest.raises(ValueError, match="Manual cp.async commit/wait"):
        tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(ir.IRModule({"main": function}))


def test_native_pipeline_rejects_excluded_explicit_async_copy():
    source = TensorView.create("source", [5], "int32")
    shared = TensorView.create("shared", [1], "int32", scope="shared")
    output = TensorView.create("output", [5], "int32")
    iteration = ir.Var("iteration", "int32")
    copy = ir.Evaluate(ir.Call("tl.tileop.async_copy", [source.region([ir.Range.from_min_extent(iteration, 1)]), shared.tensor]))
    loop = ir.For(
        iteration,
        0,
        5,
        1,
        ir.ForKind.DEFAULT,
        [copy, output.store(iteration, shared.load(0))],
        annotations={
            "software_pipeline_stage": [0, 1],
            "software_pipeline_order": [0, 1],
            "software_pipeline_async_producers": [0, 0],
        },
    )
    function = tirx.Function([source.tensor, output.tensor], [shared.allocate(), loop], ret_type=ir.TupleType([]))
    function = function.with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))
    with pytest.raises(ValueError, match="Explicit asynchronous copies must belong"):
        tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(ir.IRModule({"main": function}))


@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("planned", [False, True])
def test_native_cuda_pipeline_waits_before_explicit_copy_index_reads(depth, automatic, planned):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for indexed asynchronous copies")
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def kernel(
        source: T.Tensor((11, 128), T.float32), positions: T.Tensor((11,), T.int32), output: T.Tensor((11, 128), T.float32), steps: T.int32
    ):
        with T.Kernel(1, threads=32):
            position_shared = T.alloc_shared((1,), T.int32)
            shared = T.alloc_shared((128,), T.float32)
            for iteration in T.Pipelined(
                steps,
                num_stages=depth if planned else 0,
                stage=None if planned else [0, 0, depth - 1],
                order=None if planned else [0, 1, 2],
            ):
                T.async_copy(positions[iteration], position_shared)
                T.async_copy(source[position_shared[0], :], shared)
                for coordinate in T.Parallel(128):
                    output[iteration, coordinate] = shared[coordinate] + 1

    compiled = tilelang.compile(kernel, target="cuda", execution_backend="tvm_ffi", pass_configs={"tl.enable_async_copy": automatic})
    source = torch.randn(11, 128, device="cuda")
    positions = torch.randperm(11, device="cuda", dtype=torch.int32)
    for steps in [0, 1, 2, 5, 11]:
        output = torch.full_like(source, -99)
        compiled(source, positions, output, steps)
        expected = torch.full_like(source, -99)
        expected[:steps] = source[positions[:steps].long()] + 1
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("depth", [2, 4])
def test_native_cuda_pipeline_empty_short_and_multigroup_execution(depth):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for asynchronous pipeline execution")
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def kernel(
        first: T.Tensor((11, 128), T.float32),
        second: T.Tensor((11, 128), T.float32),
        third: T.Tensor((11, 128), T.float32),
        output_first: T.Tensor((11, 128), T.float32),
        output_second: T.Tensor((11, 128), T.float32),
        output_third: T.Tensor((11, 128), T.float32),
        steps: T.int32,
    ):
        with T.Kernel(1, threads=32):
            shared_first = T.alloc_shared((128,), T.float32)
            shared_second = T.alloc_shared((128,), T.float32)
            shared_third = T.alloc_shared((128,), T.float32)
            for iteration in T.Pipelined(steps, stage=[0, depth - 1] * 3, order=[1, 0, 3, 2, 5, 4]):
                T.copy(first[iteration, :], shared_first, disable_tma=True)
                for index in T.Parallel(128):
                    output_first[iteration, index] = shared_first[index] + 1
                T.copy(second[iteration, :], shared_second, disable_tma=True)
                for index in T.Parallel(128):
                    output_second[iteration, index] = shared_second[index] + 2
                T.copy(third[iteration, :], shared_third, disable_tma=True)
                for index in T.Parallel(128):
                    output_third[iteration, index] = shared_third[index] + 3

    compiled = tilelang.compile(kernel, target="cuda", execution_backend="tvm_ffi")
    source = compiled.get_kernel_source()
    assert len(re.findall(r"^\s+ptx_cp_async_legacy_\w+\(", source, re.MULTILINE)) == 3
    inputs = [torch.randn(11, 128, device="cuda") for _ in range(3)]
    for steps in [0, 1, 2, 5, 11]:
        outputs = [torch.full_like(tensor, -99) for tensor in inputs]
        compiled(*inputs, *outputs, steps)
        for group, (input_tensor, output_tensor) in enumerate(zip(inputs, outputs)):
            expected = torch.full_like(input_tensor, -99)
            expected[:steps] = input_tensor[:steps] + (group + 1)
            np.testing.assert_array_equal(output_tensor.cpu().numpy(), expected.cpu().numpy())


@pytest.mark.parametrize("depth", [2, 3])
def test_native_cuda_pipeline_preserves_manual_tma_barrier_lifetime(depth):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA-capable CUDA device required")
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def kernel(source: T.Tensor((4, 128), T.float16), output: T.Tensor((4, 128), T.float16)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((128,), T.float16)
            barrier = T.alloc_barrier(1)
            for iteration in T.Pipelined(4, num_stages=depth):
                T.tma_copy(source[iteration, :], shared, barrier=barrier)
                if T.get_thread_binding() == 0:
                    T.barrier_arrive(barrier)
                T.mbarrier_wait_parity(barrier, iteration % 2)
                T.copy(shared, output[iteration, :])

    compiled = tilelang.compile(kernel, target="cuda", execution_backend="tvm_ffi")
    for repeat in range(3):
        source = torch.randn(4, 128, device="cuda", dtype=torch.float16) + repeat
        output = torch.empty_like(source)
        compiled(source, output)
        torch.testing.assert_close(output, source, rtol=0, atol=0)
