import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView


pytestmark = pytest.mark.usefixtures("native_layouts")


def nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def transform(name, module, *arguments):
    return tvm_ffi.get_global_func("tl." + name)(*arguments)(module)


@pytest.mark.parametrize("tcgen05", [False, True])
@pytest.mark.parametrize("ordinal,alignment", [(0, 128), (1, 256), (2, 512), (3, 1024)])
def test_matrix_descriptors_preserve_swizzle_phase_alignment(tcgen05, ordinal, alignment):
    shared = TensorView.create("shared", [128, 128], "float16", scope="shared.dyn")
    descriptor = TensorView.create("descriptor", [1], "uint64", scope="local")
    encoding = 0 if ordinal == 0 else (4 - ordinal) * (2 if tcgen05 else 1)
    arguments = [descriptor.load(0), shared.pointer_at([0, 0])]
    arguments.extend([1, 1, 0, False, encoding] if tcgen05 else [encoding, 1, 1])
    name = "tl.initialize_tcgen05_descriptor" if tcgen05 else "tl.initialize_wgmma_descriptor"
    function = tirx.Function([], [shared.allocate(), descriptor.allocate(), ir.Evaluate(ir.Call(name, arguments))])
    function = function.with_attr("target", Target({"kind": "cuda", "arch": "sm_100a"}))
    module = transform("transform.LowerTileOp", ir.IRModule({"main": function}))
    assert int(module["main"].attrs["tl.smem_alignment_map"]["shared"]) == alignment
    allocations = [node for node in nodes(module, ir.Call) if isinstance(node.op, ir.Op) and node.op.name == "tirx.alloc_tensor"]
    assert next(allocation for allocation in allocations if allocation.ty.scope.name == "shared.dyn").ty.data_alignment == alignment
    assert tirx.analysis.check_well_formed(module)


def cuda_target():
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    return Target(
        {
            "kind": "cuda",
            "arch": "sm_" + device.compute_version.replace(".", ""),
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )


def module(body, tensors, target):
    function = tirx.Function([tensor.tensor for tensor in tensors], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "native_tile_lowering").with_attr("target", target)
    assert tirx.analysis.check_well_formed(function)
    return ir.IRModule({"main": function})


def lower(module, target):
    with target:
        module = transform("transform.LayoutInference", module)
        module = transform("transform.LowerTileOp", module)
        module = transform("transform.LowerAccessPtr", module)
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    assert not any(
        call.op.name.startswith("tl.tileop.") or call.op.name == "tl.layout_hint"
        for call in nodes(module, ir.Call)
        if isinstance(call.op, ir.Op)
    )
    return module


def tile(name, *arguments, annotations=None):
    return ir.Evaluate(ir.Call("tl.tileop." + name, list(arguments), attrs=annotations or {}))


@pytest.mark.parametrize("shape", [(257,), (7, 13), ()])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cpu_copy_fill_complete_lowering(shape, dtype):
    source = TensorView.create("source", shape, dtype)
    output = TensorView.create("output", shape, dtype)
    body = ir.SeqStmt([tile("fill", output.tensor, ir.const(-7, dtype)), tile("copy", source.tensor, output.tensor)])
    target = Target("c")
    executable = tirx.compile(lower(module(body, [source, output], target), target), target=target)
    expected = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    actual = runtime.empty(shape, dtype)
    executable(runtime.tensor(expected), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_native_aliases_share_remapped_fragment_storage(native_layouts):
    fragment = TensorView.create("fragment", [4, 8], "float32", scope="local.fragment")
    alias = TensorView.create("alias", [32], "float32", scope="local.fragment")
    output = TensorView.create("output", [32], "float32")
    layout = native_layouts.Fragment([4, 8], forward_thread_fn=lambda row, column: 0, forward_index_fn=lambda row, column: [column, row])
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    index = ir.Var("index", "int32")
    write = ir.For(
        row,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        ir.For(column, 0, 8, 1, ir.ForKind.DEFAULT, fragment.store([row, column], prim.Cast("float32", row * 8 + column))),
    )
    read = ir.For(index, 0, 32, 1, ir.ForKind.DEFAULT, output.store(index, alias.load(index)))
    hint = ir.Evaluate(ir.Call("tl.layout_hint", [fragment.tensor], attrs={"layout": layout}))
    body = ir.SeqStmt([fragment.allocate(), alias.declare(fragment.data), hint, write, read])
    target = Target("c")
    lowered = lower(module(body, [output], target), target)
    assert all(tensor.var.ty.scope.name != "local.fragment" for tensor in nodes(lowered, ir.Bind) if tirx.is_tensor_var(tensor.var))
    executable = tirx.compile(lowered, target=target)
    actual = runtime.empty((32,), "float32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(32, dtype="float32"))


@pytest.mark.parametrize("lower_barriers", [False, True])
def test_fragment_pointer_bindings_follow_physical_scope_remapping(native_layouts, lower_barriers):
    fragment = TensorView.create("fragment", [4], "uint32", scope="local.fragment")
    layout = native_layouts.Fragment([4], forward_thread_fn=lambda index: 0, forward_index_fn=lambda index: index)
    pointer = ir.Var("fragment_pointer", ir.PtrType("uint32", scope="local.fragment"))
    address = ir.Call("tl.access_ptr", [fragment.load(0), ir.const(4), ir.const(1)])
    consumer = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("consume_pointer"), pointer], ty="void"))
    hint = ir.Evaluate(ir.Call("tl.layout_hint", [fragment.tensor], attrs={"layout": layout}))
    body = ir.SeqStmt([fragment.allocate(), hint, ir.Bind(pointer, address), consumer])
    target = Target("c")
    result = lower(module(body, [], target), target)
    if lower_barriers:
        result = transform("cuda.transform.LowerSharedBarrier", result)
    bindings = [node for node in nodes(result, ir.Bind) if node.var.name == "fragment_pointer"]
    assert len(bindings) == 1
    assert bindings[0].var.ty.scope.name == "local"
    call = next(node for node in nodes(result, ir.Call) if node.op.name == "tirx.call_extern")
    assert call.args[1].same_as(bindings[0].var)
    assert not tirx.analysis.undefined_vars(result["main"].body, list(result["main"].params))


@pytest.mark.gpu
@pytest.mark.parametrize("size,threads", [(256, 32), (253, 64), (31, 128)])
def test_fragment_fill_parallel_layout_lowering_runs_on_gpu(size, threads, compile_tilelang):
    target = cuda_target()
    source = TensorView.create("source", [size], "float32")
    output = TensorView.create("output", [size], "float32")
    tile_size = max(threads, 1 << (size - 1).bit_length())
    fragment = TensorView.create("fragment", [tile_size], "float32", scope="local.fragment")
    index = ir.Var("index", "int32")
    thread = ir.Var("thread", "int32")
    loop = ir.For(
        index,
        0,
        tile_size,
        1,
        ir.ForKind.PARALLEL,
        ir.If(index < size, output.store(index, source.load(index) + fragment.load(index)), None),
    )
    body = ir.SeqStmt([fragment.allocate(), tile("fill", fragment.tensor, ir.const(3.0)), loop])
    body = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root"}, body)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(threads)], [thread], None, body)
    lowered = lower(module(body, [source, output], target), target)
    assert not any(loop.kind == ir.ForKind.PARALLEL for loop in nodes(lowered, ir.For))
    executable = compile_tilelang(lowered, target=target)
    expected = np.arange(size, dtype="float32")
    actual = runtime.empty((size,), "float32", device=runtime.cuda(0))
    executable(runtime.tensor(expected, device=runtime.cuda(0)), actual)
    np.testing.assert_array_equal(actual.numpy(), expected + 3)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("predicated", [False, True])
def test_async_copy_injection_preserves_zero_fill_and_visibility(dtype, predicated, compile_tilelang):
    target = cuda_target()
    source = TensorView.create("source", [128], dtype)
    output = TensorView.create("output", [128], dtype)
    shared = TensorView.create("shared", [128], dtype, scope="shared", data_alignment=16)
    thread = ir.Var("thread", "int32")
    lane = ir.Var("lane", "int32")
    index = thread * 4 + lane
    value = source.load(index)
    if predicated:
        value = prim.if_then_else(thread < 27, value, ir.const(0, dtype))
    copy = ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, shared.store(index, value))
    read_lane = ir.Var("read_lane", "int32")
    read_index = thread * 4 + read_lane
    read = ir.For(read_lane, 0, 4, 1, ir.ForKind.DEFAULT, output.store(read_index, shared.load(read_index)))
    body = ir.SeqStmt([shared.allocate(), copy, ir.Evaluate(tirx.op.gpu_storage_sync("shared")), read])
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [thread], None, body)
    with target:
        lowered = transform("cuda.transform.InjectPTXAsyncCopy", module(body, [source, output], target), False)
        assert any(call.op.name == "tl.ptx_cp_async" for call in nodes(lowered, ir.Call))
        lowered = transform("transform.VectorizeLoop", lowered, True)
        lowered = transform("transform.LowerAccessPtr", lowered)
    assert tirx.analysis.check_well_formed(lowered)
    executable = compile_tilelang(lowered, target=target)
    values = np.arange(128, dtype=dtype)
    actual = runtime.empty((128,), dtype, device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    if predicated:
        values[108:] = 0
    np.testing.assert_array_equal(actual.numpy(), values)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("fuse_barriers", [False, True])
def test_tma_copy_lowering_allocates_and_reuses_native_barriers(dtype, sliced, repeat, fuse_barriers, compile_tilelang):
    target = cuda_target()
    if int(runtime.cuda(0).compute_version.split(".")[0]) < 9:
        pytest.skip("TMA requires compute capability 9.0 or newer")
    source_shape = (12, 64) if sliced else (8, 32)
    output_shape = (16, 64) if sliced else (8, 32)
    source = TensorView.create("source", source_shape, dtype)
    output = TensorView.create("output", output_shape, dtype)
    shared = TensorView.create("shared", [8, 32], dtype, scope="shared", data_alignment=128)
    source_region = source.region([ir.Range(2, 10), ir.Range(16, 48)]) if sliced else source.tensor
    output_region = output.region([ir.Range(4, 12), ir.Range(16, 48)]) if sliced else output.tensor
    index = ir.Var("index", "int32")
    thread = ir.Var("thread", "int32")
    epoch = ir.Var("epoch", "int32")
    addend = prim.Cast(dtype, epoch) if repeat else ir.const(3, dtype)
    update = ir.For(
        index, 0, 256, 1, ir.ForKind.PARALLEL, shared.store([index // 32, index % 32], shared.load([index // 32, index % 32]) + addend)
    )
    body = ir.SeqStmt(
        [
            tile("copy", source_region, shared.tensor, annotations={"prefer_instruction": "tma"}),
            update,
            ir.Evaluate(tirx.op.gpu_storage_sync("shared")),
            ir.Evaluate(ir.Call("tl.fence_proxy_async", [])),
            tile("copy", shared.tensor, output_region, annotations={"prefer_instruction": "tma"}),
        ]
    )
    if repeat:
        body = ir.For(epoch, 3, 11, 2, ir.ForKind.DEFAULT, body)
    body = ir.SeqStmt([shared.allocate(), body])
    body = ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root"}, body)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    lowered = module(body, [source, output], target)
    with target:
        lowered = transform("transform.LayoutInference", lowered)
        lowered = transform("transform.LowerTileOp", lowered)
        assert lowered["main"].attrs["tl.has_tma"]
        assert tirx.analysis.check_well_formed(lowered)
        if fuse_barriers:
            lowered = transform("cuda.transform.FuseMBarrierArriveExpectTx", lowered)
            assert any(call.op.name == "tl.mbarrier_arrive_expect_tx" for call in nodes(lowered, ir.Call))
        lowered = transform("cuda.transform.LowerSharedBarrier", lowered)
        lowered = transform("cuda.transform.LowerHopperIntrin", lowered)
        lowered = transform("transform.LowerAccessPtr", lowered)
    assert tirx.analysis.check_well_formed(lowered)
    assert any(call.op.name == "tl.mbarrier_init" for call in nodes(lowered, ir.Call))
    if sliced:
        assert any(call.op.name == "tirx.cuda.tensormap_encode_tiled" for call in nodes(lowered, ir.Call))
    executable = compile_tilelang(lowered, target=target)
    values = (np.arange(np.prod(source_shape)).reshape(source_shape) % 113).astype(dtype)
    actual = runtime.tensor(np.zeros(output_shape, dtype=dtype), device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    expected = np.zeros(output_shape, dtype=dtype)
    addend = 9 if repeat else 3
    if sliced:
        expected[4:12, 16:48] = values[2:10, 16:48] + addend
    else:
        expected[:] = values + addend
    np.testing.assert_array_equal(actual.numpy(), expected)
