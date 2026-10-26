import numpy as np
import pytest
import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


def _allocation(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _view(tensor, pointer):
    return ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _data(tensor):
    return ir.Call("tirx.tensor_data_ptr", [tensor])


def _offset(pointer, offset, dtype):
    return ir.Call("tirx.ptr_byte_offset", [pointer, ir.const(offset)], ty=ir.PtrType(dtype, "shared.dyn"))


def _build(dtype, lanes=1, vector_element=False, reinterpret=False):
    count = 32 * lanes
    scalar_type = ir.PrimType(dtype)
    scalar_bytes = (scalar_type.dtype.bits + 7) // 8
    output_dtype = "uint32" if reinterpret else "float32"
    source_pointer = ir.Var("source_pointer", ir.PtrType("float32"))
    output_pointer = ir.Var("output_pointer", ir.PtrType(output_dtype))
    source = ir.Var("source", tirx.TensorType([count], "float32"))
    output = ir.Var("output", tirx.TensorType([count], output_dtype))
    arena = ir.Var("arena", tirx.TensorType([count * scalar_bytes + 32], "uint8", scope="shared.dyn"))
    element_dtype = f"{dtype}x{lanes}" if vector_element else dtype
    storage = ir.Var(
        "storage",
        tirx.TensorType([32 if vector_element else count], element_dtype, scope="shared.dyn"),
    )
    pointer = _offset(_offset(_data(arena), 16, "uint8"), 16, element_dtype)
    thread = ir.Var("thread", "int32")
    indices = [thread if lanes == 1 else prim.Ramp(thread * lanes, 1, lanes)]
    storage_indices = [thread] if vector_element else indices
    reversed_thread = 31 - thread
    read_indices = [reversed_thread if lanes == 1 else prim.Ramp(reversed_thread * lanes, 1, lanes)]
    storage_read_indices = [reversed_thread] if vector_element else read_indices
    value_dtype = dtype if lanes == 1 else f"{dtype}x{lanes}"
    loaded = ir.TensorLoad(source, indices)
    if dtype == "bool":
        zero = ir.const(0, "float32")
        if lanes > 1:
            zero = prim.Broadcast(zero, lanes)
        value = loaded != zero
    else:
        value = prim.Cast(value_dtype, loaded)
    statements = [
        _allocation(arena),
        _view(storage, pointer),
        ir.TensorStore(storage, storage_indices, value),
    ]
    statements.append(ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared.dyn")])))
    if reinterpret:
        raw = ir.Var("raw", tirx.TensorType([count], "uint32", scope="shared.dyn"))
        statements.append(_view(raw, _offset(_data(arena), 32, "uint32")))
        result = ir.TensorLoad(raw, read_indices)
    else:
        result = ir.TensorLoad(storage, storage_read_indices)
        if dtype == "bool":
            one = ir.const(1, "float32")
            if lanes > 1:
                one = prim.Broadcast(one, lanes)
            result = prim.Select(result, one, zero)
        else:
            result = prim.Cast("float32" if lanes == 1 else f"float32x{lanes}", result)
    statements.append(ir.TensorStore(output, indices, result))
    launch = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [thread],
        None,
        ir.SeqStmt(statements),
    )
    function = tirx.Function(
        [source_pointer, output_pointer],
        ir.SeqStmt([_view(source, source_pointer), _view(output, output_pointer), launch]),
    )
    function = function.with_attr("global_symbol", "arena_kernel").with_attr("calling_conv", 2)
    module = tvm_ffi.get_global_func("tl.codegen.build_webgpu")(ir.IRModule({"arena_kernel": function}), Target("webgpu"))
    return module.inspect_source("wgsl")


@pytest.fixture(scope="module")
def webgpu_device():
    wgpu = pytest.importorskip("wgpu")
    adapters = wgpu.gpu.enumerate_adapters_sync()
    adapters = [adapter for adapter in adapters if adapter.info["backend_type"] == "Vulkan" and "shader-f16" in adapter.features]
    if not adapters:
        pytest.skip("A Vulkan WebGPU adapter with shader-f16 is unavailable")
    discrete = [adapter for adapter in adapters if adapter.info["adapter_type"] == "DiscreteGPU"]
    adapter = discrete[min(5, len(discrete) - 1)] if discrete else adapters[0]
    return adapter.request_device_sync(required_features=["shader-f16"])


def _run(device, source, values, dtype="float32"):
    import wgpu

    shader = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": shader, "entry_point": "arena_kernel"})
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    input_buffer = device.create_buffer_with_data(data=values, usage=usage)
    output_buffer = device.create_buffer(size=values.size * 4, usage=usage)
    uniform = device.create_buffer_with_data(data=np.array([1, 0, 0, 0], dtype="uint32"), usage=wgpu.BufferUsage.UNIFORM)
    bindings = [
        {"binding": binding, "resource": {"buffer": buffer}} for binding, buffer in enumerate([input_buffer, output_buffer, uniform])
    ]
    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=bindings)
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, group)
    compute.dispatch_workgroups(1)
    compute.end()
    device.queue.submit([encoder.finish()])
    return np.frombuffer(device.queue.read_buffer(output_buffer), dtype=dtype).copy()


@pytest.mark.parametrize("dtype", ["float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32", "bool"])
@pytest.mark.parametrize("lanes,vector_element", [(1, False), (2, False), (4, False), (2, True), (4, True)])
def test_byte_arena_source_uses_atomic_subword_access(dtype, lanes, vector_element):
    source = _build(dtype, lanes, vector_element)
    assert "array<atomic<u32>" in source
    assert "atomicLoad(&arena[" in source
    assert "workgroupBarrier()" in source
    if ir.PrimType(dtype).dtype.bits < 32:
        assert "atomicAnd(&arena[" in source
        assert "atomicOr(&arena[" in source
    else:
        assert "atomicStore(&arena[" in source


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32", "bool"])
@pytest.mark.parametrize("lanes,vector_element", [(1, False), (2, False), (4, False), (2, True), (4, True)])
def test_byte_arena_preserves_every_thread_and_lane(webgpu_device, dtype, lanes, vector_element):
    source = _build(dtype, lanes, vector_element)
    values = np.arange(32 * lanes, dtype="float32") % 15
    if not dtype.startswith("uint"):
        values -= 7
    if dtype.startswith("float"):
        values *= 0.25
    expected = values.astype(dtype).astype("float32").reshape(32, lanes)[::-1].reshape(-1)
    np.testing.assert_array_equal(_run(webgpu_device, source, values), expected)


@pytest.mark.gpu
def test_byte_arena_reinterpreted_views_share_the_same_bytes(webgpu_device):
    source = _build("float32", reinterpret=True)
    values = np.linspace(-4, 4, 32, dtype="float32")
    expected = values.view("uint32")[::-1]
    np.testing.assert_array_equal(_run(webgpu_device, source, values, "uint32"), expected)
