import re

import pytest
import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.backend.metal import op as metal
from tvm_tirx.ir import prim
from tvm_tirx.target import Target


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _view(tensor, pointer):
    return ir.Bind(tensor, ir.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty]))


def _data(tensor):
    return ir.Call("tirx.tensor_data_ptr", [tensor])


def _launch(tag, extent, variable, body):
    return ir.RegionStmt("tirx.launch_thread", [ir.StringImm(tag), ir.const(extent)], [variable], None, body)


def _function(name, parameters, body, launch_params=()):
    return (
        tirx.Function(parameters, body)
        .with_attr("global_symbol", name)
        .with_attr("calling_conv", 2)
        .with_attr("tirx.kernel_launch_params", list(launch_params))
        .with_attr("target", Target("metal"))
    )


def _source(functions):
    module = ir.IRModule({function.attrs["global_symbol"]: function for function in functions}).with_attr("codegen.source_only", True)
    module = tirx.transform.LowerIntrin()(module)
    return tvm_ffi.get_global_func("tl.codegen.build_metal")(module, Target("metal")).inspect_source()


def _cooperative_function(name, tile_count, input_dtype):
    left = ir.Var("left", tirx.TensorType([256], input_dtype, scope="metal.cooperative_tensor"))
    right = ir.Var("right", tirx.TensorType([512], input_dtype, scope="metal.cooperative_tensor"))
    accumulator = ir.Var(
        "accumulator",
        tirx.TensorType([512 * tile_count], "float32", scope="metal.cooperative_tensor"),
    )
    destination = ir.Var(
        "destination",
        tirx.TensorType([512 * tile_count], "float32", scope="metal.cooperative_tensor"),
    )
    output = ir.Var("output", ir.PtrType("float16"))
    tile = ir.Var("tile", "int32")
    output_tile = ir.Call("tirx.ptr_byte_offset", [output, tile * 1024], ty=output.ty)
    loop = ir.For(
        tile,
        ir.const(0),
        ir.const(tile_count),
        ir.const(1),
        ir.ForKind.DEFAULT,
        [
            ir.Evaluate(metal.cooperative_tensor_fill(_data(accumulator), tile, ir.const(3, "float32"), 16, 32)),
            ir.Evaluate(
                metal.cooperative_tensor_multiply_accumulate(
                    _data(destination),
                    tile,
                    _data(left),
                    0,
                    _data(right),
                    0,
                    _data(accumulator),
                    tile,
                    16,
                    32,
                    16,
                )
            ),
            ir.Evaluate(metal.cooperative_tensor_store(_data(destination), tile, output_tile, 32, 16, 32)),
        ],
    )
    body = [
        *[_allocate(tensor) for tensor in [left, right, accumulator, destination]],
        ir.Evaluate(metal.cooperative_tensor_fill(_data(left), 0, ir.const(1, input_dtype), 16, 16)),
        ir.Evaluate(metal.cooperative_tensor_fill(_data(right), 0, ir.const(2, input_dtype), 16, 32)),
        loop,
    ]
    return _function(name, [output], _launch("threadIdx.x", 32, ir.Var("thread", "int32"), body), ["threadIdx.x"])


@pytest.mark.parametrize("tile_count", [1, 2, 4])
@pytest.mark.parametrize("input_dtype,metal_dtype", [("float16", "half"), ("bfloat16", "bfloat"), ("float32", "float")])
def test_cooperative_tensors_keep_distinct_accumulator_and_destination(tile_count, input_dtype, metal_dtype):
    source = _source([_cooperative_function("matrix_kernel", tile_count, input_dtype)])
    assert f"thread {metal_dtype} left[8]" in source
    assert f"thread {metal_dtype} right[16]" in source
    assert "thread float accumulator[" not in source
    assert "thread float destination[" not in source
    assert "for (int tile" not in source
    assert "simdgroup_index_in_threadgroup" in source
    assert "max_total_threads_per_threadgroup(32)" in source
    for tile in range(tile_count):
        assert f"auto accumulator_tile_{tile} =" in source
        assert f"auto destination_tile_{tile} =" in source
        assert f"destination_tile_{tile}[__element] = ((thread const float*)(&accumulator_tile_{tile}[0]))[__element]" in source
        assert f".run(__ct_a, __ct_b, destination_tile_{tile})" in source
        assert f"half4(*(thread float4*)(&destination_tile_{tile}[0]))" in source


def test_cooperative_tensor_state_is_function_local():
    source = _source(
        [
            _cooperative_function("first_kernel", 1, "float16"),
            _cooperative_function("second_kernel", 4, "float32"),
        ]
    )
    assert source.count("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>") == 2
    assert source.count("auto destination_tile_0 =") == 2
    assert source.count("auto destination_tile_3 =") == 1
    assert "get_left_input_cooperative_tensor<half, half, float>" in source
    assert "get_left_input_cooperative_tensor<float, float, float>" in source


@pytest.mark.parametrize("panel", [2, 4, 8])
@pytest.mark.parametrize("style", ["bits", "division"])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_mlx_coordinates_reuse_logical_grid_without_double_swizzling(panel, style, dtype):
    block_x = ir.Var("block_x", "int32")
    block_y = ir.Var("block_y", "int32")
    physical_x = prim.Cast(dtype, block_x)
    physical_y = prim.Cast(dtype, block_y)
    panel_value = ir.const(panel, dtype)
    shift = ir.const(panel.bit_length() - 1, dtype)
    if style == "bits":
        logical_x = prim.RShift(physical_x, shift)
        remainder = prim.BitwiseAnd(physical_x, ir.const(panel - 1, dtype))
    else:
        logical_x = prim.FloorDiv(physical_x, panel_value)
        remainder = prim.FloorMod(physical_x, panel_value)
    output_pointer = ir.Var("output_pointer", ir.PtrType(dtype))
    output = ir.Var("output", tirx.TensorType([4], dtype))
    statements = [
        ir.TensorStore(output, [ir.const(0)], logical_x),
        ir.TensorStore(output, [ir.const(1)], physical_y * (panel * 8192) + remainder * 8192),
        ir.TensorStore(output, [ir.const(2)], remainder),
        ir.TensorStore(output, [ir.const(3)], physical_y * (panel * 8192) + remainder * 4096),
    ]
    swizzle = ir.RegionStmt(
        "tl.scope",
        [],
        [],
        {"metal.threadblock_swizzle_pattern": ir.Tuple([ir.StringImm("rasterization2DMLX"), ir.const(panel)])},
        statements,
    )
    grid = _launch("blockIdx.y", 2, block_y, _launch("blockIdx.x", 16, block_x, swizzle))
    function = _function(
        "swizzle_kernel",
        [output_pointer],
        [_view(output, output_pointer), grid],
        ["blockIdx.x", "blockIdx.y"],
    )
    source = _source([function])
    stores = [line for line in source.splitlines() if re.search(r"output_pointer\[[0-3]\] =", line)]
    assert len(stores) == 4
    assert "blockIdx.x" in stores[0] and "__physical_blockIdx" not in stores[0]
    assert "blockIdx.y" in stores[1] and "__physical_blockIdx" not in stores[1]
    assert "8192" in stores[1] and str(panel * 8192) not in stores[1]
    assert "__physical_blockIdx.x" in stores[2]
    assert "__physical_blockIdx.y" in stores[3]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("lanes", [2, 3, 4, 6, 8])
def test_vector_load_math_select_and_store_use_native_storage(dtype, lanes):
    source_pointer = ir.Var("source_pointer", ir.PtrType("float32"))
    output_pointer = ir.Var("output_pointer", ir.PtrType("float32"))
    source = ir.Var("source", tirx.TensorType([lanes], "float32"))
    output = ir.Var("output", tirx.TensorType([lanes], "float32"))
    indices = [prim.Ramp(0, 1, lanes)]
    vector = prim.Cast(f"{dtype}x{lanes}", ir.TensorLoad(source, indices))
    one = prim.Broadcast(ir.const(1, dtype), lanes)
    computed = vector + one
    result = prim.Select(computed > one, computed, one)
    body = [
        _view(source, source_pointer),
        _view(output, output_pointer),
        ir.TensorStore(output, indices, prim.Cast(f"float32x{lanes}", result)),
    ]
    function = _function("vector_kernel", [source_pointer, output_pointer], body)
    text = _source([function])
    if lanes > 4:
        assert f"array<float, {lanes}>" in text
        if dtype in ("float16", "bfloat16"):
            assert ("packed_uint3" if lanes == 6 else "uint4") in text
    if dtype == "bfloat16":
        assert "bfloat" in text
    assert re.search(r"output_pointer(?:\[| \+)", text)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_threadgroup_barriers_cover_both_shared_storage_scopes(scope):
    function = _function("barrier_kernel", [], ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm(scope)])))
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in _source([function])
