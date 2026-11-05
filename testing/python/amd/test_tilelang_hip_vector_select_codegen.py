import tvm_ffi
import pytest
import tilelang
import tilelang.testing

import tvm_tirx
from tvm_tirx import ir, tirx


def _make_device_function(parameters, body):
    # The final HIP emitter consumes pointer parameters. Keep the tensor
    # access contracts in explicit decl_tensor bindings inside the function.
    params, declarations = [], []
    for value in parameters:
        if isinstance(value.ty, tirx.TensorType):
            pointer = ir.Var(value.name, ir.PtrType(value.ty.dtype, value.ty.scope))
            params.append(pointer)
            declarations.append(ir.Bind(value, ir.Call("tirx.decl_tensor", [pointer], ty_args=[value.ty])))
        else:
            params.append(value)
    return tirx.Function(params, [*declarations, body], ret_type=ir.PrimType("void"))


def _build_vector_select_source(
    lanes,
    wrap_input_in_let=False,
    wrap_input_in_scalar_lets=False,
    wrap_input_in_call=False,
    wrap_input_in_call_extern=False,
    wrap_input_in_reinterpret=False,
    wrap_input_in_abs=False,
    wrap_input_in_packed_call=False,
):
    input_dtype = "bfloat16" if wrap_input_in_abs else "float32"
    output_dtype = "int32" if wrap_input_in_reinterpret else input_dtype
    mask_buffer = ir.Var("mask", tirx.TensorType((lanes,), "float32", layout="default"))
    input_buffer = ir.Var("input", tirx.TensorType((lanes,), input_dtype, layout="default"))
    output_buffer = ir.Var("output", tirx.TensorType((lanes,), output_dtype, layout="default"))
    ramp = ir.prim.Ramp(0, 1, lanes)
    mask_vector = ir.TensorLoad(mask_buffer, [ramp])
    input_vector = ir.TensorLoad(input_buffer, [ramp])
    if wrap_input_in_let:
        value = tirx.Var("value", f"float32x{lanes}")
        input_vector = tirx.Let(value, input_vector, value)
    if wrap_input_in_scalar_lets:
        scalar_lets = []
        for lane in range(lanes):
            value = tirx.Var(f"value_{lane}", "float32")
            load = ir.TensorLoad(input_buffer, [lane])
            scalar_lets.append(tirx.Let(value, load, value))
        input_vector = ir.prim.Shuffle(scalar_lets, list(range(lanes)))
    if wrap_input_in_call:
        input_vector = tirx.call_pure_extern("exp2f", input_vector, ty=f"float32x{lanes}")
    if wrap_input_in_call_extern:
        input_vector = tirx.call_extern("exp2f", input_vector, ty=f"float32x{lanes}")
    if wrap_input_in_reinterpret:
        input_vector = tirx.reinterpret(input_vector, ty=f"int32x{lanes}")
    if wrap_input_in_abs:
        input_vector = ir.Call(tvm_tirx.ir.Op.get("tl.abs2"), [input_vector], ty=f"bfloat16x{lanes}")
    if wrap_input_in_packed_call:
        bias = ir.prim.Broadcast(tirx.const(1, "float32"), lanes)
        input_vector = ir.Call(tvm_tirx.ir.Op.get("tl.add2"), [input_vector, bias], ty=f"float32x{lanes}")
    mask_zero = ir.prim.Broadcast(tirx.const(0, "float32"), lanes)
    result_zero = ir.prim.Broadcast(tirx.const(0, output_dtype), lanes)
    selected = tirx.Select(mask_vector > mask_zero, input_vector, result_zero)
    body = ir.TensorStore(output_buffer, [ramp], selected)
    func = _make_device_function([mask_buffer, input_buffer, output_buffer], body)
    func = func.with_attr("global_symbol", "vector_select")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm_tirx.IRModule({"vector_select": func})
    build = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm_tirx.target.Target("rocm")).inspect_source()


def _build_vector_select_with_vector_argument_source():
    lanes = 4
    value = tirx.Var("value", "float32x4")
    mask_buffer = ir.Var("mask", tirx.TensorType((lanes,), "float32", layout="default"))
    output_buffer = ir.Var("output", tirx.TensorType((lanes,), "float32", layout="default"))
    scalar_base = ir.prim.Shuffle([ir.prim.Ramp(0, 1, lanes)], [0])
    ramp = ir.prim.Ramp(scalar_base, 1, lanes)
    mask_vector = ir.TensorLoad(mask_buffer, [ramp])
    zero = ir.prim.Broadcast(tirx.const(0, "float32"), lanes)
    selected = tirx.Select(mask_vector > zero, value, zero)
    body = ir.TensorStore(output_buffer, [ramp], selected)
    func = _make_device_function([value, mask_buffer, output_buffer], body)
    func = func.with_attr("global_symbol", "vector_select_argument")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm_tirx.IRModule({"vector_select_argument": func})
    build = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm_tirx.target.Target("rocm")).inspect_source()


def _make_lane_changing_reinterpret_select_module(source_dtype, source_lanes, result_dtype, result_lanes):
    mask_buffer = ir.Var("mask", tirx.TensorType((result_lanes,), "float32", layout="default"))
    input_buffer = ir.Var("input", tirx.TensorType((source_lanes,), source_dtype, layout="default"))
    output_buffer = ir.Var("output", tirx.TensorType((result_lanes,), result_dtype, layout="default"))
    mask_ramp = ir.prim.Ramp(0, 1, result_lanes)
    input_ramp = ir.prim.Ramp(0, 1, source_lanes)
    mask_vector = ir.TensorLoad(mask_buffer, [mask_ramp])
    input_vector = ir.TensorLoad(input_buffer, [input_ramp])
    if source_lanes < result_lanes:
        input_vector = ir.prim.Shuffle([input_vector], list(range(source_lanes)))
    reinterpreted = tirx.reinterpret(input_vector, ty=f"{result_dtype}x{result_lanes}")
    mask_zero = ir.prim.Broadcast(tirx.const(0, "float32"), result_lanes)
    result_zero = ir.prim.Broadcast(tirx.const(0, result_dtype), result_lanes)
    selected = tirx.Select(mask_vector > mask_zero, reinterpreted, result_zero)
    body = ir.TensorStore(output_buffer, [mask_ramp], selected)
    func = _make_device_function([mask_buffer, input_buffer, output_buffer], body)
    func = func.with_attr("global_symbol", "lane_changing_reinterpret_select")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm_tirx.IRModule({"lane_changing_reinterpret_select": func})


def _build_lane_changing_reinterpret_select_source(source_dtype, source_lanes, result_dtype, result_lanes):
    mod = _make_lane_changing_reinterpret_select_module(source_dtype, source_lanes, result_dtype, result_lanes)
    build = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm_tirx.target.Target("rocm")).inspect_source()


def _make_narrow_vector_select_module(dtype, lanes):
    on_true = tirx.Var("on_true", f"{dtype}x{lanes}")
    on_false = tirx.Var("on_false", f"{dtype}x{lanes}")
    ramp = ir.prim.Ramp(0, 1, lanes)
    limit = ir.prim.Broadcast(tirx.const(lanes // 2, "int32"), lanes)
    selected = tirx.Select(
        ramp < limit,
        on_true,
        on_false,
    )
    func = tirx.Function([on_true, on_false], ir.Evaluate(selected), ret_type=ir.PrimType("void"))
    func = func.with_attr("global_symbol", "narrow_vector_select")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm_tirx.IRModule({"narrow_vector_select": func})


def _build_narrow_vector_select_source(dtype, lanes):
    mod = _make_narrow_vector_select_module(dtype, lanes)
    build = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm_tirx.target.Target("rocm")).inspect_source()


def _build_scalar_select_source():
    mask_buffer = ir.Var("mask", tirx.TensorType((1,), "float32", layout="default"))
    input_buffer = ir.Var("input", tirx.TensorType((1,), "float32", layout="default"))
    output_buffer = ir.Var("output", tirx.TensorType((1,), "float32", layout="default"))
    zero = tirx.const(0, "float32")
    selected = tirx.Select(
        ir.TensorLoad(mask_buffer, [0]) > zero,
        ir.TensorLoad(input_buffer, [0]),
        zero,
    )
    body = ir.TensorStore(output_buffer, [0], selected)
    func = _make_device_function([mask_buffer, input_buffer, output_buffer], body)
    func = func.with_attr("global_symbol", "scalar_select")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm_tirx.IRModule({"scalar_select": func})
    build = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm_tirx.target.Target("rocm")).inspect_source()


def test_vector_condition_select_is_scalarized_in_hip_source():
    source = _build_vector_select_source(4)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("mask[" in line and "? input[" in line for line in assignments)
    assert "*(float4*)(input" not in source


def test_scalar_select_uses_base_ternary_path():
    source = _build_scalar_select_source()

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert assignments == ["  output[0] = ((mask[0] > 0.000000e+00f) ? input[0] : 0.000000e+00f);"]


def test_eight_lane_select_keeps_masked_loads_in_ternary_branches():
    source = _build_vector_select_source(8)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 8
    assert all("mask[" in line and "? input[" in line for line in assignments)
    assert "*(ulonglong4*)(input" not in source
    assert "ushort8" not in source


def test_vector_argument_and_scalar_shuffle_are_lane_extracted():
    source = _build_vector_select_with_vector_argument_source()

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("mask[" in line and "? (value)." in line for line in assignments)


def test_vector_let_binding_is_scalarized_without_type_mismatch():
    source = _build_vector_select_source(4, wrap_input_in_let=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("mask[" in line and "? input[" in line for line in assignments)


def test_scalar_let_loads_stay_inside_vector_select_branches():
    source = _build_vector_select_source(4, wrap_input_in_scalar_lets=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("mask[" in line and "? input[" in line for line in assignments)


def test_vector_pure_call_keeps_argument_load_in_select_branch():
    source = _build_vector_select_source(4, wrap_input_in_call=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("? exp2f(input[" in line for line in assignments)
    assert "*(float4*)(input" not in source


def test_vector_call_extern_keeps_argument_load_in_select_branch():
    source = _build_vector_select_source(4, wrap_input_in_call_extern=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("? exp2f(input[" in line for line in assignments)
    assert "*(float4*)(input" not in source


def test_vector_reinterpret_keeps_argument_load_in_select_branch():
    source = _build_vector_select_source(4, wrap_input_in_reinterpret=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    for lane, assignment in enumerate(assignments):
        assert f"input[{lane}]" in assignment
    assert "*(float4*)(input" not in source


@pytest.mark.parametrize(
    ("source_dtype", "source_lanes", "result_dtype", "result_lanes"),
    [("uint8", 8, "uint16", 4), ("uint16", 4, "uint8", 8)],
)
def test_lane_changing_reinterpret_stays_inside_vector_select_branch(source_dtype, source_lanes, result_dtype, result_lanes):
    source = _build_lane_changing_reinterpret_select_source(source_dtype, source_lanes, result_dtype, result_lanes)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == result_lanes
    for lane, assignment in enumerate(assignments):
        if source_lanes > result_lanes:
            lanes_per_result = source_lanes // result_lanes
            expected_indices = set(range(lane * lanes_per_result, (lane + 1) * lanes_per_result))
        else:
            expected_indices = {lane // (result_lanes // source_lanes)}
        for input_lane in range(source_lanes):
            assert (f"input[{input_lane}]" in assignment) == (input_lane in expected_indices)
    assert "*)(input +" not in source


def test_bfloat16_abs_keeps_packed_sign_bit_semantics():
    source = _build_vector_select_source(2, wrap_input_in_abs=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 2
    assert all("? __habs(input[" in line for line in assignments)
    assert all(">= (bfloat16_t)0" not in line for line in assignments)


def test_vector_packed_call_keeps_argument_load_in_select_branch():
    source = _build_vector_select_source(2, wrap_input_in_packed_call=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 2
    assert all("+ 1.000000e+00f" in line for line in assignments)
    for lane, assignment in enumerate(assignments):
        assert f"? (input[{lane}]" in assignment
    assert "tl::add2" not in source


@pytest.mark.parametrize(
    ("dtype", "lanes", "carrier_type"),
    [("float4_e2m1fn", 8, "fp4_e2_8_t"), ("int8", 8, "int2")],
)
def test_packed_vector_select_result_is_value_initialized(dtype, lanes, carrier_type):
    source = _build_narrow_vector_select_source(dtype, lanes)

    declarations = [line for line in source.splitlines() if line.startswith(f"  {carrier_type} __")]
    assert len(declarations) == 1
    assert declarations[0].endswith("{};")


@pytest.mark.parametrize("lanes", [8, 16, 32])
def test_wide_float4_vector_results_are_stored_by_packed_pair(lanes):
    source = _build_narrow_vector_select_source("float4_e2m1fn", lanes)

    assignments = [line for line in source.splitlines() if "?" in line and "set_" in line]
    assert len(assignments) == lanes
    for lane, assignment in enumerate(assignments):
        remaining_lanes = lanes
        remaining_lane = lane
        path = []
        while remaining_lanes > 2:
            group_size = remaining_lanes // 2
            path.append("x" if remaining_lane < group_size else "y")
            remaining_lane %= group_size
            remaining_lanes = group_size
        pair = ".".join(path)
        accessor = "x" if remaining_lane == 0 else "y"
        assert f".{pair}.set_{accessor}(" in assignment
        assert f"(({lane} < {lanes // 2}) ? (on_true).{pair}.{accessor}() : (on_false).{pair}.{accessor}())" in assignment


@pytest.mark.parametrize(
    ("dtype", "lanes", "lane_members"),
    [
        ("float8_e4m3fn", 2, ["x", "y"]),
        ("float8_e5m2", 2, ["x", "y"]),
        (
            "float8_e5m2",
            8,
            ["x.x", "x.y", "x.z", "x.w", "y.x", "y.y", "y.z", "y.w"],
        ),
    ],
)
def test_float8_vector_results_are_stored_by_lane(dtype, lanes, lane_members):
    source = _build_narrow_vector_select_source(dtype, lanes)

    if lanes == 2:
        assignments = [line for line in source.splitlines() if "tl_fp8x2_set_lane" in line]
    else:
        assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == lanes
    for lane, member in enumerate(lane_members):
        if lanes == 2:
            scalar_type = "fp8_e4_t" if "e4" in dtype else "fp8_e5_t"
            on_true = f"tl_fp8x2_get_lane<{scalar_type}>((on_true), {lane})"
            on_false = f"tl_fp8x2_get_lane<{scalar_type}>((on_false), {lane})"
            assert f"(({lane} < 1) ? {on_true} : {on_false})" in assignments[lane]
            assert assignments[lane].startswith("  tl_fp8x2_set_lane(")
            assert f", {lane}, " in assignments[lane]
        else:
            assert f"(({lane} < {lanes // 2}) ? (on_true).{member} : (on_false).{member})" in assignments[lane]


@pytest.mark.parametrize(("dtype", "lanes"), [("int4", 4), ("int4", 8), ("uint4", 4), ("uint4", 8)])
def test_integer_4bit_vector_results_use_packed_nibble_access(dtype, lanes):
    source = _build_narrow_vector_select_source(dtype, lanes)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == lanes
    assert all("((const unsigned char*)(&((on_true))))" in line for line in assignments)
    assert all("((const unsigned char*)(&((on_false))))" in line for line in assignments)
    assert all(line.startswith("  ((unsigned char*)(&(__") for line in assignments)
    assert ".x" not in "\n".join(assignments)


def test_signed_int8_vector_results_are_packed_without_signed_shifts():
    source = _build_narrow_vector_select_source("int8", 8)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 8
    assert all("static_cast<unsigned int>(static_cast<unsigned char>(" in line for line in assignments)


@pytest.mark.parametrize(
    ("lanes", "carrier_type"),
    [(8, "uint8_t"), (16, "uint16_t"), (32, "uint")],
)
def test_wide_bool_vector_results_use_packed_bit_access(lanes, carrier_type):
    source = _build_narrow_vector_select_source("bool", lanes)

    declarations = [line for line in source.splitlines() if line.startswith(f"  {carrier_type} __")]
    assert len(declarations) == 1
    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == lanes
    assert all("((const unsigned char*)(&((on_true))))" in line for line in assignments)
    assert all("((const unsigned char*)(&((on_false))))" in line for line in assignments)
    assert all(line.startswith("  ((unsigned char*)(&(__") for line in assignments)
    assert all("0x01u" in line for line in assignments)
    assert ".x" not in "\n".join(assignments)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("dtype", "lanes"),
    [
        ("float8_e4m3fn", 2),
        ("float8_e5m2", 2),
        ("int4", 4),
        ("int4", 8),
        ("uint4", 4),
        ("uint4", 8),
        ("int8", 8),
        ("bool", 8),
        ("bool", 16),
        ("bool", 32),
    ],
)
def test_narrow_vector_select_compiles_with_packed_hip_carriers(dtype, lanes):
    mod = _make_narrow_vector_select_module(dtype, lanes)
    build = tvm_ffi.get_global_func("target.build.tilelang_hip")
    build(mod, tvm_tirx.target.Target("rocm"))


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("source_dtype", "source_lanes", "result_dtype", "result_lanes"),
    [("uint8", 8, "uint16", 4), ("uint16", 4, "uint8", 8)],
)
def test_lane_changing_reinterpret_select_compiles(source_dtype, source_lanes, result_dtype, result_lanes):
    mod = _make_lane_changing_reinterpret_select_module(source_dtype, source_lanes, result_dtype, result_lanes)
    build = tvm_ffi.get_global_func("target.build.tilelang_hip")
    build(mod, tvm_tirx.target.Target("rocm"))


if __name__ == "__main__":
    test_vector_condition_select_is_scalarized_in_hip_source()
