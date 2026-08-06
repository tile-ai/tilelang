import pytest

from tilelang import tvm
from tvm import tirx


def _build_vector_select_source(
    lanes,
    wrap_input_in_let=False,
    wrap_input_in_scalar_lets=False,
    wrap_input_in_call=False,
    wrap_input_in_call_extern=False,
    wrap_input_in_reinterpret=False,
    wrap_input_in_packed_call=False,
):
    output_dtype = "int32" if wrap_input_in_reinterpret else "float32"
    mask_buffer = tirx.decl_buffer((lanes,), "float32", name="mask")
    input_buffer = tirx.decl_buffer((lanes,), "float32", name="input")
    output_buffer = tirx.decl_buffer((lanes,), output_dtype, name="output")
    ramp = tirx.Ramp(0, 1, lanes)
    mask_vector = tirx.BufferLoad(mask_buffer, [ramp])
    input_vector = tirx.BufferLoad(input_buffer, [ramp])
    if wrap_input_in_let:
        value = tirx.Var("value", f"float32x{lanes}")
        input_vector = tirx.Let(value, input_vector, value)
    if wrap_input_in_scalar_lets:
        scalar_lets = []
        for lane in range(lanes):
            value = tirx.Var(f"value_{lane}", "float32")
            load = tirx.BufferLoad(input_buffer, [lane])
            scalar_lets.append(tirx.Let(value, load, value))
        input_vector = tirx.Shuffle(scalar_lets, list(range(lanes)))
    if wrap_input_in_call:
        input_vector = tirx.call_pure_extern(f"float32x{lanes}", "exp2f", input_vector)
    if wrap_input_in_call_extern:
        input_vector = tirx.call_extern(f"float32x{lanes}", "exp2f", input_vector)
    if wrap_input_in_reinterpret:
        input_vector = tirx.reinterpret(f"int32x{lanes}", input_vector)
    if wrap_input_in_packed_call:
        bias = tirx.Broadcast(tirx.const(1, "float32"), lanes)
        input_vector = tirx.Call(f"float32x{lanes}", tvm.ir.Op.get("tl.add2"), [input_vector, bias])
    mask_zero = tirx.Broadcast(tirx.const(0, "float32"), lanes)
    result_zero = tirx.Broadcast(tirx.const(0, output_dtype), lanes)
    selected = tirx.Select(mask_vector > mask_zero, input_vector, result_zero)
    body = tirx.BufferStore(output_buffer, selected, [ramp])
    func = tirx.PrimFunc(
        [mask_buffer.data, input_buffer.data, output_buffer.data],
        body,
        buffer_map={
            mask_buffer.data: mask_buffer,
            input_buffer.data: input_buffer,
            output_buffer.data: output_buffer,
        },
    )
    func = func.with_attr("global_symbol", "vector_select")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"vector_select": func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


def _build_vector_select_with_vector_argument_source():
    lanes = 4
    value = tirx.Var("value", "float32x4")
    mask_buffer = tirx.decl_buffer((lanes,), "float32", name="mask")
    output_buffer = tirx.decl_buffer((lanes,), "float32", name="output")
    scalar_base = tirx.Shuffle([tirx.Ramp(0, 1, lanes)], [0])
    ramp = tirx.Ramp(scalar_base, 1, lanes)
    mask_vector = tirx.BufferLoad(mask_buffer, [ramp])
    zero = tirx.Broadcast(tirx.const(0, "float32"), lanes)
    selected = tirx.Select(mask_vector > zero, value, zero)
    body = tirx.BufferStore(output_buffer, selected, [ramp])
    func = tirx.PrimFunc(
        [value, mask_buffer.data, output_buffer.data],
        body,
        buffer_map={
            mask_buffer.data: mask_buffer,
            output_buffer.data: output_buffer,
        },
    )
    func = func.with_attr("global_symbol", "vector_select_argument")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"vector_select_argument": func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


def _build_narrow_vector_select_source(dtype, lanes):
    on_true = tirx.Var("on_true", f"{dtype}x{lanes}")
    on_false = tirx.Var("on_false", f"{dtype}x{lanes}")
    ramp = tirx.Ramp(0, 1, lanes)
    limit = tirx.Broadcast(tirx.const(lanes // 2, "int32"), lanes)
    selected = tirx.Select(
        ramp < limit,
        on_true,
        on_false,
    )
    func = tirx.PrimFunc([on_true, on_false], tirx.Evaluate(selected))
    func = func.with_attr("global_symbol", "narrow_vector_select")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"narrow_vector_select": func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


def _build_scalar_select_source():
    mask_buffer = tirx.decl_buffer((1,), "float32", name="mask")
    input_buffer = tirx.decl_buffer((1,), "float32", name="input")
    output_buffer = tirx.decl_buffer((1,), "float32", name="output")
    zero = tirx.const(0, "float32")
    selected = tirx.Select(
        tirx.BufferLoad(mask_buffer, [0]) > zero,
        tirx.BufferLoad(input_buffer, [0]),
        zero,
    )
    body = tirx.BufferStore(output_buffer, selected, [0])
    func = tirx.PrimFunc(
        [mask_buffer.data, input_buffer.data, output_buffer.data],
        body,
        buffer_map={
            mask_buffer.data: mask_buffer,
            input_buffer.data: input_buffer,
            output_buffer.data: output_buffer,
        },
    )
    func = func.with_attr("global_symbol", "scalar_select")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"scalar_select": func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


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


def test_vector_packed_call_keeps_argument_load_in_select_branch():
    source = _build_vector_select_source(2, wrap_input_in_packed_call=True)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 2
    assert all("? (input[" in line and "+ 1.000000e+00f" in line for line in assignments)
    assert "tl::add2" not in source


@pytest.mark.parametrize("lanes", [16, 32])
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
        (
            "float8_e5m2",
            8,
            ["x.x", "x.y", "x.z", "x.w", "y.x", "y.y", "y.z", "y.w"],
        ),
    ],
)
def test_float8_vector_results_are_stored_by_lane(dtype, lanes, lane_members):
    source = _build_narrow_vector_select_source(dtype, lanes)

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == lanes
    for lane, member in enumerate(lane_members):
        assert f"(({lane} < {lanes // 2}) ? (on_true).{member} : (on_false).{member})" in assignments[lane]


if __name__ == "__main__":
    test_vector_condition_select_is_scalarized_in_hip_source()
