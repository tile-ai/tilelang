"""Native HIP emission keeps masked loads lazy through vector rewrites."""

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx.ir import prim as P
from tvm_tirx.script import tirx as T

I = tvm.ir


def _tensor_arg(name, dtype, size):
    data = I.Var(name, I.PtrType(I.PrimType(dtype)))
    tensor = I.Var(name + "_view", tvm.tirx.TensorType((size,), dtype, layout="default"))
    bind = I.Bind(tensor, I.Call("tirx.decl_tensor", [data], ty_args=[tensor.ty]))
    return data, tensor, bind


def _source(params, body, name="vector_select", target="rocm"):
    func = tvm.tirx.Function(params, body, ret_type=I.PrimType("void")).with_attr("global_symbol", name).with_attr("calling_conv", 2)
    return tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile")(
        tvm.IRModule({name: func}), tvm.target.Target(target)
    ).inspect_source("hip")


def _select_source(lanes, wrapper="none"):
    input_dtype = "bfloat16" if wrapper == "abs" else "float32"
    output_dtype = "int32" if wrapper == "reinterpret" else input_dtype
    declarations = [_tensor_arg(n, d, lanes) for n, d in [("mask", "float32"), ("input", input_dtype), ("output", output_dtype)]]
    mask, source, output = [t for _, t, _ in declarations]
    ramp = P.Ramp(0, 1, lanes)
    load = I.TensorLoad(source, [ramp])
    if wrapper == "let":
        var = I.Var("value", load.ty)
        load = P.Let(var, load, var)
    elif wrapper == "scalar_lets":
        scalars = []
        for lane in range(lanes):
            var = I.Var("value" + str(lane), "float32")
            scalars.append(P.Let(var, I.TensorLoad(source, [lane]), var))
        load = P.Shuffle(scalars, list(range(lanes)))
    elif wrapper in ("pure", "extern"):
        op = "tirx.call_pure_extern" if wrapper == "pure" else "tirx.call_extern"
        load = I.Call(op, [I.StringImm("exp2f"), load], ty=load.ty)
    elif wrapper == "reinterpret":
        load = I.Call("tirx.reinterpret", [load], ty=I.PrimType(f"int32x{lanes}"))
    elif wrapper == "abs":
        load = I.Call("tl.abs2", [load], ty_args=[load.ty])
    elif wrapper == "packed_add":
        load = I.Call("tl.add2", [load, P.Broadcast(I.const(1, "float32"), lanes)], ty_args=[load.ty])
    condition = I.TensorLoad(mask, [ramp]) > P.Broadcast(I.const(0, "float32"), lanes)
    result = P.Select(condition, load, P.Broadcast(I.const(0, output_dtype), lanes))
    return _source([p for p, _, _ in declarations], [*[b for _, _, b in declarations], I.TensorStore(output, [ramp], result)])


@pytest.mark.parametrize("lanes", [4, 8])
@pytest.mark.parametrize("wrapper", ["none", "let", "scalar_lets", "pure", "extern", "reinterpret", "abs", "packed_add"])
def test_masked_loads_remain_in_the_selected_branch(lanes, wrapper, tmp_path):
    source = _select_source(lanes, wrapper)
    (tmp_path / "kernel.hip").write_text(source)
    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == lanes
    for lane, assignment in enumerate(assignments):
        assert f"mask[{lane}]" in assignment
        branch = assignment.split("?", 1)[1]
        assert f"input[{lane}]" in branch
    # No eager vector load may precede the ternary guard.
    assert not any("input[" in line and "?" not in line for line in source.splitlines())
    assert "*)(input" not in source


@pytest.mark.parametrize("source_dtype,source_lanes,result_dtype,result_lanes", [("uint8", 8, "uint16", 4), ("uint16", 4, "uint8", 8)])
def test_lane_changing_reinterpret_reads_only_the_selected_source_lanes(source_dtype, source_lanes, result_dtype, result_lanes, tmp_path):
    declarations = [
        _tensor_arg(n, d, s)
        for n, d, s in [("mask", "float32", result_lanes), ("input", source_dtype, source_lanes), ("output", result_dtype, result_lanes)]
    ]
    mask, source, output = [t for _, t, _ in declarations]
    out_index = P.Ramp(0, 1, result_lanes)
    value = I.Call(
        "tirx.reinterpret", [I.TensorLoad(source, [P.Ramp(0, 1, source_lanes)])], ty=I.PrimType(f"{result_dtype}x{result_lanes}")
    )
    condition = I.TensorLoad(mask, [out_index]) > P.Broadcast(I.const(0, "float32"), result_lanes)
    selected = P.Select(condition, value, P.Broadcast(I.const(0, result_dtype), result_lanes))
    code = _source([p for p, _, _ in declarations], [*[b for _, _, b in declarations], I.TensorStore(output, [out_index], selected)])
    (tmp_path / "kernel.hip").write_text(code)
    assignments = [line for line in code.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == result_lanes
    for lane, assignment in enumerate(assignments):
        if source_lanes > result_lanes:
            ratio = source_lanes // result_lanes
            expected = set(range(lane * ratio, (lane + 1) * ratio))
        else:
            expected = {lane // (result_lanes // source_lanes)}
        for input_lane in range(source_lanes):
            assert (f"input[{input_lane}]" in assignment) == (input_lane in expected)
    assert not any("input[" in line and "?" not in line for line in code.splitlines())


@pytest.mark.parametrize("threads", [32, 64, 128])
def test_launch_bounds_keep_the_subwavefront_rule(threads):
    @T.function
    def kernel():
        with T.launch_thread("threadIdx.x", threads) as _tx:
            T.evaluate(0)

    kernel = kernel.with_attr("global_symbol", "kernel").with_attr("calling_conv", 2)
    source = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile")(
        tvm.IRModule({"kernel": kernel}), tvm.target.Target("rocm")
    ).inspect_source("hip")
    assert ("__launch_bounds__" in source) == (threads >= 64)
    if threads >= 64:
        assert f"__launch_bounds__({threads})" in source


def test_hip_source_only_does_not_call_hipcc():
    called = []
    name = "tilelang_callback_hip_compile"
    old = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda *args: called.append(args), override=True)
    try:
        source = _source([], [], "empty")
        assert "void empty(" in source
        assert not called
    finally:
        if old is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, old, override=True)
