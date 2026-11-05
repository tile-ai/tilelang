"""Source host helpers preserve the entry's explicit stream and native calls."""

import ctypes

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT, SourceHostProgram


def helper_launch_graph(kind="cuda", depth=1):
    target = Target({"kind": "ascend", "keys": ["pto"]}) if kind == "pto" else Target(kind)
    n = ir.Var("n", "int64")
    a = ir.Var("A", tirx.TensorType([n], "float32", layout="default", data_alignment=1))
    b = ir.Var("B", tirx.TensorType([n], "float32", layout="default", data_alignment=1))
    globals_ = [ir.GlobalVar("main"), *[ir.GlobalVar(f"helper_{i}") for i in range(depth)]]
    functions = {}
    for index, global_var in enumerate(globals_[1:]):
        p, q = (ir.Var(name, ir.PtrType("float32", "global")) for name in ("p", "q"))
        count = ir.Var("n", "int64")
        if index == depth - 1:
            call = ir.Call(
                "tirx.call_ffi_kernel",
                [ir.StringImm("compute_kernel"), p, q, count, count],
                attrs=tirx.CallFFIKernelAttr(launch_params=["blockIdx.x"], num_kernel_args=3),
            )
        else:
            call = ir.Call(globals_[index + 2], [p, q, count], ty=ir.TupleType([]))
        functions[global_var] = tirx.Function(
            [p, q, count],
            [ir.Evaluate(call)],
            ir.TupleType([]),
            ir.DictAttrs({"global_symbol": global_var.name_hint, "target": target}),
        )
    entry = tirx.Function(
        [a, b],
        [ir.Evaluate(ir.Call(globals_[1], [a.data, b.data, n], ty=ir.TupleType([])))],
        ir.TupleType([]),
        ir.DictAttrs({"global_symbol": "main", "target": target}),
    )
    functions[globals_[0]] = entry
    host = ir.IRModule(functions).with_attr("tl.source_host_schema", 2)
    host = host.with_attr("tl.source_host_module", ir.IRModule(functions))
    device = tirx.Function(
        [ir.Var("x", ir.PtrType("float32", "global")), ir.Var("y", ir.PtrType("float32", "global")), ir.Var("n", "int64")],
        [],
        ir.TupleType([]),
        ir.DictAttrs(
            {
                "global_symbol": "compute_kernel",
                "calling_conv": 2,
                "target": target,
            }
        ),
    )
    return entry, host, ir.IRModule({"compute_kernel": device}), target


@pytest.mark.parametrize("kind", ["cuda", "hip", "ascend", "pto"])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_source_helpers_launch_on_the_entry_stream(kind, depth, roundtrip, tmp_path):
    entry, host, device, target = helper_launch_graph(kind, depth)
    if roundtrip:
        entry, host, device = (ir.load_json(ir.save_json(value)) for value in (entry, host, device))
    before = ir.save_json(host)
    program = SourceHostProgram(entry, host)
    raw = program.raw_module()
    assert tirx.analysis.check_well_formed(raw)
    ir.assert_structural_equal(raw, ir.load_json(ir.save_json(raw)))
    functions = dict(raw.functions.items())
    for _, function in functions.items():
        stream = function.params[int(function.attrs["tl.source_stream_param_index"])]
        assert stream.ty == ir.PtrType("void")
        calls = []
        tvm_ffi.structural_walk(function.body, (ir.Call, calls.append))
        for call in calls:
            if isinstance(call.op, ir.GlobalVar):
                assert len(call.args) == len(functions[call.op].params)
                assert call.args[-1].same_as(stream)
    source = program.emit_device(device, target)
    assert "compute_kernel" in source
    assert ir.save_json(host) == before
    (tmp_path / "host.cpp").write_text(source)
    (tmp_path / "raw.json").write_text(ir.save_json(raw))


@pytest.mark.parametrize("roundtrip", [False, True])
def test_nested_source_helpers_forward_the_actual_stream_value_in_compiled_host(roundtrip):
    entry, host, _, _ = helper_launch_graph("c", 3)
    if roundtrip:
        entry, host = (ir.load_json(ir.save_json(value)) for value in (entry, host))
    program = SourceHostProgram(entry, host)

    def host_oracle(call, stream):
        if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_ffi_kernel":
            return tirx.call_extern("stream_oracle", *call.args[1:4], stream, ty="void")
        return call

    source = NATIVE_SOURCE_INIT + program.emit(include_preamble=True, rewrite_call=host_oracle)
    source += r"""
extern "C" void stream_oracle(float* input, float* output, int64_t n, void* stream) {
  for (int64_t i = 0; i < n; ++i) output[i] = input[i] + (uintptr_t)stream;
}
"""
    generator = LibraryGenerator(Target("c"))
    generator.assign_pass_configs({})
    generator.assign_compile_flags([])
    generator.update_lib_code(source)
    generator.compile_lib()
    library = generator.load_lib()
    library.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    values = (ctypes.c_float * 7)(*range(7))
    output = (ctypes.c_float * 7)()
    for stream in [0, 37, 1024]:
        assert library.call(values, output, 7, ctypes.c_void_p(stream)) == 0
        assert list(output) == [float(value + stream) for value in range(7)]


def test_native_source_host_keeps_missing_helper_diagnostic():
    entry, host, _, _ = helper_launch_graph("c", 1)
    functions = dict(host.attrs["tl.source_host_module"].functions.items())
    functions = {global_var: function for global_var, function in functions.items() if function.attrs["global_symbol"] == "main"}
    host = host.with_attr("tl.source_host_module", ir.IRModule(functions))
    with pytest.raises(ValueError, match="missing a called helper"):
        SourceHostProgram(entry, host).raw_module()
