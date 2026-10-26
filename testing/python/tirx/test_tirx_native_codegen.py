import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


def cuda_target():
    device = runtime.cuda(0)
    architecture = device.compute_version.replace(".", "") if device.exist else "80"
    return Target({"kind": "cuda", "arch": f"sm_{architecture}", "backend_config": {"cuda": {"compiler": "nvcc"}}}, host="c")


def kernel(parameters=(), body=(), name="native_kernel"):
    function = tirx.Function(list(parameters), list(body), ret_type=ir.TupleType([]))
    return function.with_attr("global_symbol", name).with_attr("calling_conv", 2)


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_mma_register_offsets_count_scalar_elements(lanes, tilelang_builders):
    suffix = f"x{lanes}" if lanes > 1 else ""
    left = ir.Var("left", ir.PtrType(f"float16{suffix}", scope="local"))
    right = ir.Var("right", ir.PtrType(f"float16{suffix}", scope="local"))
    output = ir.Var("output", ir.PtrType(f"float32{suffix}", scope="local"))
    call = ir.Call(
        "tirx.ptx_legacy.mma",
        ["m16n8k16", "row", "col", "fp16", "fp16", "fp32", left, 8, right, 12, output, 16, ir.const(False)],
        ty="float32",
    )
    module = ir.IRModule({"main": kernel([left, right, output], [ir.Evaluate(call)])})
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    source = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(module, cuda_target()).inspect_source()
    assert "mma_sync<" in source
    for name, offset in [("left", 16), ("right", 24), ("output", 64)]:
        assert f"(char*){name}) + {offset})" in source


@pytest.mark.parametrize("target", ["c", "cuda"])
def test_native_codegen_adapter_executes(target, tilelang_builders):
    device = runtime.cuda(0) if target == "cuda" else runtime.cpu(0)
    if not device.exist:
        pytest.skip("Device runtime is unavailable")
    source = TensorView.create("source", [32], "float32")
    output = TensorView.create("output", [32], "float32")
    index = ir.Var("index", "int32")
    body = output.store(index, source.load(index) * 2 + 1)
    if target == "cuda":
        body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32)], [index], None, body)
        target = cuda_target()
    else:
        body = ir.For(index, 0, 32, 1, ir.ForKind.DEFAULT, body)
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr(
        "global_symbol", "native_adapter_test"
    )
    executable = tirx.compile(function, target=target)
    actual = runtime.empty((32,), "float32", device=device)
    values = np.arange(32, dtype="float32")
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), values * 2 + 1)


def test_source_only_keeps_real_source_and_never_compiles():
    calls = []
    name = "tilelang_callback_cuda_compile"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda *args: calls.append(args), override=True)
    try:
        source = 'extern "C" __global__ void raw_kernel() {}'
        function = kernel(name="raw_kernel").with_attr("code_block_source", source).with_attr("code_block_entry_name", "raw_kernel")
        module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"main": function}), cuda_target())
        assert source in module.inspect_source("cuda")
        binary, output_format, symbols = module["__tvm_cuda_binary"]()
        assert output_format == "cuda"
        assert source.encode() in bytes(binary)
        assert list(symbols) == ["raw_kernel"]
        assert not calls
        with pytest.raises(RuntimeError, match="cannot launch"):
            module["raw_kernel"]()
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)


def test_duplicate_symbols_and_raw_source_entry_are_validated():
    build = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")
    function = kernel()
    with pytest.raises(ValueError, match="Duplicate device global_symbol"):
        build(ir.IRModule({"first": function, "second": function}), cuda_target())
    function = function.with_attr("code_block_source", "invalid source").with_attr("code_block_entry_name", "wrong_entry")
    with pytest.raises(ValueError, match="entry_name"):
        build(ir.IRModule({"main": function}), cuda_target())


def test_parameter_qualifiers_use_native_indices():
    first = ir.Var("first", ir.PtrType("float32", scope="global"))
    second = ir.Var("second", ir.PtrType("float32", scope="global"))
    function = kernel([first, second]).with_attr("tirx.noalias", True)
    function = function.with_attr("tl.readonly_param_indices", [ir.const(0)]).with_attr("tl.non_restrict_param_indices", [ir.const(1)])
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(ir.IRModule({"main": function}), cuda_target())
    code = module.inspect_source("cuda")
    assert "const float* __restrict__ first" in code
    assert "float* second" in code
    assert "__restrict__ second" not in code


def test_imported_source_and_line_directives(tilelang_builders):
    output = TensorView.create("output", [1], "int32")
    location = ir.SourceLoc(ir.SourceName('input"kernel.py'), 71, 0, 71, 12)
    value = ir.Call("tirx.call_pure_extern", [ir.StringImm("imported_value")], ty="int32")
    store = ir.TensorStore(output.tensor, [0], value, loc=location)
    imported = '#ifdef __cplusplus\nextern "C" {\n#endif\nint32_t imported_value(void) { return 23; }\n#ifdef __cplusplus\n}\n#endif\n'
    body = ir.RegionStmt("tl.scope", [], [], {"pragma_import_c": imported}, store)
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "imported_source_test")
    with PassContext(config={"tl.emit_line_directives": True}):
        executable = tirx.compile(function, target="c")
    source = executable.mod.inspect_source()
    assert '#line 71 "input\\"kernel.py"' in source
    assert source.index("#include") < source.index("int32_t imported_value")
    assert source.index("int32_t imported_value") < source.index("imported_source_test(")
    actual = runtime.empty((1,), "int32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), [23])


def test_cuda_compile_callback_and_postprocessor_execute(tilelang_builders):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA runtime is unavailable")
    from tvm_tirx.backend.cuda.compiler import compile_source

    calls = []

    def postprocess(source, target):
        calls.append("postprocess")
        return source + "\n"

    def compile_callback(source, target, config):
        calls.append("compile")
        assert config["tl.emit_line_directives"]
        result = compile_source(source, {"cuda": {"compiler": "nvcc", "target_format": "ptx"}}, target=target)
        return result.binary

    callbacks = {"tilelang_callback_cuda_postproc": postprocess, "tilelang_callback_cuda_compile": compile_callback}
    previous = {name: tvm_ffi.get_global_func(name, allow_missing=True) for name in callbacks}
    try:
        for name, callback in callbacks.items():
            tvm_ffi.register_global_func(name, callback, override=True)
        output = TensorView.create("output", [1], "int32")
        thread = ir.Var("thread", "int32")
        body = ir.RegionStmt(
            "tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(1)], [thread], None, output.store(0, ir.const(31))
        )
        function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "compile_callback_test")
        with PassContext(config={"tl.emit_line_directives": True}):
            executable = tirx.compile(function, target=cuda_target())
        actual = runtime.empty((1,), "int32", device=device)
        executable(actual)
        np.testing.assert_array_equal(actual.numpy(), [31])
        assert calls == ["postprocess", "compile"]
    finally:
        for name, callback in previous.items():
            if callback is None:
                tvm_ffi.remove_global_func(name)
            else:
                tvm_ffi.register_global_func(name, callback, override=True)


def test_cpu_parallel_loops_execute_with_openmp(tmp_path, monkeypatch, tilelang_builders):
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")
    output = TensorView.create("output", [8, 8], "int32")
    row = ir.Var("row", "int32")
    column = ir.Var("column", "int32")
    worker = ir.Call("tirx.call_extern", [ir.StringImm("omp_get_thread_num")], ty="int32")
    body = ir.For(
        row,
        0,
        8,
        1,
        ir.ForKind.PARALLEL,
        ir.For(column, 0, 8, 1, ir.ForKind.PARALLEL, output.store([row, column], worker)),
        annotations={"tl.cpu_num_threads": ir.const(2)},
    )
    function = tirx.Function([output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "parallel_codegen_test")
    executable = tirx.compile(function, target="c")
    source = executable.mod.inspect_source()
    assert "#pragma omp parallel for collapse(2) num_threads(2)" in source
    library = tmp_path / "parallel_codegen.so"
    executable.export_library(str(library), options=["-fopenmp"])
    loaded = runtime.load_module(str(library))
    actual = runtime.empty((8, 8), "int32")
    loaded["parallel_codegen_test"](actual)
    np.testing.assert_array_equal(np.unique(actual.numpy()), [0, 1])
