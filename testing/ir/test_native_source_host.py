"""Actual source adapters consume the typed host graph across serialization."""

import pytest
import torch
import tvm_ffi
import ctypes
from tvm_tirx import ir, tirx

import tilelang
import tilelang.cpu.language as T
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang.jit.adapter.source_host import SourceHostProgram
from test_native_cpu_jit import add_one


@pytest.mark.parametrize("size", [0, 1, 17, 257])
def test_public_cython_cpu_compiles_calls_and_allocates(size):
    function = add_one(T.dynamic("n", "int64"))
    kernel = tilelang.compile(
        function,
        target="c",
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cython",
        out_idx=[1],
    )
    a = torch.arange(size, dtype=torch.float32)
    actual = kernel(a)
    torch.testing.assert_close(actual, a + 1, rtol=0, atol=0)
    assert actual.shape == a.shape
    assert "tl_arg_0" in kernel.adapter.get_host_source()


@pytest.mark.parametrize("roundtrip", [False, True])
def test_raw_host_and_device_json_reloads_execute(roundtrip):
    function = add_one(T.dynamic("n", "int64"))
    artifact = tilelang.lower(function, target="c", target_host={"kind": "llvm", "mcpu": "generic"})
    host, device = artifact.host_mod, artifact.device_mod
    if roundtrip:
        function = ir.load_json(ir.save_json(function))
        host = ir.load_json(ir.save_json(host))
        device = ir.load_json(ir.save_json(device))
    before = ir.save_json(host)
    program = SourceHostProgram(function, host)
    raw = program.raw_module()
    assert tirx.analysis.check_well_formed(raw)
    calls = []
    tvm_ffi.structural_walk(raw, (ir.Call, calls.append))
    assert any(getattr(call.op, "name", "") == "tirx.call_extern" and str(call.args[0].value) == "main_kernel" for call in calls)
    adapter = CythonKernelAdapter(
        artifact.params,
        [1],
        "c",
        function,
        host_mod=host,
        device_mod=device,
        device_kernel_source=artifact.kernel_source,
        pass_configs={},
    )
    a = torch.arange(257, dtype=torch.float32)
    torch.testing.assert_close(adapter(a), a + 1, rtol=0, atol=0)
    assert ir.save_json(host) == before


def test_source_host_rejects_missing_native_graph():
    function = add_one(T.dynamic("n", "int64"))
    with pytest.raises(ValueError, match="retained by MakePackedAPI"):
        SourceHostProgram(function, ir.IRModule({"main": function}))


@pytest.mark.parametrize("storage_offset", [0, 3, 8])
def test_offset_only_parameter_uses_logical_pointer_once(storage_offset):
    @T.prim_func
    def main(A: T.Buffer((7,), "float32", align=4, offset_factor=4), B: T.Tensor((7,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(7):
                B[i] = A[i] + T.float32(1)

    kernel = tilelang.compile(
        main,
        target="c",
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cython",
        out_idx=[1],
    )
    data = torch.arange(7 + storage_offset, dtype=torch.float32)[storage_offset:]
    torch.testing.assert_close(kernel(data), data + 1, rtol=0, atol=0)
    plan = kernel.adapter._source_abi_plan
    offset = plan.tensors[0].element_offset
    assert isinstance(offset, ir.Var) and offset in plan.implicit_symbols
    assert plan.bind_arguments([data, None]).values[offset] == 0


@pytest.mark.parametrize("roundtrip", [False, True])
def test_empty_tuple_return_survives_packing_and_source_emission(roundtrip):
    from tvm_tirx.target import Target

    n = ir.Var("n", "int64")
    function = tirx.Function(
        [n],
        [ir.If(n == 0, [ir.Return(ir.Tuple([]))], []), ir.Return(ir.Tuple([]))],
        ir.TupleType([]),
        ir.DictAttrs(
            {
                "global_symbol": "main",
                "target": Target("c", Target({"kind": "llvm", "mcpu": "generic"})),
            }
        ),
    )
    packed = tvm_ffi.get_global_func("tl.transform.MakePackedAPI")()(ir.IRModule({"main": function}))
    if roundtrip:
        function = ir.load_json(ir.save_json(function))
        packed = ir.load_json(ir.save_json(packed))
    raw = SourceHostProgram(function, packed).raw_module()
    returns = []
    tvm_ffi.structural_walk(raw, (ir.Return, returns.append))
    assert tirx.analysis.check_well_formed(raw)
    assert returns and all(value.value.ty == ir.PrimType("int32") for value in returns)


def test_source_void_call_return_keeps_side_effect():
    from tvm_tirx.target import Target
    from tilelang.jit.adapter.libgen import LibraryGenerator
    from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT

    pointer = ir.Var("counter", ir.PtrType("int32"))
    function = tirx.Function(
        [pointer],
        [ir.Return(tirx.call_extern("increment", pointer, ty="void"))],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "main", "target": Target("c")}),
    )
    global_var = ir.GlobalVar("main")
    module = ir.IRModule({global_var: function}).with_attr("tl.source_host_schema", 2)
    module = module.with_attr("tl.source_host_module", ir.IRModule({global_var: function}))
    program = SourceHostProgram(function, module)
    source = NATIVE_SOURCE_INIT + program.emit(include_preamble=True)
    source += '\nextern "C" void increment(int32_t* counter) { ++*counter; }\n'
    generator = LibraryGenerator(Target("c"))
    generator.assign_pass_configs({})
    generator.assign_compile_flags([])
    generator.update_lib_code(source)
    generator.compile_lib()
    library = generator.load_lib()
    counter = ctypes.c_int32(0)
    assert library.call(ctypes.byref(counter), None) == 0
    assert counter.value == 1


@pytest.mark.parametrize("canonical_void", [False, True])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_packed_void_return_preserves_effect_and_returns_none(canonical_void, roundtrip):
    from tvm_tirx.target import Target
    from tilelang.testing.packed import compile_packed_function

    calls = []
    callback_name = "tl.testing.packed_void_return_effect"
    tvm_ffi.register_global_func(callback_name, lambda: calls.append(1), override=True)
    n = ir.Var("n", "int32")
    void_type = ir.TupleType([]) if canonical_void else ir.PrimType("void")
    effect = ir.Call("tirx.call_packed", [ir.StringImm(callback_name)], ty=void_type)
    target = Target("c")
    function = tirx.Function(
        [n],
        [ir.If(n == 0, [ir.Return(ir.Tuple([]))], []), ir.Return(effect)],
        void_type,
        ir.DictAttrs({"global_symbol": "main", "target": Target(target, host=target)}),
    )
    module = ir.IRModule({"main": function})
    if roundtrip:
        module = ir.load_json(ir.save_json(module))
    executable = compile_packed_function(module)
    assert executable(0) is None and calls == []
    assert executable(1) is None and calls == [1]
    assert executable(0) is None and calls == [1]
