"""Parameter identity regressions for CuTeDSL host source and launches."""

import ast

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper


def _scalar_program(name):
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32), n: T.int32, m: T.int32):
        with T.Kernel(n * 2 + m, threads=32) as bx:
            if bx == 0 and T.get_thread_binding(0) == 0:
                A[0] = n
                A[1] = m

    old = main.params[2]
    new = tilelang.tvm.tirx.Var(name, "int32")
    params = list(main.params)
    params[2] = new
    return tilelang.tvm.tirx.PrimFunc(
        params,
        tilelang.tvm.tirx.stmt_functor.substitute(main.body, {old: new}),
        buffer_map=main.buffer_map,
        attrs=main.attrs,
    )


def _lower(program, arch="sm_120"):
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

    try:
        check_cutedsl_available()
    except ImportError as error:
        pytest.skip(str(error))
    target = determine_target({"kind": "cutedsl", "arch": arch}, return_object=True)
    artifact = tilelang.lower(program, target=target, enable_host_codegen=False, enable_device_compile=False)
    wrapper = TLCuTeDSLSourceWrapper(
        tilelang.tvm.IRModule({program.attrs["global_symbol"]: program}),
        artifact.kernel_source,
        target,
        device_mod=artifact.device_mod,
        host_mod=artifact.host_mod,
    )
    return artifact, wrapper


@pytest.mark.parametrize("name", ["m", "n", "stream", "device_id", "_lib", "ctypes", "kernel_wrapper", "_tl_arg_0"])
def test_cutedsl_host_preserves_scalar_identity(name):
    program = _scalar_program(name)
    artifact, wrapper = _lower(program)
    compile(wrapper.host_func, "cutedsl_host.py", "exec")
    tree = ast.parse(wrapper.host_func)
    call = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "call")
    names = [arg.arg for arg in call.args.args]
    assert len(names) == len(set(names)) == 5
    assert names[-2:] == ["stream", "device_id"]
    host_args = wrapper._collect_host_kernel_call_sites()[0]["function_params"]
    generated_names = dict(zip([program.buffer_map[program.params[0]].data, *program.params[1:]], names[:3], strict=True))
    kernel_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == wrapper.function_names[0]
    )
    assert [ast.unparse(arg) for arg in kernel_call.args] == [
        generated_names[arg] + ("_" if arg.dtype == "handle" else "") for arg in host_args
    ]
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
        and node.func.value is kernel_call
    )
    grid = next(keyword.value for keyword in launch.keywords if keyword.arg == "grid")
    assert eval(compile(ast.Expression(grid), "grid.py", "eval"), {names[1]: 2, names[2]: 5}) == [9, 1, 1]
    launcher = wrapper.get_launcher_cpp_code()
    assert all(f"&{generated_names[arg]}" in launcher for arg in host_args)
    assert len(host_args) == len(next(iter(artifact.device_mod.functions.values())).params)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("name", ["n", "stream", "device_id", "_lib", "ctypes", "kernel_wrapper", "_tl_arg_0"])
def test_cutedsl_launch_preserves_scalar_identity(name):
    program = _scalar_program(name)
    kernel = tilelang.compile(program, target={"kind": "cutedsl", "arch": "sm_120"}, execution_backend="cutedsl")
    output = torch.empty(2, dtype=torch.int32, device="cuda")
    for first, second in ((2, 5), (7, 3)):
        kernel(output, first, second)
        torch.testing.assert_close(output, torch.tensor([first, second], dtype=torch.int32, device="cuda"))


def _shape_program():
    length = T.dynamic("size")
    other_length = T.dynamic("size")

    @T.prim_func
    def main(stream: T.Tensor((length,), T.int32), device_id: T.Tensor((other_length,), T.int32)):
        with T.Kernel(length, threads=32) as bx:
            if bx == 0 and T.get_thread_binding(0) == 0:
                stream[0] = other_length
        with T.Kernel(other_length, threads=32) as bx:
            if bx == 0 and T.get_thread_binding(0) == 0:
                device_id[0] = length

    return main, length, other_length


def test_cutedsl_host_preserves_distinct_same_named_shape_vars():
    main, length, other_length = _shape_program()
    _, wrapper = _lower(main)
    compile(wrapper.host_func, "cutedsl_shape_host.py", "exec")
    args, _ = wrapper._collect_function_args()
    assert [arg["var"] for arg in args[-2:]] == [length, other_length]
    assert len({arg["name"] for arg in args}) == 4


@tilelang.testing.requires_cuda
def test_cutedsl_launch_preserves_distinct_same_named_shape_vars():
    main, _, _ = _shape_program()
    kernel = tilelang.compile(main, target={"kind": "cutedsl", "arch": "sm_120"}, execution_backend="cutedsl")
    for first, second in ((3, 7), (5, 2)):
        a = torch.empty(first, dtype=torch.int32, device="cuda")
        b = torch.empty(second, dtype=torch.int32, device="cuda")
        kernel(a, b)
        assert a[0].item() == second
        assert b[0].item() == first


def test_cutedsl_host_preserves_tma_argument_identity():
    @T.prim_func
    def main(A: T.Tensor((64, 128), T.float16), B: T.Tensor((64, 128), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            barrier = T.alloc_barrier(1)
            T.tma_copy(A[0:64, 0:64], shared, barrier=barrier[0])
            T.barrier_arrive(barrier[0])
            T.mbarrier_wait_parity(barrier[0], 0)
            T.tma_copy(shared, B[0:64, 0:64])
            T.tma_store_wait(0)

    _, wrapper = _lower(main, "sm_90")
    assert wrapper.tma_descriptor_args
    compile(wrapper.host_func, "cutedsl_tma_host.py", "exec")
    args, _ = wrapper._collect_function_args()
    assert {info["globalAddress"] for info in wrapper.tma_desc_info.values()} == {arg["name"] for arg in args}
    assert "cuTensorMapEncodeTiled" in wrapper.get_launcher_cpp_code()


def _host_adapter(program):
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({"main": program})
    adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()
    return adapter


def test_cutedsl_adapter_preserves_shape_and_stride_identity():
    length = T.dynamic("size")
    stride = T.dynamic("size")

    @T.prim_func
    def main(A: T.StridedTensor[(length,), (stride,), T.float32]):
        T.evaluate(0)

    adapter = _host_adapter(main)
    tensor = torch.empty_strided((7,), (3,))
    assert adapter._resolve_dynamic_symbolic_value(length, [tensor]) == 7
    assert adapter._resolve_dynamic_symbolic_value(stride, [tensor]) == 3
    with pytest.raises(KeyError, match="ambiguous name-only"):
        adapter._resolve_dynamic_symbolic_value(T.dynamic("size"), [tensor])


def test_cutedsl_adapter_does_not_append_explicit_shape_scalar():
    tvm = tilelang.tvm
    length = tvm.tirx.Var("length", "int32")
    handle = tvm.tirx.Var("A", "handle")
    buffer = tvm.tirx.decl_buffer((length,), "float32", name="A")
    program = tvm.tirx.PrimFunc([handle, length], tvm.tirx.Evaluate(0), buffer_map={handle: buffer})
    adapter = _host_adapter(program)
    assert adapter.dynamic_symbolic_order == []
    assert adapter._resolve_dynamic_symbolic_value(length, [torch.empty(7), 7]) == 7


def test_cutedsl_repeated_kernel_uses_each_call_sites_scalar_identity():
    tvm = tilelang.tvm
    program = _scalar_program("n")
    _, wrapper = _lower(program)
    original = wrapper._collect_host_kernel_call_sites()[0]
    swapped = {program.params[1]: program.params[2], program.params[2]: program.params[1]}
    calls = [
        tvm.tirx.Evaluate(tvm.tirx.call_packed(original["function_name"], *original["function_params"])),
        tvm.tirx.Evaluate(
            tvm.tirx.call_packed(
                original["function_name"],
                *[tvm.tirx.stmt_functor.substitute(arg, swapped) for arg in original["function_params"]],
            )
        ),
    ]
    entry = wrapper._host_entry_func()
    gvar = next(gvar for gvar, func in wrapper.host_mod.functions.items() if func.same_as(entry))
    wrapper.host_mod.update_func(gvar, entry.with_body(tvm.tirx.SeqStmt(calls)))
    wrapper.update_lib_code(wrapper.lib_code)
    tree = ast.parse(wrapper.host_func)
    call = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "call")
    names = [arg.arg for arg in call.args.args]
    grids = [
        next(keyword.value for keyword in node.keywords if keyword.arg == "grid")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch"
    ]
    assert [eval(compile(ast.Expression(grid), "grid.py", "eval"), {names[1]: 2, names[2]: 5}) for grid in grids] == [
        [9, 1, 1],
        [12, 1, 1],
    ]
