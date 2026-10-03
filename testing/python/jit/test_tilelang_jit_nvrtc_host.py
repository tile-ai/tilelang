import ast

import torch
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import extrac_params
from tilelang.jit.adapter.nvrtc import is_nvrtc_available

if not is_nvrtc_available:
    pytest.skip("cuda-python is required to import the NVRTC adapter", allow_module_level=True)

from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper


def _make_scalar_identity_program(name):
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


@pytest.mark.parametrize("name", ["n", "stream", "kernels", "ctypes", "config", "arg_values", "res"])
def test_nvrtc_source_wrapper_preserves_scalar_identity(name):
    program = _make_scalar_identity_program(name)
    params = program.params
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    artifact = tilelang.lower(program, target=target, enable_host_codegen=False, enable_device_compile=False)
    wrapper = TLNVRTCSourceWrapper(
        tilelang.tvm.IRModule({"main": program}),
        artifact.kernel_source,
        target,
        device_mod=artifact.device_mod,
        host_mod=artifact.host_mod,
    )
    compile(wrapper.host_func, "nvrtc_host.py", "exec")
    tree = ast.parse(wrapper.host_func)
    call = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "call")
    names = [arg.arg for arg in call.args.args]
    assert names[0] == "kernels" and names[-1] == "stream"
    assert len(names) == len(set(names)) == 5
    values = next(
        node.value
        for node in ast.walk(call)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "arg_values" for target in node.targets)
    )
    assert isinstance(values, ast.Tuple)
    host_args = []

    def visit(node):
        if (
            isinstance(node, tilelang.tvm.tirx.Call)
            and node.op == tilelang.tvm.ir.Op.get("tirx.tvm_call_packed")
            and node.args[0] == wrapper.function_names[0]
        ):
            host_args.extend(node.args[1:4])

    for host in artifact.host_mod.functions.values():
        tilelang.tvm.tirx.stmt_functor.post_order_visit(host.body, visit)
    expected = {
        program.buffer_map[params[0]].data: f"{names[1]}.data_ptr()",
        params[1]: names[2],
        params[2]: names[3],
    }
    assert [ast.unparse(arg) for arg in values.elts] == [expected[arg] for arg in host_args]
    grid = next(
        node.value
        for node in ast.walk(call)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "gridDimX" for target in node.targets)
    )
    assert eval(compile(ast.Expression(grid), "grid.py", "eval"), {names[2]: 2, names[3]: 5}) == 9


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("name", ["n", "stream", "kernels", "ctypes", "config", "arg_values", "res"])
def test_nvrtc_launch_preserves_scalar_identity(name):
    program = _make_scalar_identity_program(name)
    kernel = tilelang.compile(program, execution_backend="nvrtc")
    output = torch.empty(2, dtype=torch.int32, device="cuda")
    for first, second in ((2, 5), (7, 3)):
        kernel(output, first, second)
        torch.testing.assert_close(output, torch.tensor([first, second], dtype=torch.int32, device="cuda"))


@tilelang.testing.requires_cuda
def test_nvrtc_launch_preserves_same_named_buffer_shapes():
    length = T.dynamic("size")
    other_length = T.dynamic("size")

    @T.prim_func
    def main(kernels: T.Tensor((length,), T.int32), stream: T.Tensor((other_length,), T.int32)):
        with T.Kernel(length, threads=32) as bx:
            if bx == 0 and T.get_thread_binding(0) == 0:
                kernels[0] = other_length
        with T.Kernel(other_length, threads=32) as bx:
            if bx == 0 and T.get_thread_binding(0) == 0:
                stream[0] = length

    kernel = tilelang.compile(main, execution_backend="nvrtc")
    for first, second in ((3, 7), (5, 2)):
        a = torch.empty(first, dtype=torch.int32, device="cuda")
        b = torch.empty(second, dtype=torch.int32, device="cuda")
        kernel(a, b)
        assert a[0].item() == second
        assert b[0].item() == first


def test_nvrtc_adapter_forwards_distinct_same_named_shape_and_stride_vars():
    length = T.dynamic("size")
    stride = T.dynamic("size")

    @T.prim_func
    def main(A: T.StridedTensor[(length,), (stride,), T.float32]):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))
    tensor = torch.empty_strided((7,), (3,))
    adapter._wrap_forward_from_prebuild_lib(tensor, stream=0)
    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (7, 3)
    assert stream == 0
    with pytest.raises(KeyError, match="ambiguous name-only"):
        adapter._resolve_dynamic_symbolic_value(T.dynamic("size"), [tensor])


def test_nvrtc_source_wrapper_preserves_tma_argument_identity():
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

    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        artifact = tilelang.lower(main, target=target, enable_host_codegen=False, enable_device_compile=False)
    wrapper = TLNVRTCSourceWrapper(
        tilelang.tvm.IRModule({"main": main}),
        artifact.kernel_source,
        target,
        device_mod=artifact.device_mod,
        host_mod=artifact.host_mod,
    )
    assert wrapper.tma_descriptor_args, "This control must reach native TMA lowering"
    compile(wrapper.host_func, "nvrtc_tma_host.py", "exec")
    tree = ast.parse(wrapper.host_func)
    call = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "call")
    names = [arg.arg for arg in call.args.args]
    assert len(names) == len(set(names)) == 4
    assert f"{names[1]}.data_ptr()" in wrapper.host_func
    assert f"{names[2]}.data_ptr()" in wrapper.host_func
    assignments = [node for node in ast.walk(call) if isinstance(node, ast.Assign)]
    descriptors = [
        node
        for node in assignments
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "cuTensorMapEncodeTiled"
    ]
    assert descriptors
    argument_values = next(
        node.value for node in assignments if any(isinstance(target, ast.Name) and target.id == "arg_values" for target in node.targets)
    )
    assert len(argument_values.elts) == len(next(iter(artifact.device_mod.functions.values())).params)


def _make_host_only_adapter(program, result_idx=None):
    adapter = NVRTCKernelAdapter.__new__(NVRTCKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.params = extrac_params(program)
    adapter.result_idx = [] if result_idx is None else result_idx
    adapter.param_dtypes = [param.torch_dtype() for param in adapter.params]
    adapter.param_shapes = [list(param.shape) for param in adapter.params]
    adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
    adapter.target = "cuda"
    return adapter


def test_nvrtc_adapter_forwards_scalar_primfunc_parameters():
    @T.prim_func
    def main(A: T.Tensor((8,), T.float32), offset: T.int32):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty(8)
    adapter._wrap_forward_from_prebuild_lib(tensor, 3, stream=0)

    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (3,)
    assert stream == 0


def test_nvrtc_adapter_forwards_dynamic_strides_after_dynamic_shapes():
    length = T.dynamic("length")
    stride = T.dynamic("stride")

    @T.prim_func
    def main(A: T.StridedTensor[(length,), (stride,), T.float32]):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty_strided((7,), (3,))
    adapter._wrap_forward_from_prebuild_lib(tensor, stream=0)

    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (7, 3)
    assert stream == 0


def test_nvrtc_adapter_scales_sub_byte_dynamic_strides():
    length = T.dynamic("length")
    stride = T.dynamic("stride")

    @T.prim_func
    def main(A: T.StridedTensor[(length,), (stride,), T.int4]):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main)
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty_strided((7,), (3,), dtype=torch.int8)
    adapter._wrap_forward_from_prebuild_lib(tensor, stream=0)

    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is tensor
    assert args[1:] == (7, 6)
    assert stream == 0


def test_nvrtc_adapter_resolves_output_shape_from_later_input():
    length = T.dynamic("length")

    @T.prim_func
    def main(B: T.Tensor((length,), T.float32), A: T.Tensor((length,), T.float32)):
        T.evaluate(0)

    adapter = _make_host_only_adapter(main, result_idx=[0])
    forwarded = []
    adapter._forward_from_prebuild_lib = lambda *args, stream: forwarded.append((args, stream))

    tensor = torch.empty(7)
    output = adapter._wrap_forward_from_prebuild_lib(tensor, stream=0)

    assert output.shape == (7,)
    assert len(forwarded) == 1
    args, stream = forwarded[0]
    assert args[0] is output
    assert args[1] is tensor
    assert args[2:] == (7,)
    assert stream == 0
