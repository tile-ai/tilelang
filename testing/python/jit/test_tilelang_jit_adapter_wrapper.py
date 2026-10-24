import pytest
import torch
from types import SimpleNamespace

import tilelang
from tilelang import tvm
from tilelang.engine.lower import extrac_params
from tilelang.engine.param import dump_kernel_params, load_kernel_params
from tilelang.jit.adapter.shape import OutputShapeResolver
from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCPUSourceWrapper(
            scheduled_ir_module=tvm.IRModule(),
            source="",
            target=tvm.target.Target("c"),
        )


def _output_shape_program(dtype="float32"):
    tir = tvm.tirx
    n, stride, k = (tir.Var(name, "int32") for name in ("n", "stride", "k"))
    source = tir.decl_buffer((n,), dtype, name="A", strides=(stride,))
    output = tir.decl_buffer((n * 2, (n + k + 3) // 4, stride + 1), dtype, name="C")
    return tir.PrimFunc([output.data, source.data, k], tir.Evaluate(0), buffer_map={output.data: output, source.data: source}).with_attr(
        "global_symbol", "main"
    )


@pytest.mark.parametrize("backend", ["cython", "pto", "nvrtc", "cutedsl", "tvm_ffi"])
def test_adapter_allocates_expression_output(backend, monkeypatch):
    from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter, CythonKernelWrapper
    from tilelang.jit.adapter.pto.adapter import PTOKernelAdapter, PTOKernelWrapper
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    classes = dict(
        cython=CythonKernelAdapter,
        pto=PTOKernelAdapter,
        nvrtc=NVRTCKernelAdapter,
        cutedsl=CuTeDSLKernelAdapter,
        tvm_ffi=TVMFFIKernelAdapter,
    )
    cls = classes[backend]
    program = _output_shape_program()
    adapter = cls.__new__(cls)
    adapter.ir_module = tvm.IRModule({"main": program})
    # Cached metadata contains new Var objects. Allocation must use the same
    # PrimFunc for expressions and bindings, rather than those cached objects.
    adapter.params = load_kernel_params(dump_kernel_params(extrac_params(program)))
    adapter.result_idx = [0]
    adapter.target = "cuda"
    adapter.param_dtypes = [param.torch_dtype() for param in adapter.params]
    adapter.output_shape_resolver = OutputShapeResolver(program, [0])
    if backend == "cutedsl":
        adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()
    else:
        adapter.dynamic_symbolic_map = adapter._process_dynamic_symbolic()
    calls = []
    source = torch.empty_strided((7,), (5,))
    if backend in ("cython", "pto"):
        lib = SimpleNamespace(call=lambda *args: calls.append(args) or 0)
        wrapper_cls = CythonKernelWrapper if backend == "cython" else PTOKernelWrapper
        wrapper = wrapper_cls([0], adapter.params, lib, lambda: torch.device("cpu"), lambda device: 0)
        wrapper.set_output_shape_resolver(adapter.output_shape_resolver)
        wrapper.set_dynamic_symbolic_map(adapter.dynamic_symbolic_map)
        wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
        wrapper.set_ptr_map({})
        if backend == "pto":
            wrapper.set_scalar_param_vars(adapter._process_scalar_param_vars())
        output = wrapper.forward([source, 3], stream=0, skip_tensor_validation=True)
        assert [arg.value for arg in calls[0][2:-1]] == [3, 7, 5]
    elif backend == "tvm_ffi":
        adapter._ffi_callee_allocated_output_abi = False
        adapter.executable = lambda *args: calls.append(args)
        output = adapter._convert_torch_func()(source, 3)
        assert calls[0][1] is source
        assert calls[0][2:] == (3,)
    else:
        adapter._forward_from_prebuild_lib = lambda *args, **kwargs: calls.append(args)
        if backend == "cutedsl":
            # Exercise real allocation on CPU and stop at the adapter's normal
            # CUDA-device check; do not pretend to launch a CuTeDSL kernel.
            allocated = []
            empty = torch.empty

            def capture_empty(*args, **kwargs):
                tensor = empty(*args, **kwargs)
                allocated.append(tensor)
                return tensor

            monkeypatch.setattr(torch, "empty", capture_empty)
            with pytest.raises(ValueError, match="require CUDA tensors"):
                adapter._wrap_forward_from_prebuild_lib(source, 3, stream=0)
            assert len(allocated) == 1
            output = allocated[0]
        else:
            output = adapter._wrap_forward_from_prebuild_lib(source, 3, stream=0)
            assert calls[0][1] is source
            assert calls[0][2:] == (3, 7, 5)
    assert output.shape == (14, 3, 6)


def test_output_shape_requires_live_bindings():
    resolver = OutputShapeResolver(_output_shape_program(), [0])
    with pytest.raises(ValueError, match="output C dimension 0.*n.*no live input"):
        resolver(0, [None, None, 3])


@pytest.mark.parametrize("case", ["negative", "unresolved", "zero_divisor"])
def test_output_shape_reports_invalid_expression(case):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    expressions = {
        "negative": (n - 8, "negative extent -1"),
        "unresolved": (tir.call_extern("int32", "unresolved_dim", n), "did not resolve to an integer"),
        "zero_divisor": (n // (n - 7), "Divide by zero"),
    }
    expr, reason = expressions[case]
    output = tir.decl_buffer((expr,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0") as error:
        OutputShapeResolver(func, [1])(1, [7, None])
    assert reason in str(error.value)


@pytest.mark.parametrize("value", [None, 1.5, torch.tensor(7)])
def test_output_shape_rejects_noninteger_scalar(value):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    output = tir.decl_buffer((n * 2,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0"):
        OutputShapeResolver(func, [1])(1, [value, None])


def test_output_shape_optional_carrier_and_packed_units():
    from tilelang.jit.adapter.pto.adapter import PTOKernelAdapter, PTOKernelWrapper

    tir = tvm.tirx
    n = tir.Var("n", "int32")
    a = tir.decl_buffer((n,), "int4", name="A")
    b = tir.decl_buffer((n,), "int4", name="B")
    c = tir.decl_buffer((n * 2,), "int4", name="C")
    func = tir.PrimFunc([a.data, b.data, c.data], tir.Evaluate(0), buffer_map={a.data: a, b.data: b, c.data: c})
    resolver = OutputShapeResolver(func, [2], shape_scale=lambda buffer, dim: 2)
    source = torch.empty(7, dtype=torch.int8)
    assert resolver(2, [None, source, None]) == [28]
    adapter = PTOKernelAdapter.__new__(PTOKernelAdapter)
    adapter.ir_module = tvm.IRModule({"main": func})
    adapter.result_idx = [2]
    calls = []
    wrapper = PTOKernelWrapper(
        [2], extrac_params(func), SimpleNamespace(call=lambda *args: calls.append(args) or 0), lambda: torch.device("cpu"), lambda device: 0
    )
    wrapper.set_output_shape_resolver(resolver)
    wrapper.set_dynamic_symbolic_map(adapter._process_dynamic_symbolic())
    wrapper.set_dynamic_symbolic_sources(adapter._process_dynamic_symbolic_sources())
    wrapper.set_param_storage_metadata([(0, 2)] * 3)
    wrapper.set_ptr_map({})
    output = wrapper.forward([None, source], stream=0, skip_tensor_validation=True)
    assert output.shape == (14,)
    assert calls[0][-2].value == 14


def test_output_shape_preserves_symbol_identity():
    tir = tvm.tirx
    left, right = tir.Var("n", "int32"), tir.Var("n", "int32")
    output = tir.decl_buffer((left + right,), "float32", name="C")
    func = tir.PrimFunc([left, right, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    assert OutputShapeResolver(func, [2])(2, [3, 7, None]) == [10]
    with pytest.raises(ValueError, match="ambiguous symbol name 'n'"):
        OutputShapeResolver(func, [2], require_unique_names=True)


@pytest.mark.parametrize("backend", ["cython", "tvm_ffi"])
@pytest.mark.parametrize("source_kind", ["shape", "scalar"])
def test_expression_output_and_cache(backend, source_kind, tmp_path, monkeypatch):
    import tilelang.cpu.language as T
    from tilelang.cache import _dispatch_map
    from tilelang.env import env
    from tilelang.jit import JITKernel

    if backend == "tvm_ffi" and not tvm.runtime.enabled("llvm"):
        pytest.skip("LLVM is required for native TVM-FFI CPU execution")
    n = T.dynamic("n")

    @T.prim_func
    def from_shape(A: T.Tensor((n,), "float32")):
        C = T.empty((n * 2,), "float32")
        with T.Kernel(1):
            for i in T.serial(n * 2):
                C[i] = A[i % n]
        return C

    @T.prim_func
    def from_scalar(A: T.Tensor((1,), "float32"), size: T.int32):
        C = T.empty((size * 2,), "float32")
        D = T.empty((size,), "float32")
        with T.Kernel(1):
            for i in T.serial(size):
                C[i * 2] = A[0]
                C[i * 2 + 1] = A[0]
                D[i] = A[0]
        return C, D

    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    cache = _dispatch_map[backend]
    monkeypatch.setattr(cache, "_memory_cache", {})
    program = from_shape if source_kind == "shape" else from_scalar
    target = "c" if backend == "cython" else "llvm"
    kernel = tilelang.compile(program, target=target, execution_backend=backend)
    cache._memory_cache.clear()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("The second compile must restore the disk cache")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", unexpected_compile)
    restored = tilelang.compile(program, target=target, execution_backend=backend)
    assert restored is not kernel
    for compiled in (kernel, restored):
        for size in (0, 7):
            if source_kind == "shape":
                source = torch.arange(size, dtype=torch.float32)
                torch.testing.assert_close(compiled(source), source.repeat(2))
            else:
                large, small = compiled(torch.tensor([3.0]), size)
                torch.testing.assert_close(large, torch.full((size * 2,), 3.0))
                torch.testing.assert_close(small, torch.full((size,), 3.0))


if __name__ == "__main__":
    pytest.main([__file__])
