"""Tests for CuTeDSL host codegen integration."""

import ast

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tvm_tirx import ir, tirx
from tilelang._compat.stmt_functor import ir_transform, post_order_visit

CUTEDSL_SM90_TARGET = {"kind": "cutedsl", "arch": "sm_90"}


def _require_cutedsl():
    """Skip when the CuTeDSL Python stack is unavailable."""
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _lower_cutedsl(program):
    """Lower a TileLang program and build its CuTeDSL host wrapper."""
    _require_cutedsl()

    from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper
    from tilelang.backend.target import determine_target

    target = determine_target(CUTEDSL_SM90_TARGET)
    with target:
        artifact = tilelang.lower(program, target=target)
    mod = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    wrapper = TLCuTeDSLSourceWrapper(mod, artifact.kernel_source, target, artifact.device_mod, artifact.host_mod)
    return artifact, wrapper


def _host_program_arguments(wrapper):
    constructor = next(
        node
        for node in ast.walk(ast.parse(wrapper.host_func))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CuTeHostProgram"
    )
    return [ast.literal_eval(argument) for argument in constructor.args[:-1]]


def _host_kernel_calls(wrapper):
    host = ir.load_json(_host_program_arguments(wrapper)[0])
    calls = []

    def collect(node):
        if isinstance(node, ir.Evaluate) and isinstance(node.value, ir.Call):
            call = node.value
            if isinstance(call.op, ir.Op) and call.op.name == "tirx.call_ffi_kernel":
                calls.append(call)

    post_order_visit(host, collect)
    return calls


def _rewrap(program, artifact, *, host_mod=None, device_mod=None):
    from tilelang.jit.adapter.cutedsl.wrapper import TLCuTeDSLSourceWrapper
    from tilelang.backend.target import determine_target

    return TLCuTeDSLSourceWrapper(
        ir.IRModule({program.attrs["global_symbol"]: program}),
        artifact.kernel_source,
        determine_target(CUTEDSL_SM90_TARGET),
        artifact.device_mod if device_mod is None else device_mod,
        artifact.host_mod if host_mod is None else host_mod,
    )


def single_kernel_program(N, block_size=64, dtype=T.float32):
    """Create a single-kernel CuTeDSL program."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        """Write one output tensor from one device kernel."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    B[idx] = A[idx] + 1.0

    return main


def repeated_dynamic_shape_program():
    """Create a program where one dynamic symbol appears in multiple params."""

    N = T.dynamic("N")

    @T.prim_func
    def main(
        optional: T.Tensor((N,), T.float32),
        source: T.Tensor((N,), T.float32),
        out: T.Tensor((N,), T.float32),
    ):
        T.evaluate(0)

    return main


def optional_stride_symbol_program():
    """Create a program with an optional strided tensor stride symbol."""

    N = T.dynamic("N")
    stride = T.dynamic("stride")

    @T.prim_func
    def main(
        source: T.Tensor((N,), T.float32),
        optional: T.StridedTensor[(N,), (stride,), T.float32],
        out: T.Tensor((N,), T.float32),
    ):
        T.evaluate(0)

    return main


def optional_shape_symbol_program():
    """Create a program with shape symbols that only belong to optional inputs."""

    N = T.dynamic("N")
    M = T.dynamic("M")

    @T.prim_func
    def main(
        source: T.Tensor((N,), T.float32),
        optional_a: T.Tensor((M,), T.float32),
        optional_b: T.Tensor((M,), T.float32),
        out: T.Tensor((N,), T.float32),
    ):
        T.evaluate(0)

    return main


def two_kernel_program(N, block_size=64, dtype=T.float32):
    """Create a program with two lowered host kernel call sites."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        """Launch two device kernels from one host wrapper."""

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    B[idx] = A[idx] + 1.0

        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx2,):
            for j in T.Parallel(block_size):
                idx = bx2 * block_size + j
                if idx < N:
                    C[idx] = B[idx] * 2.0

    return main


@tilelang.testing.requires_cuda
def test_cutedsl_host_wrapper_follows_lowered_host_call_sites():
    """Verify generated CuTeDSL launchers use lowered host call-site order."""

    _, wrapper = _lower_cutedsl(two_kernel_program(64))

    call_sites = _host_kernel_calls(wrapper)
    call_names = [call.args[0].value for call in call_sites]
    assert len(call_names) == 2
    assert call_names == wrapper.function_names
    assert [plan["name"] for plan in _host_program_arguments(wrapper)[2]] == call_names
    for call in call_sites:
        assert "stream" in call.attrs.launch_fields
        assert call.attrs.num_kernel_args == 2


@tilelang.testing.requires_cuda
def test_cutedsl_host_wrapper_preserves_repeated_same_kernel_call_sites():
    """Verify repeated host launches of the same device kernel remain distinct."""

    program = single_kernel_program(64)
    artifact, original = _lower_cutedsl(program)
    function_name = original.function_names[0]

    def repeat(node):
        if isinstance(node, ir.Evaluate) and isinstance(node.value, ir.Call):
            if isinstance(node.value.op, ir.Op) and node.value.op.name == "tirx.call_ffi_kernel":
                return ir.SeqStmt([ir.Evaluate(node.value), ir.Evaluate(node.value)])

    host_mod = ir_transform(artifact.host_mod, None, repeat)
    wrapper = _rewrap(program, artifact, host_mod=host_mod)
    call_sites = _host_kernel_calls(wrapper)
    call_names = [call.args[0].value for call in call_sites]
    assert call_names == [function_name, function_name]
    assert wrapper.function_names == [function_name]
    assert len(wrapper.kernel_plans) == 1
    ir.assert_structural_equal(call_sites[0], call_sites[1])


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_ignore_dead_helper_funcs():
    """Verify CuTeDSL launch collection only scans the host entry function."""

    program = single_kernel_program(64)
    artifact, original = _lower_cutedsl(program)
    entry = artifact.host_mod[program.attrs["global_symbol"]]
    dead_helper = ir.load_json(ir.save_json(entry)).with_attr({"global_symbol": "dead_helper", "tirx.is_entry_func": False})
    functions = dict(artifact.host_mod.functions.items())
    functions[ir.GlobalVar("dead_helper")] = dead_helper
    wrapper = _rewrap(program, artifact, host_mod=ir.IRModule(functions))
    assert [call.args[0].value for call in _host_kernel_calls(wrapper)] == original.function_names
    ir.assert_structural_equal(ir.load_json(_host_program_arguments(original)[0]), ir.load_json(_host_program_arguments(wrapper)[0]))


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_require_launch_metadata():
    """Verify missing launch metadata for a real host call site fails fast."""

    program = single_kernel_program(64)
    artifact, wrapper = _lower_cutedsl(program)
    function_name = wrapper.function_names[0]
    with pytest.raises(ValueError, match=f"{function_name}.*missing native device metadata"):
        _rewrap(program, artifact, device_mod=ir.IRModule({}))

    def invalidate(node):
        if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.call_ffi_kernel":
            attrs = tirx.CallFFIKernelAttr(
                launch_params=node.attrs.launch_params,
                launch_fields=node.attrs.launch_fields,
                num_kernel_args=100,
                kernel_attrs=node.attrs.kernel_attrs,
            )
            return ir.Call(node.op, node.args, attrs=attrs, ty=node.ty)

    with pytest.raises(ValueError, match="operand count does not match its signature"):
        _rewrap(program, artifact, host_mod=ir_transform(artifact.host_mod, None, invalidate))


@tilelang.testing.requires_cuda
def test_cutedsl_adapter_exposes_device_and_host_sources():
    """Verify CuTeDSL adapter exposes device, host, and combined sources."""

    _require_cutedsl()

    kernel = tilelang.compile(
        single_kernel_program(64),
        target=CUTEDSL_SM90_TARGET,
        execution_backend="cutedsl",
    )

    device_source = kernel.get_kernel_source(kernel_only=True)
    host_source = kernel.get_host_source()
    combined_source = kernel.get_kernel_source(kernel_only=False)

    assert "@cute.kernel" in device_source
    assert "def call(" in host_source
    assert "CuTeHostProgram" in host_source
    assert "'sm_90'" in host_source
    assert "return _program(arguments, stream, device_id)" in host_source
    assert device_source in combined_source
    assert host_source in combined_source


def test_cutedsl_adapter_resolves_dynamic_symbol_from_live_tensor_candidate():
    """Dynamic shape resolution should skip optional None tensor params."""

    _require_cutedsl()

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    program = repeated_dynamic_shape_program()
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()

    assert len(adapter.dynamic_symbolic_order) == 1
    dynamic_symbol = adapter.dynamic_symbolic_order[0]
    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbol, [None, torch.empty(7), torch.empty(7)]) == 7
    with pytest.raises(TypeError, match="no live tensor source"):
        adapter._resolve_dynamic_symbolic_value(dynamic_symbol, [None, None, None])


def test_cutedsl_adapter_allows_optional_stride_symbol_without_live_tensor():
    """Dynamic stride-only symbols may be absent for optional tensor params."""

    _require_cutedsl()

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    program = optional_stride_symbol_program()
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()

    dynamic_symbols = {sym.name: sym for sym in adapter.dynamic_symbolic_order}
    param_values = [torch.empty(7), None, torch.empty(7)]

    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["N"], param_values) == 7
    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["stride"], param_values) == 0


def test_cutedsl_adapter_allows_optional_abi_shape_symbol_without_live_tensor():
    """Dynamic shape symbols may be absent when only needed as dead ABI args."""

    _require_cutedsl()

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    program = optional_shape_symbol_program()
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()

    dynamic_symbols = {sym.name: sym for sym in adapter.dynamic_symbolic_order}
    param_values = [torch.empty(7), None, None, torch.empty(7)]

    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["N"], param_values) == 7
    with pytest.raises(TypeError, match="no live tensor source"):
        adapter._resolve_dynamic_symbolic_value(dynamic_symbols["M"], param_values)
    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["M"], param_values, require_live_shape=False) == 0


def test_cutedsl_cache_restore_does_not_fallback_to_host_source(monkeypatch, tmp_path):
    """Verify missing cached generated modules stay distinct from host wrappers."""

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
    from tilelang.jit.adapter.cutedsl import adapter as cutedsl_adapter

    class FakeLibraryGenerator:
        """Minimal CuTeDSL library generator stub for cache restore tests."""

        def __init__(self, target, verbose):
            self.target = target
            self.verbose = verbose
            self.pymodule = object()

        def assign_compile_flags(self, compile_flags):
            """Accept compile flags without invoking the real generator."""
            self.compile_flags = compile_flags

        def load_lib(self, lib_path):
            """Pretend to load a cached Python module from disk."""
            self.libpath = lib_path

    monkeypatch.setattr(cutedsl_adapter, "CuTeDSLLibraryGenerator", FakeLibraryGenerator)

    prim_func = tilelang.tvm.tirx.PrimFunc([], tilelang.tvm.tirx.Evaluate(0)).with_attr("global_symbol", "main")
    missing_kernel_py = tmp_path / "missing_kernel.py"
    restored = CuTeDSLKernelAdapter.from_database(
        params=[],
        result_idx=[],
        target=CUTEDSL_SM90_TARGET,
        func_or_mod=prim_func,
        host_kernel_source="# host wrapper",
        device_kernel_source="# device kernel",
        kernel_lib_path=str(missing_kernel_py),
    )

    assert restored.get_generated_module_source() is None
    assert restored.get_host_source() == "# host wrapper"
    assert restored.get_kernel_source(kernel_only=False) == "# device kernel\n\n# host wrapper"


if __name__ == "__main__":
    tilelang.testing.main()
