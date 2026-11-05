"""Tests for CuTeDSL host codegen integration."""

from types import SimpleNamespace

import tvm_ffi
from tvm_tirx import ir, tirx

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

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
    mod = ir.IRModule({program.attrs["global_symbol"]: program})
    wrapper = TLCuTeDSLSourceWrapper(mod, artifact.kernel_source, target, artifact.device_mod, artifact.host_mod)
    return artifact, wrapper


def _host_entry_gvar_and_func(wrapper):
    """Return the retained native source-host entry, before raw ABI projection."""
    program = wrapper.native_host.program
    return program.entry_global, program.entry


def _native_abi_plan(program):
    from tilelang.engine.param import KernelParam
    from tilelang.jit.adapter.abi import source_abi_plan
    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = ir.IRModule({program.attrs["global_symbol"]: program})
    adapter.params = [KernelParam.from_var(parameter) for parameter in program.params]
    adapter.result_idx = []
    adapter._initialize_native_arguments()
    return source_abi_plan(adapter)


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
    """Native launch entries preserve two lowered host call sites in order."""
    _, wrapper = _lower_cutedsl(two_kernel_program(64))
    call_sites = list(wrapper.native_host.kernels.items())
    call_names = [site["symbol"] for _, site in call_sites]
    assert len(call_names) == 2
    assert call_names == wrapper.function_names
    # The canonical provider owns C++ launching; the Python host carries the
    # same ordered launch graph and complete per-call native metadata.
    assert wrapper.get_launcher_cpp_code() is None
    first, second = [key for key, _ in call_sites]
    host_source = wrapper.host_func
    assert host_source.index(f"kernels[{first!r}](") < host_source.index(f"kernels[{second!r}](")
    events = []
    namespace = {}
    exec(host_source, namespace)
    source, temporary, output = [torch.empty(64, device="cuda") for _ in range(3)]
    namespace["call"](
        {key: lambda *args, _key=key: events.append(_key) for key, _ in call_sites},
        source.data_ptr(),
        temporary.data_ptr(),
        output.data_ptr(),
    )
    assert events == [first, second]


@tilelang.testing.requires_cuda
def test_cutedsl_host_wrapper_preserves_repeated_same_kernel_call_sites():
    """Repeated occurrences remain two launches of the same native device entry."""
    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    function_name = next(iter(wrapper.native_host.kernels.values()))["symbol"]
    entry_gvar, entry_func = _host_entry_gvar_and_func(wrapper)

    def duplicate(statement):
        if (
            isinstance(statement.value, ir.Call)
            and isinstance(statement.value.op, ir.Op)
            and statement.value.op.name == "tirx.call_ffi_kernel"
        ):
            return ir.SeqStmt([statement, ir.Evaluate(statement.value)])
        return statement

    body = tvm_ffi.structural_map(entry_func.body, (ir.Evaluate, duplicate))
    functions = dict(wrapper.native_host.program.functions)
    functions[entry_gvar] = entry_func.with_body(body)
    wrapper.host_mod = wrapper.host_mod.with_attr("tl.source_host_module", ir.IRModule(functions))
    wrapper.update_lib_code(wrapper.source)
    call_sites = list(wrapper.native_host.kernels.items())
    assert [site["symbol"] for _, site in call_sites] == [function_name, function_name]
    assert call_sites[0][1] is not call_sites[1][1]
    assert wrapper.get_launcher_cpp_code() is None
    keys = [key for key, _ in call_sites]
    assert wrapper.host_func.index(f"kernels[{keys[0]!r}](") < wrapper.host_func.index(f"kernels[{keys[1]!r}](")
    events = []
    namespace = {}
    exec(wrapper.host_func, namespace)
    source, output = [torch.empty(64, device="cuda") for _ in range(2)]
    namespace["call"]({key: lambda *args, _key=key: events.append(_key) for key in keys}, source.data_ptr(), output.data_ptr())
    assert events == keys


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_ignore_dead_helper_funcs():
    """An unreachable helper contributes no launch to the host entry execution."""
    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    call_sites = dict(wrapper.native_host.kernels)
    function_name = next(iter(call_sites.values()))["symbol"]
    dead_helper = tirx.Function(
        [],
        ir.SeqStmt([ir.Evaluate(tirx.call_packed(function_name))]),
        ir.TupleType([]),
    ).with_attr("global_symbol", "dead_helper")
    functions = dict(wrapper.native_host.program.functions)
    functions[ir.GlobalVar("dead_helper")] = dead_helper
    wrapper.host_mod = wrapper.host_mod.with_attr("tl.source_host_module", ir.IRModule(functions))
    wrapper.update_lib_code(wrapper.source)
    assert wrapper.native_host.kernels == call_sites


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_require_launch_metadata():
    """Missing or inconsistent typed launch metadata fails before CUDA launch."""
    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    entry_gvar, entry_func = _host_entry_gvar_and_func(wrapper)
    original_module = wrapper.host_mod
    for corruption in ("attrs", "kernel_arity", "launch_arity"):

        def corrupt(call, corruption=corruption):
            if not isinstance(call.op, ir.Op) or call.op.name != "tirx.call_ffi_kernel":
                return call
            attrs, args = call.attrs, list(call.args)
            if corruption == "attrs":
                attrs = None
            elif corruption == "kernel_arity":
                attrs = tirx.CallFFIKernelAttr(
                    launch_params=list(attrs.launch_params),
                    launch_fields=list(attrs.launch_fields),
                    num_kernel_args=int(attrs.num_kernel_args) + 1,
                    kernel_attrs=dict(attrs.kernel_attrs),
                )
            else:
                args.pop()
            return ir.Call(call.op, args, attrs=attrs, ty_args=call.ty_args, ty=call.ty)

        body = tvm_ffi.structural_map(entry_func.body, (ir.Call, corrupt))
        functions = dict(wrapper.native_host.program.functions)
        functions[entry_gvar] = entry_func.with_body(body)
        wrapper.host_mod = original_module.with_attr("tl.source_host_module", ir.IRModule(functions))
        try:
            with pytest.raises((ValueError, TypeError), match="metadata|arity|invalid native IR|CallFFIKernel"):
                wrapper.update_lib_code(wrapper.source)
        finally:
            wrapper.host_mod = original_module


@tilelang.testing.requires_cuda
def test_cutedsl_adapter_exposes_device_and_host_sources():
    """Device, generated native host, and combined sources stay distinguishable."""
    _require_cutedsl()
    kernel = tilelang.compile(single_kernel_program(64), target=CUTEDSL_SM90_TARGET, execution_backend="cutedsl")
    device_source = kernel.get_kernel_source(kernel_only=True)
    host_source = kernel.get_host_source()
    combined_source = kernel.get_kernel_source(kernel_only=False)
    assert "@cute.kernel" in device_source
    assert "def call(" in host_source
    namespace = {}
    exec(host_source, namespace)
    assert namespace["_cutedsl_arch"] == "sm_90"
    assert namespace["_native_cutedsl_schema"] == 1
    assert namespace["_cutedsl_compile_specs"]
    assert kernel.adapter.lib_generator.runtime._entries == {}  # Device compilation remains lazy.
    assert device_source in combined_source
    assert host_source in combined_source


def test_cutedsl_adapter_resolves_dynamic_symbol_from_live_tensor_candidate():
    """Dynamic shape resolution skips absent inputs and requires a real carrier."""
    _require_cutedsl()
    plan = _native_abi_plan(repeated_dynamic_shape_program())
    assert len(plan.implicit_symbols) == 1
    dynamic_symbol = plan.implicit_symbols[0]
    assert plan.resolve_implicit_value(dynamic_symbol, plan.bind_arguments([None, torch.empty(7), torch.empty(7)])) == 7
    with pytest.raises(TypeError, match="no live tensor source"):
        plan.resolve_implicit_value(dynamic_symbol, plan.bind_arguments([None, None, None]))


def test_cutedsl_adapter_allows_optional_stride_symbol_without_live_tensor():
    """A proven dead stride-only ABI slot preserves its zero placeholder."""
    _require_cutedsl()
    plan = _native_abi_plan(optional_stride_symbol_program())
    dynamic_symbols = {sym.name: sym for sym in plan.implicit_symbols}
    param_values = [torch.empty(7), None, torch.empty(7)]
    bindings = plan.bind_arguments(param_values)
    assert plan.resolve_implicit_value(dynamic_symbols["N"], bindings) == 7
    assert plan.resolve_implicit_value(dynamic_symbols["stride"], bindings) == 0


def test_cutedsl_adapter_allows_optional_abi_shape_symbol_without_live_tensor():
    """Optional shapes may fill dead raw ABI slots; live shapes remain required."""
    _require_cutedsl()
    plan = _native_abi_plan(optional_shape_symbol_program())
    dynamic_symbols = {sym.name: sym for sym in plan.implicit_symbols}
    param_values = [torch.empty(7), None, None, torch.empty(7)]
    bindings = plan.bind_arguments(param_values)
    assert plan.resolve_implicit_value(dynamic_symbols["N"], bindings) == 7
    with pytest.raises(TypeError, match="no live tensor source"):
        plan.resolve_implicit_value(dynamic_symbols["M"], bindings)
    assert plan.resolve_implicit_value(dynamic_symbols["M"], bindings, require_live_shape=False) == 0


def test_cutedsl_cache_restore_does_not_fallback_to_host_source(monkeypatch, tmp_path):
    """Verify missing cached generated modules stay distinct from host wrappers."""

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter
    from tilelang.jit.adapter.cutedsl import adapter as cutedsl_adapter

    class FakeLibraryGenerator:
        """Minimal CuTeDSL library generator stub for cache restore tests."""

        def __init__(self, target, verbose):
            self.target = target
            self.verbose = verbose
            self.pymodule = SimpleNamespace(_function_names=[])
            self.runtime = SimpleNamespace(kernels={})

        def assign_compile_flags(self, compile_flags):
            """Accept compile flags without invoking the real generator."""
            self.compile_flags = compile_flags

        def load_lib(self, lib_path):
            """Pretend to load a cached Python module from disk."""
            self.libpath = lib_path

    monkeypatch.setattr(cutedsl_adapter, "CuTeDSLLibraryGenerator", FakeLibraryGenerator)

    prim_func = tirx.Function([], ir.SeqStmt([ir.Evaluate(ir.const(0))]), ir.TupleType([])).with_attr("global_symbol", "main")
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
