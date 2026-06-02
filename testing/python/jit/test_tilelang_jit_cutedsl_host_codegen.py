"""Tests for CuTeDSL host codegen integration."""

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
    from tilelang.utils.target import determine_target

    target = determine_target(CUTEDSL_SM90_TARGET)
    with target:
        artifact = tilelang.lower(program, target=target)
    mod = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    wrapper = TLCuTeDSLSourceWrapper(mod, artifact.kernel_source, target, artifact.device_mod, artifact.host_mod)
    return artifact, wrapper


def _host_entry_gvar_and_func(wrapper):
    """Return the GlobalVar and PrimFunc selected as the wrapper host entry."""
    entry_func = wrapper._host_entry_func()
    for gvar, host_func in wrapper.host_mod.functions.items():
        if host_func.same_as(entry_func):
            return gvar, host_func
    raise AssertionError("Unable to locate wrapper host entry function in host_mod")


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
        with T.Kernel(T.ceildiv(N, 64), threads=64) as (bx,):
            for i in T.Parallel(64):
                idx = bx * 64 + i
                if idx < N:
                    out[idx] = source[idx]

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
        with T.Kernel(T.ceildiv(N, 64), threads=64) as (bx,):
            for i in T.Parallel(64):
                idx = bx * 64 + i
                if idx < N:
                    out[idx] = source[idx]

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

    call_sites = wrapper._collect_host_kernel_call_sites()
    call_names = [call_site["function_name"] for call_site in call_sites]
    assert len(call_names) == 2
    assert call_names == wrapper.function_names

    launcher_cpp = wrapper.get_launcher_cpp_code()
    first_launch = launcher_cpp.index(f"Launch kernel 0: {call_names[0]}")
    second_launch = launcher_cpp.index(f"Launch kernel 1: {call_names[1]}")
    assert first_launch < second_launch

    host_source = wrapper.host_func
    first_cubin_launch = host_source.index(f"{call_names[0]}(")
    second_cubin_launch = host_source.index(f"{call_names[1]}(")
    assert first_cubin_launch < second_cubin_launch


@tilelang.testing.requires_cuda
def test_cutedsl_host_wrapper_preserves_repeated_same_kernel_call_sites():
    """Verify repeated host launches of the same device kernel remain distinct."""

    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    original_call_site = wrapper._collect_host_kernel_call_sites()[0]
    function_name = original_call_site["function_name"]
    repeated_body = tilelang.tvm.tirx.SeqStmt(
        [
            tilelang.tvm.tirx.Evaluate(tilelang.tvm.tirx.call_packed(function_name, *original_call_site["function_params"])),
            tilelang.tvm.tirx.Evaluate(tilelang.tvm.tirx.call_packed(function_name, *original_call_site["function_params"])),
        ]
    )

    entry_gvar, entry_func = _host_entry_gvar_and_func(wrapper)
    wrapper.host_mod = tilelang.tvm.IRModule({entry_gvar.name_hint: entry_func.with_body(repeated_body)})

    call_sites = wrapper._collect_host_kernel_call_sites()
    call_names = [call_site["function_name"] for call_site in call_sites]
    assert call_names == [function_name, function_name]
    assert call_sites[0] is not call_sites[1]

    wrapper.update_lib_code(wrapper.source)
    launcher_cpp = wrapper.get_launcher_cpp_code()
    first_launch = launcher_cpp.index(f"Launch kernel 0: {function_name}")
    second_launch = launcher_cpp.index(f"Launch kernel 1: {function_name}")
    assert first_launch < second_launch

    host_source = wrapper.host_func
    kernel_wrapper_source = host_source[host_source.index("def kernel_wrapper") :]
    first_cubin_launch = kernel_wrapper_source.index(f"{function_name}(")
    second_cubin_launch = kernel_wrapper_source.index(f"{function_name}(", first_cubin_launch + 1)
    assert first_cubin_launch < second_cubin_launch


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_ignore_dead_helper_funcs():
    """Verify CuTeDSL launch collection only scans the host entry function."""

    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    call_sites = wrapper._collect_host_kernel_call_sites()
    function_name = call_sites[0]["function_name"]

    entry_gvar, entry_func = _host_entry_gvar_and_func(wrapper)
    dead_helper = tilelang.tvm.tirx.PrimFunc(
        [],
        tilelang.tvm.tirx.Evaluate(tilelang.tvm.tirx.call_packed(function_name)),
    ).with_attr("global_symbol", "dead_helper")
    wrapper.host_mod = tilelang.tvm.IRModule(
        {
            entry_gvar.name_hint: entry_func,
            "dead_helper": dead_helper,
        }
    )

    assert wrapper._collect_host_kernel_call_sites() == call_sites


@tilelang.testing.requires_cuda
def test_cutedsl_host_call_sites_require_launch_metadata():
    """Verify missing launch metadata for a real host call site fails fast."""

    _, wrapper = _lower_cutedsl(single_kernel_program(64))
    function_name = wrapper._collect_host_kernel_call_sites()[0]["function_name"]

    for metadata_name in ("block_info", "grid_info", "dynamic_smem_buf"):
        original_metadata = getattr(wrapper, metadata_name)
        metadata = dict(original_metadata)
        metadata.pop(function_name)
        try:
            setattr(wrapper, metadata_name, metadata)
            with pytest.raises(AssertionError, match=f".*{function_name}.*{metadata_name}.*"):
                wrapper.update_lib_code(wrapper.source)
        finally:
            setattr(wrapper, metadata_name, original_metadata)


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
    assert "--gpu-arch=sm_90" in host_source
    assert "_generate_cubin_if_needed" in host_source
    assert device_source in combined_source
    assert host_source in combined_source


def test_cutedsl_adapter_resolves_dynamic_symbol_from_live_tensor_candidate():
    """Dynamic shape resolution should skip optional None tensor params."""

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

    from tilelang.jit.adapter.cutedsl.adapter import CuTeDSLKernelAdapter

    program = optional_stride_symbol_program()
    adapter = CuTeDSLKernelAdapter.__new__(CuTeDSLKernelAdapter)
    adapter.ir_module = tilelang.tvm.IRModule({program.attrs["global_symbol"]: program})
    adapter.dynamic_symbolic_map, adapter.dynamic_symbolic_order = adapter._process_dynamic_symbolic()

    dynamic_symbols = {sym.name: sym for sym in adapter.dynamic_symbolic_order}
    param_values = [torch.empty(7), None, torch.empty(7)]

    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["N"], param_values) == 7
    assert adapter._resolve_dynamic_symbolic_value(dynamic_symbols["stride"], param_values) == 0


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
