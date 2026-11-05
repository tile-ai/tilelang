"""The compiler for TL programs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import tilelang.transform
import tvm_tirx
from tvm_tirx import tirx
from tvm_tirx.ir import CallingConv
from tvm_tirx.target import Target
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.engine.semantic_check import PreLowerSemanticCheck
from tilelang.backend.module import BackendContext, create_backend_context
from tilelang.instrumentation import (
    compile_pass_instrumentation,
    current_compile_pass_instrumentation,
    instrument_current_pass_context,
)


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tirx.Function):
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c" and not is_cpacked:
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tirx.Function):
    # Native CPU split functions use ordinary C calls, so the launch calling
    # convention alone cannot classify them. SplitHostDevice marks every
    # externally callable compute function, including LLVM and C kernels.
    return has_device_kernel_launch(func.attrs) or bool(func.attrs.get("tirx.is_global_func", False))


def get_device_call(is_device_c: bool = False) -> Callable[[tirx.Function], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tirx.Function], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


def extrac_params(func: tirx.Function) -> list[KernelParam]:
    """Read the original public signature before native ABI lowering."""
    return [KernelParam.from_var(parameter) for parameter in func.params]


def _make_parameter_alias_contract_explicit(mod: tvm_tirx.IRModule) -> tvm_tirx.IRModule:
    """Expose the kernel ABI's alias default before dependence analysis.

    TileLang's previous SplitHostDevice supplied noalias=True to device
    functions. Earlier passes need the same contract to distinguish independent
    parameters. An explicit function value wins over the pass-context default;
    non-restrict annotations remain operands for the analyses that consume them.
    Standalone native passes receive no new alias assumption.
    """
    default = bool(tilelang.transform.get_pass_context().config.get("tirx.noalias", True))
    result = mod.clone()
    for global_var, function in mod.functions.items():
        if isinstance(function, tirx.Function) and "tirx.noalias" not in function.attrs:
            result[global_var] = function.with_attr("tirx.noalias", default)
    return result


def host_codegen(
    host_mod: tvm_tirx.IRModule,
    context: BackendContext,
) -> tvm_tirx.IRModule:
    """Generate host-side code from the lowered IR module.

    Parameters
    ----------
    host_mod : tvm_tirx.IRModule
        The host-side IR module to compile.
    context : BackendContext
        Resolved backend, device target, host target, and execution policy.
    """
    target_host = context.target_host
    host_mod = tirx.transform.BindTarget(target_host)(host_mod)
    # Packed wrappers retain the original public dtype checks. Legalize their
    # host storage only after splitting; device code generators own the native
    # BF16/FP8 instruction types and must not inherit a host storage rewrite.
    host_mod = tirx.transform.FP8StorageLegalize()(host_mod)
    host_mod = tirx.transform.BF16StorageLegalize()(host_mod)
    host_mod = tilelang.transform.PrepareAnyList()(host_mod)
    host_mod = tirx.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tilelang.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tilelang.transform.LowerIntrin()(host_mod)
    host_mod = context.preprocess_host_codegen(host_mod)
    return context.codegen_host(host_mod)


def _prepare_device_codegen_mod(device_mod: tvm_tirx.IRModule) -> tvm_tirx.IRModule:
    # TileLang's rule maps take precedence over the shared target/default
    # maps. Applying the generic finalizer first irreversibly expands calls
    # such as Ascend rsqrt before the owning emitter can select its domain.
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tilelang.transform.Simplify()(device_mod)
    return tilelang.transform.HoistBroadcastValues()(device_mod)


def device_codegen(device_mod: tvm_tirx.IRModule, context: BackendContext) -> tvm_tirx.IRModule:
    device_mod = _prepare_device_codegen_mod(device_mod)
    return context.codegen_device(device_mod, compile_device=True)


def device_codegen_without_compile(device_mod: tvm_tirx.IRModule, context: BackendContext) -> tvm_tirx.IRModule:
    device_mod = _prepare_device_codegen_mod(device_mod)
    return context.codegen_device(device_mod, compile_device=False)


def lower_to_host_device_ir(
    func_or_mod: tirx.Function | tvm_tirx.IRModule,
    context: BackendContext,
    runtime_only: bool = False,
    *,
    enable_host_codegen: bool = True,
) -> tuple[tvm_tirx.IRModule, tvm_tirx.IRModule, list[KernelParam] | None, Target, Target]:
    """Lower to split IR under a per-compilation instrumentation session.

    Direct callers receive a packable host module by default, including
    grouped compilation which generates host code after this call returns.
    A resolved NVRTC source launch can explicitly forgo that unused wrapper.
    """

    has_session = current_compile_pass_instrumentation() is not None
    with compile_pass_instrumentation(name="lower-to-host-device-ir"):
        attach_instruments = nullcontext() if has_session else instrument_current_pass_context()
        with attach_instruments:
            mod = func_or_mod
            params = None
            if isinstance(func_or_mod, tirx.Function):
                func = func_or_mod
                params = extrac_params(func) if not runtime_only else None
                mod = tvm_tirx.IRModule({func.attrs["global_symbol"]: func})

            target = context.target
            target_host = context.target_host

            _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
            _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

            mod = _make_parameter_alias_contract_explicit(mod)
            # Run backend-independent semantic checks before target-specific lowering.
            PreLowerSemanticCheck(mod)

            # This is an execution decision, not a property accepted from a
            # caller's module. Native host codegen still requires packed IR,
            # even when the context's default execution backend is NVRTC.
            mod = mod.with_attr(
                "tl.source_host_only",
                context.execution_backend.name == "nvrtc" and not enable_host_codegen,
            )
            mod = context.lower(mod)

            host_mod = tirx.transform.Filter(_is_host_call)(mod)
            device_mod = tirx.transform.Filter(_is_device_call)(mod)

            return host_mod, device_mod, params, target, target_host


def lower_with_context(
    func_or_mod: tirx.Function | tvm_tirx.IRModule,
    context: BackendContext,
    *,
    runtime_only: bool = False,
    enable_host_codegen: bool = False,
    enable_device_compile: bool = False,
) -> CompiledArtifact:
    """Compile using an existing context without resolving backend state again.

    ``enable_host_codegen`` builds the host runtime module when true. It is
    disabled by default because JIT adapters provide their own host-side
    integration. ``enable_device_compile`` similarly controls whether device
    code is compiled at this stage; JIT adapters compile it separately by
    default.
    """

    has_session = current_compile_pass_instrumentation() is not None
    with compile_pass_instrumentation(name="lower-with-context"):
        attach_instruments = nullcontext() if has_session else instrument_current_pass_context()
        with attach_instruments:
            return _lower_with_context_impl(
                func_or_mod,
                context,
                runtime_only=runtime_only,
                enable_host_codegen=enable_host_codegen,
                enable_device_compile=enable_device_compile,
            )


def _lower_with_context_impl(
    func_or_mod: tirx.Function | tvm_tirx.IRModule,
    context: BackendContext,
    *,
    runtime_only: bool,
    enable_host_codegen: bool,
    enable_device_compile: bool,
) -> CompiledArtifact:
    """Implementation executed inside the caller's analyzer context."""

    host_mod, device_mod, params, target, target_host = lower_to_host_device_ir(
        func_or_mod,
        context,
        runtime_only=runtime_only,
        enable_host_codegen=enable_host_codegen,
    )

    codegen_mod = device_codegen(device_mod, context) if enable_device_compile else device_codegen_without_compile(device_mod, context)
    kernel_source = codegen_mod.inspect_source()

    if enable_host_codegen:
        host_runtime = host_codegen(host_mod, context)
        host_runtime.import_module(codegen_mod)
        if host_mod.attrs and "tl.output_module_schema" in host_mod.attrs:
            import tvm_ffi

            host_runtime = tvm_ffi.get_global_func("tl.runtime.MakeOutputModule")(host_runtime, host_mod.attrs["tl.output_module_schema"])
        return CompiledArtifact(
            host_mod,
            device_mod,
            params,
            kernel_source,
            rt_mod=host_runtime,
            target=target,
            target_host=target_host,
        )

    return CompiledArtifact(
        host_mod,
        device_mod,
        params,
        kernel_source,
        target=target,
        target_host=target_host,
    )


def lower(
    func_or_mod: tirx.Function | tvm_tirx.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only: bool = False,
    enable_host_codegen: bool = False,
    enable_device_compile: bool = False,
) -> CompiledArtifact:
    """Compile after resolving exactly one backend context.

    ``enable_host_codegen`` builds the host runtime module when true. It is
    disabled by default because JIT adapters provide their own host-side
    integration. ``enable_device_compile`` similarly controls whether device
    code is compiled at this stage; JIT adapters compile it separately by
    default.
    """

    has_session = current_compile_pass_instrumentation() is not None
    with compile_pass_instrumentation(name="lower"):
        attach_instruments = nullcontext() if has_session else instrument_current_pass_context()
        with attach_instruments:
            context = create_backend_context(target, target_host, "auto")
            return lower_with_context(
                func_or_mod,
                context,
                runtime_only=runtime_only,
                enable_host_codegen=enable_host_codegen,
                enable_device_compile=enable_device_compile,
            )
