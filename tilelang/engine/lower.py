"""The compiler for TL programs."""

from __future__ import annotations

from collections.abc import Callable
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tirx
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.engine.semantic_check import PreLowerSemanticCheck
from tilelang.utils.target import determine_target
from tilelang.backend import resolve_backend


def is_cpu_device_backend(target: Target):
    return target.kind.name in {"c", "llvm"}


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tirx.PrimFunc):
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC

    # CPU targets keep callable PrimFuncs in the device module for source/codegen.
    if "target" in attrs and attrs["target"].kind.name in {"c", "llvm"} and not is_cpacked:
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tirx.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


def extrac_params(func: tirx.PrimFunc) -> list[KernelParam]:
    tensor_types = []
    for var in func.params:
        if var in func.buffer_map:
            tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
        else:
            tensor_types.append(KernelParam.from_var(var))
    return tensor_types


def canon_target_host(target: str | Target, target_host: str | Target | None):
    if not target_host:
        target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"

    return target_host


def host_codegen(host_mod: tvm.IRModule, target_host: Target, target: Target | None = None) -> tvm.IRModule:
    """Generate host-side code from the lowered IR module.

    Parameters
    ----------
    host_mod : tvm.IRModule
        The host-side IR module to compile.
    target_host : Target
        The host compilation target (e.g. "llvm" or "c").
    target : Target, optional
        The device target.  When the device target is Metal, the pass
        MarkHostMetalContext is applied so that the generated host code
        contains the Metal/MPS synchronisation logic.
    """
    host_mod = tirx.transform.BindTarget(target_host)(host_mod)
    host_mod = tirx.transform.FP8StorageLegalize()(host_mod)
    host_mod = tirx.transform.BF16StorageLegalize()(host_mod)
    host_mod = tirx.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tirx.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tilelang.transform.LowerIntrin()(host_mod)
    combine_context_call = getattr(tirx.transform, "CombineContextCall", None)
    if combine_context_call is not None:
        host_mod = combine_context_call()(host_mod)
    if target is not None:
        host_mod = resolve_backend(target).preprocess_host_codegen(host_mod, target)
    if target_host.kind.name == "llvm":
        host_mod = tvm.ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm.ffi.get_global_func("target.build.tilelang_c_host")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return resolve_backend(target).codegen(device_mod, target, compile=True)


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    return resolve_backend(target).codegen(device_mod, target, compile=False)


def lower_to_host_device_ir(
    func_or_mod: tirx.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only: bool = False,
) -> tuple[tvm.IRModule, tvm.IRModule, list[KernelParam] | None, Target, Target]:
    """Lower input TIR to split host/device IRModules without backend codegen."""

    mod = func_or_mod
    params = None
    if isinstance(func_or_mod, tirx.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})

    if isinstance(target, str):
        target = determine_target(target)

    target_host = canon_target_host(target, target_host)

    target_host = tvm.target.Target(target_host)
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Run backend-independent semantic checks before target-specific lowering.
    PreLowerSemanticCheck(mod)

    backend = resolve_backend(target)
    mod = backend.lower(mod, target)

    host_mod = tirx.transform.Filter(_is_host_call)(mod)
    device_mod = tirx.transform.Filter(_is_device_call)(mod)

    return host_mod, device_mod, params, target, target_host


def lower(
    func_or_mod: tirx.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
    """
    enable_host_codegen: whether to enable host codegen, default is False, as we have our
    own host codegen implementation in jit.
    enable_device_compile: whether to enable device codegen, default is False, as we have our
    own device codegen implementation in jit.
    """

    host_mod, device_mod, params, target, target_host = lower_to_host_device_ir(
        func_or_mod=func_or_mod,
        target=target,
        target_host=target_host,
        runtime_only=runtime_only,
    )

    codegen_mod = device_codegen(device_mod, target) if enable_device_compile else device_codegen_without_compile(device_mod, target)
    kernel_source = codegen_mod.inspect_source()

    if enable_host_codegen:
        host_mod = host_codegen(host_mod, target_host, target=target)
        host_mod.import_module(codegen_mod)
        return CompiledArtifact(host_mod, device_mod, params, kernel_source, rt_mod=host_mod)

    return CompiledArtifact(host_mod, device_mod, params, kernel_source)
