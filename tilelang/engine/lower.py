"""The compiler for TL programs."""

from __future__ import annotations

import re
from collections.abc import Callable
import tilelang.transform
from tilelang import tvm as tvm
from tvm import tirx
import tvm_ffi
from tvm.ir import CallingConv
from tvm.target import Target
from tilelang.contrib import hipcc, nvcc
from tilelang.env import COMPOSABLE_KERNEL_INCLUDE_DIR, CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey
from tilelang.engine.param import KernelParam, CompiledArtifact
from tilelang.engine.semantic_check import PreLowerSemanticCheck
from tilelang.backend.target import determine_target
from tilelang.backend.pass_pipeline import resolve_pipeline


def is_cpu_device_backend(target: Target):
    return target.kind.name == "c"


def has_device_kernel_launch(attrs) -> bool:
    """Check if the attributes indicate a device kernel launch."""
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def is_device_call_c_device(func: tirx.PrimFunc):
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC

    # Check if it's a C target
    if "target" in attrs and attrs["target"].kind.name == "c" and not is_cpacked:
        return True

    return has_device_kernel_launch(attrs)


def is_device_call(func: tirx.PrimFunc):
    return has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return is_device_call_c_device if is_device_c else is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


_CUDA_GLOBAL_KERNEL_PATTERN = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)')


def _collect_external_cuda_kernel_names(source: str) -> list[str]:
    kernel_names: list[str] = []
    seen_names: set[str] = set()
    for match in _CUDA_GLOBAL_KERNEL_PATTERN.finditer(source):
        kernel_name = match.group(1)
        if kernel_name not in seen_names:
            kernel_names.append(kernel_name)
            seen_names.add(kernel_name)
    return kernel_names


@tvm_ffi.register_global_func("tilelang_callback_cuda_validate", override=True)
def tilelang_callback_cuda_validate(device_mod):
    for _, base_func in device_mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc) or not base_func.attrs:
            continue

        code_block_source = base_func.attrs.get("code_block_source")
        if code_block_source is None:
            continue

        global_symbol = base_func.attrs.get("global_symbol")
        if global_symbol is None:
            raise ValueError("CodeGenTileLangCUDA expects source-kernel PrimFunc to have the global_symbol attribute")

        expected_name = str(global_symbol)
        code_block_entry_name = base_func.attrs.get("code_block_entry_name")
        if code_block_entry_name is not None and str(code_block_entry_name) != expected_name:
            raise ValueError("T.CUDASourceCodeKernel expects the lowered device global_symbol to match entry_name")

        kernel_names = _collect_external_cuda_kernel_names(str(code_block_source))
        if not kernel_names:
            raise ValueError("T.CUDASourceCodeKernel expects external CUDA source to declare at least one __global__ kernel")
        if expected_name not in kernel_names:
            raise ValueError(
                "T.CUDASourceCodeKernel expected device global_symbol "
                f"`{expected_name}` to match a __global__ kernel in the provided CUDA source. "
                f"Available entries: {', '.join(kernel_names)}"
            )


@tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
def tilelang_callback_cuda_compile(code, target, pass_config=None):
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))

    arch = [f"-arch=sm_{target_arch}"]
    compile_format = "cubin"

    # Read pass-config keys (string-valued) like in jit.adapter.libgen.compile_lib
    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))

    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    if ptxas_usage_level is not None:
        ptxas_usage_level = int(ptxas_usage_level)
    verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

    options = [
        # tl_templates/cuda/reduce.h uses explicit lambda template parameters
        # (`[&]<typename T>(T) { ... }`) which require C++20.
        "-std=c++20",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]
    # Merge extra device compiler flags from pass config, if provided
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex

        if isinstance(extra_flags, str):
            tokens = shlex.split(extra_flags)
        else:
            tokens = []
            for flag in extra_flags:
                if isinstance(flag, str):
                    tokens.extend(shlex.split(flag))
                else:
                    tokens.append(str(flag))
        options += tokens

    verbose = False
    if enable_fast_math:
        options.append("--use_fast_math")
    if ptxas_usage_level is not None:
        options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
    if verbose_ptxas_output:
        options.append("--ptxas-options=--verbose")
        options.append("-w")  # Suppress warnings to make ptxas output more readable
        verbose = True

    ptx = nvcc.compile_cuda(
        code,
        compile_format,
        arch,
        options=options,
        verbose=verbose,
    )

    return ptx


@tvm_ffi.register_global_func("tilelang_callback_hip_compile", override=True)
def tilelang_callback_hip_compile(code, target):
    from tilelang.rocm.target import target_get_mcpu

    arch = target_get_mcpu(target)
    hsaco = hipcc.compile_hip(
        code,
        target_format="hsaco",
        arch=arch,
        options=[
            "-std=c++17",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + COMPOSABLE_KERNEL_INCLUDE_DIR,
        ],
        verbose=False,
    )

    return hsaco


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
    if target is not None and target.kind.name == "metal":
        from tilelang.metal.transform import MarkHostMetalContext

        host_mod = MarkHostMetalContext()(host_mod)
    if target_host.kind.name == "llvm":
        host_mod = tvm.ffi.get_global_func("target.build.llvm")(host_mod, target_host)
    elif target_host.kind.name == "c":
        host_mod = tvm.ffi.get_global_func("target.build.tilelang_c_host")(host_mod, target_host)
    else:
        raise ValueError(f"Target host {target_host.kind.name} is not supported")
    return host_mod


def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tirx.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

    if target.kind.name == "cuda":
        global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda")
        device_mod = tvm.ffi.get_global_func(global_func)(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip")(device_mod, target)
    elif target.kind.name == "metal":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_metal")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


def device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tirx.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)

    if target.kind.name == "cuda":
        global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda") + "_without_compile"
        device_mod = tvm.ffi.get_global_func(global_func)(device_mod, target)
    elif target.kind.name == "hip":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_hip_without_compile")(device_mod, target)
    elif target.kind.name == "c":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_c")(device_mod, target)
    elif target.kind.name == "llvm":
        device_mod = tvm.ffi.get_global_func("target.build.llvm")(device_mod, target)
    elif target.kind.name == "webgpu":
        device_mod = tvm.ffi.get_global_func("target.build.webgpu")(device_mod, target)
    elif target.kind.name == "metal":
        device_mod = tvm.ffi.get_global_func("target.build.tilelang_metal")(device_mod, target)
    else:
        raise ValueError(f"Target {target.kind.name} is not supported")

    return device_mod


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

    pipeline = resolve_pipeline(target)
    mod = pipeline.lower(mod, target)

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
