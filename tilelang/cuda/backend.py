from __future__ import annotations

import re

import tvm_ffi
from tvm import tirx
from tvm.target import Target

from tilelang import tvm
from tilelang.backend import Backend, ExecutionBackendSpec, register_backend
from tilelang.backend.codegen import build_device_with_global_func
from tilelang.contrib import nvcc
from tilelang.cuda.pipeline import cuda_pipeline
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey

_CUDA_GLOBAL_KERNEL_PATTERN = re.compile(r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)')
_CALLBACKS_REGISTERED = False


def _is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" in target.keys


def _is_plain_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" not in target.keys


def _default_cuda_execution_backend(target: Target) -> str:
    return "cutedsl" if _is_cutedsl_target(target) else "tvm_ffi"


def target_get_warp_size(target: Target) -> int:
    return int(target.attrs.get("thread_warp_size", 32))


def _is_nvrtc_available() -> bool:
    from tilelang.jit.adapter.nvrtc import is_nvrtc_available

    return bool(is_nvrtc_available)


def _is_cutedsl_available() -> bool:
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except ImportError:
        return False
    return True


def _collect_external_cuda_kernel_names(source: str) -> list[str]:
    kernel_names: list[str] = []
    seen_names: set[str] = set()
    for match in _CUDA_GLOBAL_KERNEL_PATTERN.finditer(source):
        kernel_name = match.group(1)
        if kernel_name not in seen_names:
            kernel_names.append(kernel_name)
            seen_names.add(kernel_name)
    return kernel_names


def _tilelang_callback_cuda_validate(device_mod):
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


def _tilelang_callback_cuda_compile(code, target, pass_config=None):
    target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))

    arch = [f"-arch=sm_{target_arch}"]
    compile_format = "cubin"

    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))

    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    if ptxas_usage_level is not None:
        ptxas_usage_level = int(ptxas_usage_level)
    verbose_ptxas_output = bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False))

    options = [
        "-std=c++20",
        "-I" + TILELANG_TEMPLATE_PATH,
        "-I" + CUTLASS_INCLUDE_DIR,
    ]
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
        options.append("-w")
        verbose = True

    return nvcc.compile_cuda(
        code,
        compile_format,
        arch,
        options=options,
        verbose=verbose,
    )


def register_cuda_callbacks() -> None:
    global _CALLBACKS_REGISTERED
    if _CALLBACKS_REGISTERED:
        return
    tvm_ffi.register_global_func("tilelang_callback_cuda_validate", f=_tilelang_callback_cuda_validate, override=True)
    tvm_ffi.register_global_func("tilelang_callback_cuda_compile", f=_tilelang_callback_cuda_compile, override=True)
    _CALLBACKS_REGISTERED = True


def cuda_device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda")
    return build_device_with_global_func(device_mod, target, global_func)


def cuda_device_codegen_without_compile(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    global_func = "target.build.tilelang_" + ("cutedsl" if "cutedsl" in target.keys else "cuda") + "_without_compile"
    return build_device_with_global_func(device_mod, target, global_func)


cuda_backend = Backend(
    name="cuda",
    target_kinds=("cuda",),
    import_path="tilelang.cuda",
    pipeline=cuda_pipeline,
    device_codegen=cuda_device_codegen,
    device_codegen_without_compile=cuda_device_codegen_without_compile,
    register_callbacks=register_cuda_callbacks,
    features={
        "warp_size": target_get_warp_size,
        "supports_tma": nvcc.have_tma,
        "supports_pdl": nvcc.have_pdl,
    },
    execution_backends={
        "tvm_ffi": ExecutionBackendSpec(
            name="tvm_ffi",
            supports_target=_is_plain_cuda_target,
            enable_host_codegen=True,
            enable_device_compile=True,
        ),
        "cython": ExecutionBackendSpec(
            name="cython",
            supports_target=_is_plain_cuda_target,
        ),
        "nvrtc": ExecutionBackendSpec(
            name="nvrtc",
            is_available=_is_nvrtc_available,
            supports_target=_is_plain_cuda_target,
        ),
        "cutedsl": ExecutionBackendSpec(
            name="cutedsl",
            is_available=_is_cutedsl_available,
            supports_target=_is_cutedsl_target,
        ),
    },
    default_execution_backend=_default_cuda_execution_backend,
    cmake_name="CUDA",
)

register_backend(cuda_backend, override=True)
