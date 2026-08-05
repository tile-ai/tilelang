from __future__ import annotations

import re

from tvm import tirx

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.spec import BackendSpec, register_backend
from tilelang.contrib import nvcc
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH, env
from tilelang.transform import PassConfigKey

from . import codegen, execution_backend, pipeline

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


def tilelang_callback_cuda_compile(code, target, pass_config=None):
    from tilelang.cache.cuda_binary_cache import CUDABinaryCache

    target_arch, target_code = nvcc.get_target_arch_and_code(target)
    target_code_list = nvcc.get_target_code_list(target_code)
    gencode_code = nvcc.format_target_code_for_gencode(target_code)
    if gencode_code is None:
        arch = [f"-arch=sm_{target_arch}"]
    else:
        arch = ["-gencode", f"arch=compute_{target_arch},code={gencode_code}"]
    compile_format = "fatbin" if len(target_code_list) > 1 else "cubin"

    cfg = pass_config or {}
    enable_fast_math = bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False))
    ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
    if ptxas_usage_level is not None:
        ptxas_usage_level = int(ptxas_usage_level)

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

    verbose = env.get_default_verbose()
    if enable_fast_math:
        options.append("--use_fast_math")
    if ptxas_usage_level is not None:
        options.append(f"--ptxas-options=--register-usage-level={ptxas_usage_level}")
    if verbose:
        options.append("--ptxas-options=--verbose")
        options.append("-w")

    cache_key = CUDABinaryCache.make_key(
        code=code,
        target_kind=target.kind.name,
        target_arch=target_arch,
        target_code=target_code_list,
        compile_format=compile_format,
        options=options,
    )
    cached_binary = CUDABinaryCache.load(cache_key, compile_format)
    if cached_binary is not None:
        return bytearray(cached_binary)

    binary = nvcc.compile_cuda(code, compile_format, arch, options=options, verbose=verbose)
    CUDABinaryCache.save(cache_key, compile_format, binary)
    return binary


BACKEND = register_backend(
    BackendSpec(
        name="cuda",
        target_kinds=("cuda",),
        supports_target=codegen.is_plain_cuda_target,
        pipelines={"cuda": pipeline.CUDA_PIPELINE},
        device_codegens={
            "cuda": (
                DeviceCodegen(
                    "cuda",
                    build=codegen.build_cuda,
                    build_without_compile=codegen.build_cuda_without_compile,
                ),
            )
        },
        execution_backends=execution_backend.CUDA_EXECUTION_BACKENDS,
        callbacks={
            "tilelang_callback_cuda_validate": tilelang_callback_cuda_validate,
            "tilelang_callback_cuda_compile": tilelang_callback_cuda_compile,
        },
    )
)
