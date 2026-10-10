from __future__ import annotations
from functools import cache
import os.path as osp
import cuda.bindings.nvrtc as nvrtc
from typing import Literal
from tvm.target import Target
from tilelang.env import CUDA_HOME
from .cuda_include import discover_cuda_include_paths
from .nvcc import get_target_arch, get_target_arch_and_code


@cache
def get_nvrtc_version() -> tuple[int, int]:
    """Return the loaded NVRTC's version, cached for this process."""
    result, major, minor = nvrtc.nvrtcVersion()
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get NVRTC version: {result}"
    return (major, minor)


def get_compile_options(cuda_home: str | None = None) -> list[str]:
    """Return CUDA header paths and compatibility defines for NVRTC."""
    include_paths = discover_cuda_include_paths(cuda_home or CUDA_HOME or "/usr/local/cuda")
    major = get_nvrtc_version()[0]
    options = [*(f"-I{path}" for path in include_paths), f"-D__CUDACC_VER_MAJOR__={major}"]
    # CUDA <13 uses the legacy cuda/std search root. CUDA 13 exposes it
    # through include/cccl; adding the deeper root can expose private CCCL
    # headers whose tuple declarations conflict with CuTe under NVRTC.
    if major < 13:
        options += [f"-I{path}/cuda/std" for path in include_paths if not path.endswith(osp.join("include", "cccl"))]
    return options


def compile_cuda(
    code: str,
    target_format: Literal["ptx", "cubin"] = "ptx",
    arch: int | str | None = None,
    options: str | list[str] | None = None,
    verbose: bool = False,
) -> bytearray:
    """Compile cuda code with NVRTC.

    Parameters
    ----------
    code : str
        The cuda code.

    target_format : Literal["ptx", "cubin"]
        The target format of nvrtc compiler.

    arch : Optional[Union[int, str]]
        The CUDA architecture code. String tokens preserve the exact suffix,
        such as "90", "90a", or "100f". Integer values use the legacy
        architecture-specific suffix for SM90 and newer. If None, use the
        current target architecture or detect it from the GPU, as in NVCC.

    options : Optional[Union[str, List[str]]]
        The additional options.

    verbose : bool
        Whether to print the verbose output.

    Return
    ------
    result_bytes : bytearray
        The bytearray of the cubin or ptx code.
    """
    if arch is None:
        arch, _ = get_target_arch_and_code(Target.current(allow_none=True))
    elif isinstance(arch, int):
        arch = get_target_arch(divmod(arch, 10))
    prefix = "compute" if target_format == "ptx" else "sm"
    arch_option = f"--gpu-architecture={prefix}_{arch}"

    file_name = "tvm_kernels"
    if target_format not in ["cubin", "ptx"]:
        raise ValueError("target_format must be cubin or ptx")

    final_options = ["-default-device"]
    if get_nvrtc_version() >= (12, 8):
        final_options += ["-pch"]
    if arch is not None:
        final_options += [arch_option]

    if options:
        if isinstance(options, str):
            final_options += [options]
        elif isinstance(options, list):
            final_options += options
        else:
            raise ValueError("options must be str or list of str")

    code = "#include <tl_templates/cuda/nvrtc_std.h>\n" + code

    if "cudaGridDependencySynchronize" in code or "cudaTriggerProgrammaticLaunchCompletion" in code:
        code = '#include "cuda_device_runtime_api.h"\n' + code

    code_bytes = bytes(code, "utf-8")
    result, program = nvrtc.nvrtcCreateProgram(code_bytes, bytes(file_name, "utf-8"), 0, [], [])
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to create program: {result}"

    try:
        options_bytes = [bytes(flag, "utf-8") for flag in final_options]
        compile_result = nvrtc.nvrtcCompileProgram(program, len(options_bytes), options_bytes)[0]

        if compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            msg = f"{code}\nCompilation error:\n"
            if verbose:
                result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
                assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log size: {result}"
                log_bytes = bytes(log_size)
                result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
                assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log: {result}"
                msg += f"{log_bytes.decode('utf-8')}\n"
            else:
                msg += "Turn on verbose to see the full compilation log."
            msg += f"Options: {' '.join(final_options)}\n"
            raise RuntimeError(msg)

        if target_format == "cubin":
            result, cubin_size = nvrtc.nvrtcGetCUBINSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN size: {result}"
            result_bytes = bytes(cubin_size)
            result = nvrtc.nvrtcGetCUBIN(program, result_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get CUBIN: {result}"
        else:
            result, ptx_size = nvrtc.nvrtcGetPTXSize(program)
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX size: {result}"
            result_bytes = bytes(ptx_size)
            result = nvrtc.nvrtcGetPTX(program, result_bytes)[0]
            assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get PTX: {result}"
    finally:
        result = nvrtc.nvrtcDestroyProgram(program)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to destroy program: {result}"

    return result_bytes
