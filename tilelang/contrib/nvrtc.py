from __future__ import annotations
import cuda.bindings.nvrtc as nvrtc
from typing import Literal
from tvm_tirx.target import Target
from tvm_tirx.support.nvcc import find_cuda_path
from tilelang.env import CUDA_HOME, TILELANG_TEMPLATE_PATH
from .nvcc import get_target_compute_version, parse_compute_version


def get_nvrtc_version() -> tuple[int, int]:
    result, major, minor = nvrtc.nvrtcVersion()
    assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get NVRTC version: {result}"
    return (major, minor)


def compile_cuda(
    code: str,
    target_format: Literal["ptx", "cubin"] = "ptx",
    arch: int | None = None,
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

    arch : Optional[int]
        The cuda architecture code.

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
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "80", "90", "90a", etc.
        major, minor = parse_compute_version(get_target_compute_version(Target.current(allow_none=True)))
        arch = major * 10 + minor
    prefix = "compute" if target_format == "ptx" else "sm"
    suffix = "a" if arch >= 90 else ""
    arch_option = f"--gpu-architecture={prefix}_{arch}{suffix}"

    file_name = "tvm_kernels"
    if target_format not in ["cubin", "ptx"]:
        raise ValueError("target_format must be cubin or ptx")

    final_options = ["-default-device"]
    from tilelang.jit.adapter.nvrtc.include_paths import discover_cuda_include_paths

    final_options += [f"-I{path}" for path in discover_cuda_include_paths(CUDA_HOME or find_cuda_path())]
    final_options.append(f"-I{TILELANG_TEMPLATE_PATH}")
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

    options_bytes = [bytes(flag, "utf-8") for flag in final_options]
    compile_result = nvrtc.nvrtcCompileProgram(program, len(options_bytes), options_bytes)[0]

    if compile_result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        msg = f"{code}\nCompilation error:\n"
        result, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log size: {result}"
        log_bytes = bytes(log_size)
        result = nvrtc.nvrtcGetProgramLog(program, log_bytes)[0]
        assert result == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to get program log: {result}"
        msg += f"{log_bytes.decode('utf-8')}\n"
        msg += f"Options: {' '.join(final_options)}\n"
        nvrtc.nvrtcDestroyProgram(program)
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

    # Destroy handler
    assert nvrtc.nvrtcDestroyProgram(program)[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"Failed to destroy program: {result}"

    return result_bytes
