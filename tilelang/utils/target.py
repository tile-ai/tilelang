from __future__ import annotations


import torch

from platform import mac_ver
from typing import Literal
from tilelang import tvm as tvm
from tilelang import _ffi_api
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

SUPPORTED_TARGETS: dict[str, str] = {
    "auto": "Auto-detect CUDA/HIP/Metal based on availability.",
    "cuda": "CUDA GPU target (supports options such as `cuda -arch=sm_80`).",
    "hip": "ROCm HIP target (supports options like `hip -mcpu=gfx90a`).",
    "metal": "Apple Metal target for arm64 Macs.",
    "llvm": "LLVM CPU target (accepts standard TVM LLVM options).",
    "webgpu": "WebGPU target for browser/WebGPU runtimes.",
    "c": "C source backend.",
    "cutedsl": "CuTe DSL GPU target.",
}


def describe_supported_targets() -> dict[str, str]:
    """
    Return a mapping of supported target names to usage descriptions.
    """
    return dict(SUPPORTED_TARGETS)


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system by locating the CUDA path.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        nvcc.find_cuda_path()
        return True
    except Exception:
        return False


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def check_metal_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if not mac_release:
        return False
    # todo: check torch version?
    return arch == "arm64"


def determine_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> str:
    """
    Select the correct FP8 dtype string for the current platform.
    - CUDA defaults to FP8 E4M3FN / E5M2.
    - ROCm uses FNUZ except gfx950 (OCP), which prefers non-FNUZ when available.
    """
    if fp8_format not in {"e4m3", "e5m2"}:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")
    if torch.version.hip is None:
        return "float8_e4m3fn" if fp8_format == "e4m3" else "float8_e5m2"
    if not torch.cuda.is_available():
        return "float8_e4m3fnuz" if fp8_format == "e4m3" else "float8_e5m2fnuz"
    props = torch.cuda.get_device_properties(0)
    gcn_arch = getattr(props, "gcnArchName", "")
    if fp8_format == "e4m3":
        if gcn_arch.startswith("gfx950"):
            return "float8_e4m3fn"
        return "float8_e4m3fnuz"
    if gcn_arch.startswith("gfx950") and hasattr(torch, "float8_e5m2"):
        return "float8_e5m2"
    return "float8_e5m2fnuz"


def determine_torch_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> torch.dtype:
    dtype_name = determine_fp8_type(fp8_format)
    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        raise RuntimeError(f"PyTorch does not expose dtype {dtype_name}")
    return torch_dtype


def normalize_cutedsl_target(target: str | Target) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "cuda" and "cutedsl" in target.keys:
            return target
        return None

    if target.startswith("cutedsl"):
        cuda_target_str = target.replace("cutedsl", "cuda", 1)

        try:
            temp_target = Target(cuda_target_str)

            target_dict = dict(temp_target.export())
            target_dict["keys"] = list(set(target_dict["keys"]) | {"cutedsl"})

            return Target(target_dict)
        except Exception:
            return None

    return None


def determine_target(target: str | Target | Literal["auto"] = "auto", return_object: bool = False) -> str | Target:
    """
    Determine the appropriate target for compilation (CUDA, HIP, or manual selection).

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will automatically detect whether CUDA or HIP is available.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target ("cuda", "hip", or a valid Target object).

    Raises:
        ValueError: If no CUDA or HIP is available and the target is "auto".
        AssertionError: If the target is invalid.
    """

    return_var: str | Target = target

    if target == "auto":
        target = tvm.target.Target.current(allow_none=True)
        if target is not None:
            return target
        # Check for CUDA and HIP availability
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()

        # Determine the target based on availability
        if is_cuda_available:
            if torch.cuda.is_available() and (cap := torch.cuda.get_device_capability(0)):
                return_var = Target({"kind": "cuda", "arch": f"sm_{nvcc.get_target_arch(cap)}"})
            else:
                return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        elif check_metal_availability():
            return_var = "metal"
        else:
            raise ValueError("No CUDA or HIP or MPS available on this system.")

    else:
        possible_cutedsl_target = normalize_cutedsl_target(target)
        if possible_cutedsl_target is not None:
            try:
                from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available  # lazy

                check_cutedsl_available()
            except ImportError as e:
                raise AssertionError(f"CuTeDSL backend is not available. Please install tilelang-cutedsl package. {str(e)}") from e

            return_var = possible_cutedsl_target
        else:
            # Validate the target if it's not "auto"
            if isinstance(target, Target):
                return_var = target
            elif isinstance(target, str):
                normalized_target = target.strip()
                if not normalized_target:
                    raise AssertionError(f"Target {target} is not supported")
                try:
                    Target(normalized_target)
                except Exception as err:
                    examples = ", ".join(f"`{name}`" for name in SUPPORTED_TARGETS)
                    raise AssertionError(
                        f"Target {target} is not supported. Supported targets include: {examples}. "
                        "Pass additional options after the base name, e.g. `cuda -arch=sm_80`."
                    ) from err
                return_var = normalized_target
            else:
                raise AssertionError(f"Target {target} is not supported")

    if isinstance(return_var, Target):
        return return_var
    if return_object:
        if isinstance(return_var, Target):
            return return_var
        return Target(return_var)
    return return_var


def target_is_cuda(target: Target) -> bool:
    return _ffi_api.TargetIsCuda(target)


def target_is_hip(target: Target) -> bool:
    return _ffi_api.TargetIsRocm(target)


def target_is_metal(target: Target) -> bool:
    return _ffi_api.TargetIsMetal(target)


def target_is_volta(target: Target) -> bool:
    return _ffi_api.TargetIsVolta(target)


def target_is_turing(target: Target) -> bool:
    return _ffi_api.TargetIsTuring(target)


def target_is_ampere(target: Target) -> bool:
    return _ffi_api.TargetIsAmpere(target)


def target_is_hopper(target: Target) -> bool:
    return _ffi_api.TargetIsHopper(target)


def target_is_sm120(target: Target) -> bool:
    return _ffi_api.TargetIsSM120(target)


def target_is_cdna(target: Target) -> bool:
    return _ffi_api.TargetIsCDNA(target)


def target_has_async_copy(target: Target) -> bool:
    return _ffi_api.TargetHasAsyncCopy(target)


def target_has_ldmatrix(target: Target) -> bool:
    return _ffi_api.TargetHasLdmatrix(target)


def target_has_stmatrix(target: Target) -> bool:
    return _ffi_api.TargetHasStmatrix(target)


def target_has_bulk_copy(target: Target) -> bool:
    return _ffi_api.TargetHasBulkCopy(target)


def target_get_warp_size(target: Target) -> int:
    return _ffi_api.TargetGetWarpSize(target)
