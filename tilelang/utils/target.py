from __future__ import annotations


import torch

from platform import mac_ver
from typing import Literal
from tilelang import tvm as tvm
from tilelang import language as T
from tilelang import _ffi_api
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

SUPPORTED_TARGETS: dict[str, str] = {
    "auto": "Auto-detect CUDA/HIP/Metal based on availability.",
    "cuda": "CUDA GPU target (supports options such as `cuda -arch=sm_80`).",
    "hip": "ROCm HIP target (supports options like `hip -mcpu=gfx90a`).",
    "metal": "Apple Metal target for arm64 Macs.",
    "hexagon": "Qualcomm Hexagon DSP target.",
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


def check_hexagon_availability() -> bool:
    """
    Check if Hexagon support is available in the TVM build.
    Returns:
        bool: True if Hexagon is available, False otherwise.
    """
    try:
        # Check if Hexagon runtime is enabled in TVM
        return tvm.runtime.enabled("hexagon")
    except Exception:
        return False


def determine_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> str:
    """
    Select the correct FP8 dtype string for the current platform.
    - CUDA defaults to FP8 E4M3FN / E5M2.
    - ROCm uses FNUZ except gfx950 (OCP), which prefers non-FNUZ when available.
    """
    if fp8_format not in {"e4m3", "e5m2"}:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")
    if torch.version.hip is None:
        return T.float8_e4m3fn if fp8_format == "e4m3" else T.float8_e5m2
    if not torch.cuda.is_available():
        return T.float8_e4m3fnuz if fp8_format == "e4m3" else T.float8_e5m2fnuz
    props = torch.cuda.get_device_properties(0)
    gcn_arch = getattr(props, "gcnArchName", "")
    if fp8_format == "e4m3":
        if gcn_arch.startswith("gfx950"):
            return T.float8_e4m3fn
        return T.float8_e4m3fnuz
    if gcn_arch.startswith("gfx950") and hasattr(T, "float8_e5m2"):
        return T.float8_e5m2
    return T.float8_e5m2fnuz


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


def is_hexagon_target(target) -> bool:
    """Return True for LLVM targets cross-compiled to Hexagon."""
    if target is None:
        return False
    # Check if it's a TVM Target object
    if hasattr(target, "kind"):
        if target.kind.name == "hexagon":
            return True
        if target.kind.name == "llvm":
            return "hexagon" in str(target.attrs.get("mtriple", "")).lower()
    # Check if it's a string
    return "hexagon" in str(target).lower()


def determine_target(target: str | Target | Literal["auto"] = "auto", return_object: bool = False) -> str | Target:
    """
    Determine the appropriate target for compilation (CUDA, HIP, Hexagon, or manual selection).

    Args:
        target (Union[str, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will automatically detect available accelerators.
            - If a string or Target, it is directly validated.

    Returns:
        Union[str, Target]: The selected target string or Target object.

    Raises:
        ValueError: If no compatible accelerator is found and the target is "auto".
        AssertionError: If the target is invalid.
    """
    return_var: str | Target = target

    if target == "auto":
        curr = tvm.target.Target.current(allow_none=True)
        if curr is not None:
            return curr

        # Check for accelerator availability in order of preference
        if check_cuda_availability():
            if torch.cuda.is_available() and (cap := torch.cuda.get_device_capability(0)):
                return_var = Target({"kind": "cuda", "arch": f"sm_{nvcc.get_target_arch(cap)}"})
            else:
                return_var = "cuda"
        elif check_hip_availability():
            return_var = "hip"
        elif check_metal_availability():
            return_var = "metal"
        elif check_hexagon_availability():
            return_var = "llvm -mtriple=hexagon -mcpu=hexagonv73"
        else:
            raise ValueError("No compatible accelerator found (CUDA/HIP/Metal/Hexagon).")
    else:
        return_var = target

    # Handle Backend-Specific Normalization (Shorthands)
    if isinstance(return_var, str) and "hexagon" in return_var.lower() and "-mtriple" not in return_var:
        return_var = "llvm -mtriple=hexagon -mcpu=hexagonv73"

    # Handle CuTeDSL special case
    possible_cutedsl = normalize_cutedsl_target(return_var)
    if possible_cutedsl is not None:
        return_var = possible_cutedsl

    # Final Validation and conversion to object if requested
    if return_object:
        return return_var if isinstance(return_var, Target) else Target(return_var)
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


def target_is_gfx950(target: Target) -> bool:
    return _ffi_api.TargetIsGfx950(target)


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
