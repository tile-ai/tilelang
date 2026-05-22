from __future__ import annotations


from platform import mac_ver
from typing import Literal

import torch

from tilelang import tvm as tvm
from tilelang import language as T
from tilelang import _ffi_api
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

TargetConfig = dict[str, object]
TargetLike = str | TargetConfig | Target

SUPPORTED_TARGETS: dict[str, str] = {
    "auto": "Auto-detect CUDA/HIP/Metal based on availability.",
    "cuda": "CUDA GPU target. Use dict options such as {'kind': 'cuda', 'arch': 'sm_90'}.",
    "hip": "ROCm HIP target. Use dict options such as {'kind': 'hip', 'mcpu': 'gfx942'}.",
    "metal": "Apple Metal target for arm64 Macs.",
    "hexagon": "Qualcomm Hexagon DSP target.",
    "llvm": "LLVM CPU target. Use dict options such as {'kind': 'llvm', 'mcpu': 'native'}.",
    "webgpu": "WebGPU target for browser/WebGPU runtimes.",
    "c": "C source backend.",
    "cutedsl": "CuTe DSL GPU target. Use dict options such as {'kind': 'cutedsl', 'arch': 'sm_90'}.",
}

ROCM_MTRIPLE = "amdgcn-amd-amdhsa-hcc"


def normalize_rocm_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0]
    return normalized if normalized.startswith("gfx") else None


def target_get_mcpu(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    return normalize_rocm_arch(target.attrs.get("mcpu"))


def rocm_warp_size_for_arch(arch: str | None) -> int | None:
    if arch is None:
        return None
    if arch.startswith("gfx9"):
        return 64
    if arch.startswith(("gfx10", "gfx11", "gfx12")):
        return 32
    return None


def with_rocm_target_attrs(target: Target) -> Target:
    if target.kind.name != "hip":
        return target
    arch = target_get_mcpu(target)
    if arch is None:
        return target

    target_dict = dict(target.export())
    target_dict.setdefault("mtriple", ROCM_MTRIPLE)
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    else:
        target_dict.pop("thread_warp_size", None)
    return Target(target_dict)


def _detect_torch_rocm_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return normalize_rocm_arch(getattr(props, "gcnArchName", None))


def _rocm_target_from_arch(arch: str | None) -> Target | str:
    if arch is None:
        return "hip"
    target_dict = {
        "kind": "hip",
        "mcpu": arch,
        "mtriple": ROCM_MTRIPLE,
    }
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    return Target(target_dict)


def _detect_torch_cuda_arch() -> str | None:
    """Return the CUDA SM architecture detected from PyTorch, if available."""
    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(0)
    return f"sm_{nvcc.get_target_arch(cap)}" if cap else None


def _cuda_target_from_arch(arch: str | None) -> Target | str:
    """Build a CUDA target while preserving the legacy bare string fallback."""
    if arch is None:
        return "cuda"
    return Target({"kind": "cuda", "arch": arch})


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


def _with_cutedsl_key(target: Target | str) -> Target:
    if not isinstance(target, Target):
        target = Target(target)
    target_dict = dict(target.export())
    target_dict["keys"] = list(dict.fromkeys([*target_dict.get("keys", ()), "cutedsl"]))
    return Target(target_dict)


def normalize_cutedsl_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "cuda" and "cutedsl" in target.keys:
            return target
        return None

    if isinstance(target, dict):
        if target.get("kind") == "cutedsl":
            cuda_target = dict(target)
            cuda_target["kind"] = "cuda"
            try:
                return _with_cutedsl_key(Target(cuda_target))
            except Exception:
                return None
        try:
            temp_target = Target(target)
        except Exception:
            return None
        if temp_target.kind.name == "cuda" and "cutedsl" in temp_target.keys:
            return temp_target
        return None

    if target.strip() == "cutedsl":
        try:
            return _with_cutedsl_key(_cuda_target_from_arch(_detect_torch_cuda_arch()))
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


def determine_target(target: TargetLike | Literal["auto"] = "auto", return_object: bool = False) -> str | TargetConfig | Target:
    """
    Determine the appropriate target for compilation (CUDA, HIP, Hexagon, or manual selection).

    Args:
        target (Union[str, dict, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will automatically detect available accelerators.
            - If a string, dict, or Target, it is directly validated.

    Returns:
        Union[str, dict, Target]: The selected target ("cuda", "hip", a config dict, or a Target object).

    Raises:
        ValueError: If no compatible accelerator is found and the target is "auto".
        AssertionError: If the target is invalid.
    """

    return_var: str | TargetConfig | Target = target

    return_var: str | TargetConfig | Target = target

    if target == "auto":
        target = tvm.target.Target.current(allow_none=True)
        if target is not None:
            return with_rocm_target_attrs(target)
        # ROCm PyTorch exposes devices through torch.cuda. If CUDA tooling is
        # also present, prefer HIP so APUs such as gfx1151 are not misread as
        # CUDA architectures like sm_115a.
        if torch.version.hip is not None and check_hip_availability():
            return_var = _rocm_target_from_arch(_detect_torch_rocm_arch())
        else:
            # Check for CUDA and HIP availability
            is_cuda_available = check_cuda_availability()
            is_hip_available = check_hip_availability()

            # Determine the target based on availability
            if is_cuda_available:
                return_var = _cuda_target_from_arch(_detect_torch_cuda_arch())
            elif is_hip_available:
                return_var = _rocm_target_from_arch(_detect_torch_rocm_arch())
            elif check_metal_availability():
                return_var = "metal"
            elif check_hexagon_availability():
                return_var = {"kind": "llvm", "mtriple": "hexagon", "mcpu": "hexagonv73"}
            else:
                raise ValueError("No CUDA, HIP, MPS, or Hexagon available on this system.")
    else:
        return_var = target

    # Handle Backend-Specific Normalization (Shorthands)
    if isinstance(return_var, str) and "hexagon" in return_var.lower():
        return_var = {"kind": "llvm", "mtriple": "hexagon", "mcpu": "hexagonv73"}

    elif isinstance(return_var, dict) and return_var.get("mtriple") == "hexagon":
        return_var.setdefault("kind", "llvm")
        return_var.setdefault("mcpu", "hexagonv73")

    # Handle CuTeDSL special case
    possible_cutedsl_target = normalize_cutedsl_target(return_var)
    if possible_cutedsl_target is not None:
        return_var = possible_cutedsl_target
    else:
        # Validate the target if it's not "auto"
        if isinstance(target, Target):
            return_var = with_rocm_target_attrs(target)
        elif isinstance(target, dict):
            try:
                parsed_target = Target(target)
            except Exception as err:
                raise AssertionError(
                    f"Target {target} is not supported. Pass a valid target config dict, e.g. `{{'kind': 'cuda', 'arch': 'sm_80'}}`."
                ) from err
            if parsed_target.kind.name == "hip" and target_get_mcpu(parsed_target) is not None:
                return_var = with_rocm_target_attrs(parsed_target)
            else:
                return_var = target
        elif isinstance(target, str):
            normalized_target = target.strip()
            if not normalized_target:
                raise AssertionError(f"Target {target} is not supported")
            try:
                parsed_target = Target(normalized_target)
            except Exception as err:
                examples = ", ".join(f"`{name}`" for name in SUPPORTED_TARGETS)
                raise AssertionError(
                    f"Target {target} is not supported. Supported targets include: {examples}. "
                    "Pass target options as a dict, e.g. `{'kind': 'cuda', 'arch': 'sm_80'}`."
                ) from err
            if parsed_target.kind.name == "hip" and target_get_mcpu(parsed_target) is not None:
                return_var = with_rocm_target_attrs(parsed_target)
            else:
                return_var = normalized_target
        else:
            raise AssertionError(f"Target {target} is not supported")

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


def target_is_rdna(target: Target) -> bool:
    return _ffi_api.TargetIsRDNA(target)


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


def target_get_rdna_generation(target: Target) -> int:
    return _ffi_api.TargetGetRDNAGeneration(target)
