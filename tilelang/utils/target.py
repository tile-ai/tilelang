from typing import Literal, Union
from tilelang import tvm as tvm
from tilelang import _ffi_api
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

TENSTORRENT_TARGET = "tenstorrent"


def _is_tenstorrent_backend_enabled() -> bool:
    """Detect whether the Tenstorrent backend has been registered with TVM."""
    return bool(tvm.get_global_func("target.build.tilelang_tt", allow_missing=True))


AVALIABLE_TARGETS = {
    "auto",
    "cuda",
    "hip",
    "webgpu",
    "c",  # represent c source backend
    "llvm",
    TENSTORRENT_TARGET,
}

_HAS_TENSTORRENT_BACKEND = _is_tenstorrent_backend_enabled()


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


def determine_target(target: Union[str, Target, Literal["auto"]] = "auto",
                     return_object: bool = False) -> Union[str, Target]:
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

    return_var: Union[str, Target] = target

    if target == "auto":
        target = tvm.target.Target.current(allow_none=True)
        if target is not None:
            return target
        # Check for CUDA and HIP availability
        is_cuda_available = check_cuda_availability()
        is_hip_available = check_hip_availability()

        # Determine the target based on availability
        if is_cuda_available:
            return_var = "cuda"
        elif is_hip_available:
            return_var = "hip"
        else:
            raise ValueError("No CUDA or HIP available on this system.")
    else:
        # Validate the target if it's not "auto"
        target_kind = target.kind.name if isinstance(target, Target) else target
        assert isinstance(
            target, Target) or target_kind in AVALIABLE_TARGETS, f"Target {target} is not supported"
        if target_kind == TENSTORRENT_TARGET and not _HAS_TENSTORRENT_BACKEND:
            raise ValueError(
                "Tenstorrent backend requires TileLang to be built with TL_TT_BACKEND enabled.")
        return_var = target

    if return_object:
        return Target(return_var)
    return return_var


def target_is_cuda(target: Target) -> bool:
    return _ffi_api.TargetIsCuda(target)


def target_is_hip(target: Target) -> bool:
    return _ffi_api.TargetIsRocm(target)


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
