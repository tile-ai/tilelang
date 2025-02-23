# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Literal, Union
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.contrib import rocm
from tilelang.contrib import nvcc

AVALIABLE_TARGETS = {
    "auto",
    "cuda",
    "hip",
    "webgpu",
    "c",  # represent c source backend
    "llvm",
}


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
        assert isinstance(
            target, Target) or target in AVALIABLE_TARGETS, f"Target {target} is not supported"
        return_var = target

    if return_object:
        return Target(return_var)

    return return_var
