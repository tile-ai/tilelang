from __future__ import annotations

from .arch_base import TileDevice
from .cuda import *
from .cpu import *
from .cdna import *
from .rdna import *
from .metal import *
from tvm.target import Target
import torch
from tilelang.utils.target import determine_target, target_get_rdna_generation, target_is_rdna


def get_arch(target: str | Target = "cuda") -> TileDevice:
    if isinstance(target, str):
        target = Target(target)

    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        return CPU(target)
    elif target.kind.name == "hip":
        if target_is_rdna(target):
            if target_get_rdna_generation(target) == 11:
                return RDNA(target)
            raise ValueError(f"RDNA device model currently supports gfx11 targets only, got {target}.")
        return CDNA(target)
    elif target.kind.name == "metal":
        return METAL(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")


def auto_infer_current_arch() -> TileDevice:
    # TODO(lei): This is a temporary solution to infer the current architecture
    # Can be replaced by a more sophisticated method in the future
    if torch.version.hip is not None:
        return get_arch(determine_target("auto", return_object=True))
    if torch.cuda.is_available():
        return get_arch("cuda")
    elif torch.mps.is_available():
        return get_arch("metal")
    else:
        return get_arch("llvm")


__all__ = [
    "is_cpu_arch",
    "is_cuda_arch",
    "is_volta_arch",
    "is_ampere_arch",
    "is_ada_arch",
    "is_hopper_arch",
    "is_tensorcore_supported_precision",
    "has_mma_support",
    "is_cdna_arch",
    "is_rdna_arch",
    "is_metal_arch",
    "CUDA",
    "CDNA",
    "RDNA",
    "METAL",
    "CPU",
]
