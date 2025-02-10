# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .arch_base import TileDevice
from .cuda import CUDA
from .cpu import CPU
from .cdna import CDNA
from typing import Union
from tvm.target import Target


def get_arch(target: Union[str, Target] = "cuda") -> TileDevice:
    if isinstance(target, str):
        target = Target(target)

    if target.kind.name == "cuda":
        return CUDA(target)
    elif target.kind.name == "llvm":
        return CPU(target)
    elif target.kind.name == "hip":
        return CDNA(target)
    else:
        raise ValueError(f"Unsupported target: {target.kind.name}")


def auto_infer_current_arch() -> TileDevice:
    # TODO(lei): This is a temporary solution to infer the current architecture
    # Can be replaced by a more sophisticated method in the future
    return get_arch("cuda")


from .cpu import is_cpu_arch  # noqa: F401
from .cuda import (
    is_cuda_arch,  # noqa: F401
    is_volta_arch,  # noqa: F401
    is_ampere_arch,  # noqa: F401
    is_ada_arch,  # noqa: F401
    is_hopper_arch,  # noqa: F401
    is_tensorcore_supported_precision,  # noqa: F401
    has_mma_support,  # noqa: F401
)
from .cdna import is_cdna_arch  # noqa: F401
