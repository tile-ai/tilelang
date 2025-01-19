# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Library information. This is a standalone file that can be used to get various info. 
Modified from flashinfer
"""

import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags
from tilelang.env import (
    CUTLASS_INCLUDE_DIR,  # noqa: F401
    TILELANG_TEMPLATE_PATH,  # noqa: F401
)


def _initialize_torch_cuda_arch_flags():
    import os
    from tilelang.contrib import nvcc
    from tilelang.utils.target import determine_target

    target = determine_target(return_object=True)
    # create tmp source file for torch cpp extension
    compute_version = "".join(nvcc.get_target_compute_version(target).split("."))
    # set TORCH_CUDA_ARCH_LIST
    major = compute_version[0]
    minor = compute_version[1]

    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _get_workspace_dir_name() -> pathlib.Path:
    try:
        with warnings.catch_warnings():
            # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
            warnings.filterwarnings("ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch")
            flags = _get_cuda_arch_flags()
        arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
    except Exception:
        arch = "noarch"
    # e.g.: $HOME/.cache/tilelang/75_80_89_90/
    return pathlib.Path.home() / ".cache" / "tilelang" / arch


# use pathlib
_initialize_torch_cuda_arch_flags()
TILELANG_JIT_WORKSPACE_DIR = _get_workspace_dir_name()
TILELANG_JIT_DIR = TILELANG_JIT_WORKSPACE_DIR / "cached_ops"
TILELANG_GEN_SRC_DIR = TILELANG_JIT_WORKSPACE_DIR / "generated"
