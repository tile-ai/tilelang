# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import os
import torch
import warnings
from torch.utils.cpp_extension import load, _import_module_from_library
from tilelang.env import TILELANG_CACHE_DIR, TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR

# Define paths
compress_util = os.path.join(TILELANG_TEMPLATE_PATH, "tl_templates/cuda/compress_sm90.cu")
# Cache directory for compiled extensions
_CACHE_DIR = os.path.join(TILELANG_CACHE_DIR, "sparse_compressor")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _get_cached_lib():
    name = 'compress_lib'
    cached_path = os.path.join(_CACHE_DIR, f"{name}.so")

    if os.path.exists(cached_path):
        try:
            return _import_module_from_library(name, cached_path)
        except Exception:
            # If loading fails, recompile
            pass

    from tilelang.env import _initialize_torch_cuda_arch_flags
    # Set TORCH_CUDA_ARCH_LIST
    _initialize_torch_cuda_arch_flags()

    # Compile if not cached or loading failed
    return load(
        name=name,
        sources=[compress_util],
        extra_cuda_cflags=[
            '-O2',
            '-std=c++17',
            '-lineinfo',
            f'-I{CUTLASS_INCLUDE_DIR}',
            f'-I{CUTLASS_INCLUDE_DIR}/../tools/util/include',
            '-arch=sm_90',
        ],
        build_directory=_CACHE_DIR,
    )


def compress_sm90(A: torch.Tensor, block_k: int,
                  transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if block_k > 128:
        block_k = 128
        # Ref: https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L145-L146
        warnings.warn(
            f"block_k {block_k} is too large, set to 128 for sm90 compression.", stacklevel=2)
    # Load the library (will use cache if available)
    compress_lib = _get_cached_lib()

    return compress_lib.compress_sm90(A, block_k, transposed)
