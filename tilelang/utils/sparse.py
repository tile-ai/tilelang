from __future__ import annotations
import contextlib
import os
import torch
import warnings
from tilelang.contrib import nvcc
from tilelang.utils.tensor import is_float8_dtype, fp8_remove_negative_zeros_
from torch.utils.cpp_extension import load, _import_module_from_library
from tilelang import env

# Include version information to ensure different versions use separate caches
from tilelang import __version__

# Define paths
compress_util = os.path.join(env.TILELANG_TEMPLATE_PATH, "tl_templates/cuda/compress_sm90.cu")
# Cache directory for compiled extensions
_TORCH_CUDA_VERSION = torch.version.cuda or "cpu"
_CACHE_DIR = os.path.join(
    env.TILELANG_CACHE_DIR,
    "sparse_compressor",
    __version__,
    f"torch_cuda_{_TORCH_CUDA_VERSION}",
)
os.makedirs(_CACHE_DIR, exist_ok=True)


def _torch_cuda_runtime_link_dir() -> str | None:
    """Return a cache-local lib dir that makes -lcudart resolve to PyTorch's cudart."""
    cuda_version = torch.version.cuda
    if not cuda_version:
        return None

    major = cuda_version.split(".", maxsplit=1)[0]
    torch_root = os.path.dirname(torch.__file__)
    runtime_dir = os.path.abspath(os.path.join(torch_root, "..", "nvidia", f"cu{major}", "lib"))
    runtime_lib = os.path.join(runtime_dir, f"libcudart.so.{major}")
    if not os.path.exists(runtime_lib):
        return None

    link_dir = os.path.join(_CACHE_DIR, "cuda_runtime_lib")
    os.makedirs(link_dir, exist_ok=True)
    link_path = os.path.join(link_dir, "libcudart.so")
    if os.path.lexists(link_path) and os.path.realpath(link_path) != os.path.realpath(runtime_lib):
        os.remove(link_path)
    with contextlib.suppress(FileExistsError):
        os.symlink(runtime_lib, link_path)
    return link_dir


def _get_cached_lib():
    name = "compress_lib"

    if os.path.exists(os.path.join(_CACHE_DIR, f"{name}.so")):
        try:
            return _import_module_from_library(name, _CACHE_DIR, is_python_module=True)
        except Exception:
            pass

    # Set TORCH_CUDA_ARCH_LIST
    env._initialize_torch_cuda_arch_flags()
    extra_ldflags = []
    runtime_link_dir = _torch_cuda_runtime_link_dir()
    if runtime_link_dir is not None:
        extra_ldflags.append(f"-L{runtime_link_dir}")
        runtime_dir = os.path.dirname(os.path.realpath(os.path.join(runtime_link_dir, "libcudart.so")))
        extra_ldflags.append(f"-Wl,-rpath,{runtime_dir}")

    # Compile if not cached or loading failed
    return load(
        name=name,
        sources=[compress_util],
        extra_cuda_cflags=[
            "-O2",
            "-std=c++17",
            "-lineinfo",
            f"-I{env.CUTLASS_INCLUDE_DIR}",
            f"-I{env.CUTLASS_INCLUDE_DIR}/../tools/util/include",
            "-arch=sm_90",
        ],
        extra_ldflags=extra_ldflags,
        build_directory=_CACHE_DIR,
    )


def compress_sm90(A: torch.Tensor, block_k: int, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if block_k > 128:
        block_k = 128
        # Ref: https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L145-L146
        warnings.warn(f"block_k {block_k} is too large, set to 128 for sm90 compression.", stacklevel=2)
    # Load the library (will use cache if available)
    compress_lib = _get_cached_lib()

    return compress_lib.compress_sm90(A, block_k, transposed)


def compress_sm80(A: torch.Tensor, transposed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    except ImportError as err:
        raise ImportError(
            "SparseSemiStructuredTensor is not available in this version of PyTorch. Please install a compatible version."
        ) from err
    orig_val = SparseSemiStructuredTensor._FORCE_CUTLASS
    try:
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
        if transposed is not False:
            raise NotImplementedError("transposed flag is deprecated by pytorch")
        compressed = to_sparse_semi_structured(A)
        return compressed.packed, compressed.meta
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = orig_val


def compress(A: torch.Tensor, transposed: bool, arch: str | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a tensor using the appropriate method based on the CUDA architecture.
    """
    if arch is None:
        arch = nvcc.get_target_compute_version()

    compute_version = nvcc.parse_compute_version(arch)

    if compute_version >= (9, 0):
        return compress_sm90(A, transposed=transposed, **kwargs)
    elif compute_version >= (8, 0):
        if transposed:
            A = A.t().contiguous()
        origin_dtype = A.dtype
        if is_float8_dtype(origin_dtype):
            fp8_remove_negative_zeros_(A)
            A = A.view(torch.int8)
        A_sp, E = compress_sm80(A, transposed=False)
        if is_float8_dtype(origin_dtype):
            A_sp = A_sp.view(origin_dtype)
        if transposed:
            A_sp = A_sp.t().contiguous()
        return A_sp, E
    else:
        raise ValueError(f"Unsupported CUDA compute version: {compute_version}. Supported versions are sm_80 and sm_90.")


def randn_semi_sparse(M: int, K: int, dtype=torch.float16, device="cuda", transposed: bool = False):
    """
    Generate a random semi-sparse tensor. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.randn((M, K), dtype=torch.float, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor.to(dtype)  # dtype like float8 might not have randn kernel


def randint_semi_sparse(M: int, K: int, low: int, high: int, dtype=torch.int32, device="cuda", transposed: bool = False):
    """
    Generate a random semi-sparse integer tensor. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        low (int): Lower bound of the random integers
        high (int): Upper bound of the random integers
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.randint(low, high, (M, K), dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor


def arange_semi_sparse(M: int, K: int, dtype=torch.float16, device="cuda", transposed: bool = False):
    """
    Generate a semi-sparse tensor with values from 0 to M*K-1. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = 2, 4
    if dtype == torch.float32:
        elem, group = 1, 2
    tensor = torch.arange(M * K, dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor
