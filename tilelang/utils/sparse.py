from __future__ import annotations
import torch
import tilelang
import tilelang.language as T
from tilelang.language.dtypes import _TORCH_DTYPE_TO_STR, dtype
from tvm import DataType

_ELEM_PER_THREAD = 32
_BLOCK_M = 16

_DTYPE_CONFIG = {
    torch.float16: (2, 4, T.int16),
    torch.bfloat16: (2, 4, T.int16),
    torch.float32: (1, 2, T.int16),
    torch.int8: (2, 4, T.int16),
}

for _name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _DTYPE_CONFIG[_dt] = (2, 4, T.int16)


def _e_factor(meta_dtype: str, group: int, elem: int) -> int:
    bits_per_pos = (group - 1).bit_length()
    bits_per_group = elem * bits_per_pos
    return (DataType(meta_dtype).bits // bits_per_group) * group


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
)
def _compress_fn(D, dtype, meta_dtype, group=4, elem=2, block_M=_BLOCK_M, elem_per_thread=_ELEM_PER_THREAD):
    e_factor = _e_factor(meta_dtype, group, elem)
    bits_per_pos = (group - 1).bit_length()
    bits_per_group = elem * bits_per_pos
    S = T.dynamic("S")
    print(f"{D=} {elem=} {group=} {e_factor=} {bits_per_pos=} {bits_per_group=}")
    print(f"{[S, D * elem // group]=}, {[S, D // e_factor]=}")

    @T.prim_func
    def kernel(
        dense: T.Tensor([S, D], dtype),
        nonzero: T.Tensor([S, D * elem // group], dtype),
        meta: T.Tensor([S, D // e_factor], meta_dtype),
    ):
        with T.Kernel(S // block_M, threads=(block_M, D // elem_per_thread)) as (bz,):
            tm = T.get_thread_binding(0)
            tn = T.get_thread_binding(1)
            dense_local = T.alloc_local([elem_per_thread], dtype)
            sparse_local = T.alloc_local([elem_per_thread * elem // group], dtype)
            meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)
            nz_idx = T.alloc_local([elem], T.uint8)
            nz_count = T.alloc_var(dtype=T.uint8)

            T.clear(sparse_local)
            T.clear(meta_local)

            T.copy(dense[bz * block_M + tm, tn * elem_per_thread : (tn + 1) * elem_per_thread], dense_local)

            for g_i in range(elem_per_thread // group):
                T.clear(nz_idx)
                local_idx = g_i * group

                nz_count = 0
                for i in T.serial(group):
                    if dense_local[local_idx + i] != 0:
                        nz_idx[nz_count] = i
                        nz_count = nz_count + 1

                for i in T.serial(elem):
                    sparse_local[local_idx * elem // group + i] = dense_local[local_idx + nz_idx[i]]
                    meta_local[local_idx // e_factor] |= T.shift_left(
                        nz_idx[i].astype(meta_dtype),
                        (bits_per_group * (g_i % (e_factor // group)) + bits_per_pos * i).astype(meta_dtype),
                    )

            sparse_per_thread = elem_per_thread * elem // group
            T.copy(sparse_local, nonzero[bz * block_M + tm, tn * sparse_per_thread : (tn + 1) * sparse_per_thread])
            T.copy(meta_local, meta[bz * block_M + tm, tn * (elem_per_thread // e_factor) : (tn + 1) * (elem_per_thread // e_factor)])

    return kernel


def compress(
    A: torch.Tensor,
    meta_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.dtype not in _DTYPE_CONFIG:
        raise ValueError(f"Unsupported dtype {A.dtype}. Supported: {list(_DTYPE_CONFIG)}")
    assert A.is_contiguous(), "Input must be contiguous"
    assert A.dim() == 2, "Input must be 2D"

    elem, group = _DTYPE_CONFIG[A.dtype][:2]
    meta_dtype = dtype(_TORCH_DTYPE_TO_STR[meta_dtype]) if meta_dtype is not None else _DTYPE_CONFIG[A.dtype][2]
    S, D = A.shape
    assert D % _ELEM_PER_THREAD == 0, f"Last dim D={D} must be divisible by {_ELEM_PER_THREAD}"
    assert S % _BLOCK_M == 0, f"Rows S={S} must be divisible by block_M={_BLOCK_M}"

    A_sparse, E = _compress_fn(D, dtype(_TORCH_DTYPE_TO_STR[A.dtype]), meta_dtype, group, elem, _BLOCK_M, _ELEM_PER_THREAD)(A)

    return A_sparse, E


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
