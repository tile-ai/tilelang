import torch
import tilelang
import tilelang.language as T
from tilelang.language.dtypes import _TORCH_DTYPE_TO_STR, dtype

GROUP_CONFIG: dict[dtype, tuple[int, int]] = {
    T.float: (1, 2),
    T.float16: (2, 4),
    T.bfloat16: (2, 4),
    T.int8: (2, 4),
    T.uint8: (2, 4),
    T.float8_e4m3: (2, 4),
    T.float8_e5m2: (2, 4),
}

_BITS_PER_GROUP = 4


def get_e_factor(a_dtype: dtype, meta_dtype: dtype) -> int:
    """Return how many a_dtype elements are indexed by one meta_dtype element."""
    _, group = GROUP_CONFIG[a_dtype]
    return (dtype(meta_dtype).bits // _BITS_PER_GROUP) * group


def get_e_replicate_factor(a_dtype: dtype) -> int:
    """Return how many consecutive threads share the same logical metadata value."""
    return 1 if dtype(a_dtype).bits <= 8 else 2


def _to_tl_dtype(torch_dtype: torch.dtype) -> dtype:
    return dtype(_TORCH_DTYPE_TO_STR[torch_dtype])


_ELEM_PER_THREAD = 32
_BLOCK_M = 16
_BLOCK_K = 1024
_DEFAULT_META_DTYPE = T.int16


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
)
def _compress_fn(D, dtype, meta_dtype, block_M=_BLOCK_M, block_K=_BLOCK_K, elem_per_thread=_ELEM_PER_THREAD):
    e_factor = get_e_factor(dtype, meta_dtype)
    S = T.dynamic("S")
    assert elem_per_thread >= e_factor
    assert block_K % elem_per_thread == 0

    if dtype.bits <= 16:
        elem, group = 2, 4

        @T.prim_func
        def compress_8bit_16bit_ordered_metadata(
            dense: T.Tensor([S, D], dtype),
            nonzero: T.Tensor([S, D * elem // group], dtype),
            meta: T.Tensor([S, D // e_factor], meta_dtype),
        ):
            with T.Kernel(S // block_M, D // block_K, threads=(block_M, block_K // elem_per_thread)) as (bz, bk):
                tm = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                dense_local = T.alloc_local([elem_per_thread], dtype)
                sparse_local = T.alloc_local([elem_per_thread * elem // group], dtype)
                meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)
                nz_idx = T.alloc_local([elem], T.uint8)
                nz_count = T.alloc_var(dtype=T.uint8)

                T.clear(sparse_local)
                T.clear(meta_local)

                k_base = bk * block_K
                T.copy(
                    dense[bz * block_M + tm, bk * block_K + tn * elem_per_thread : k_base + (tn + 1) * elem_per_thread],
                    dense_local,
                )

                for gid in T.unroll(elem_per_thread // group):
                    T.clear(nz_idx)
                    local_idx = gid * group

                    nz_count = 0
                    for i in T.unroll(group):
                        nz_idx[nz_count] = T.if_then_else(dense_local[local_idx + i] != 0, i, nz_idx[nz_count])
                        nz_count = T.if_then_else(dense_local[local_idx + i] != 0, nz_count + 1, nz_count)

                    T.device_assert(nz_count <= elem, "More nonzeros than expected in a group")

                    if nz_count == 1:
                        if nz_idx[0] == 0:
                            nz_idx[1] = 1
                        else:
                            nz_idx[0], nz_idx[1] = nz_idx[1], nz_idx[0]
                    elif nz_count == 0:
                        nz_idx[0], nz_idx[1] = 0, 1

                    for i in T.unroll(elem):
                        sparse_local[local_idx * elem // group + i] = dense_local[local_idx + nz_idx[i]]
                        meta_local[local_idx // e_factor] |= T.shift_left(
                            nz_idx[i].astype(meta_dtype),
                            (4 * (gid % (e_factor // group)) + 2 * i),
                        )

                sparse_per_thread = elem_per_thread * elem // group
                sparse_base = k_base * elem // group
                meta_base = k_base // e_factor
                T.copy(
                    sparse_local,
                    nonzero[
                        bz * block_M + tm,
                        sparse_base + tn * sparse_per_thread : sparse_base + (tn + 1) * sparse_per_thread,
                    ],
                )
                T.copy(
                    meta_local,
                    meta[
                        bz * block_M + tm,
                        meta_base + tn * (elem_per_thread // e_factor) : meta_base + (tn + 1) * (elem_per_thread // e_factor),
                    ],
                )

        return compress_8bit_16bit_ordered_metadata
    elif dtype.bits == 32:
        elem, group = 1, 2

        @T.prim_func
        def compress_32bit_ordered_metadata(
            dense: T.Tensor([S, D], dtype),
            nonzero: T.Tensor([S, D * elem // group], dtype),
            meta: T.Tensor([S, D // e_factor], meta_dtype),
        ):
            with T.Kernel(S // block_M, D // block_K, threads=(block_M, block_K // elem_per_thread)) as (bz, bk):
                tm = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                dense_local = T.alloc_local([elem_per_thread], dtype)
                sparse_local = T.alloc_local([elem_per_thread * elem // group], dtype)
                meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)
                nz_idx = T.alloc_local([elem], T.uint8)
                nz_count = T.alloc_var(dtype=T.uint8)

                T.clear(sparse_local)
                T.clear(meta_local)

                k_base = bk * block_K
                T.copy(
                    dense[bz * block_M + tm, k_base + tn * elem_per_thread : k_base + (tn + 1) * elem_per_thread],
                    dense_local,
                )

                for gid in T.unroll(elem_per_thread // group):
                    T.clear(nz_idx)
                    local_idx = gid * group

                    nz_count = 0
                    for i in T.unroll(group):
                        nz_idx[nz_count] = T.if_then_else(dense_local[local_idx + i] != 0, i, nz_idx[nz_count])
                        nz_count = T.if_then_else(dense_local[local_idx + i] != 0, nz_count + 1, nz_count)

                    T.device_assert(nz_count <= elem, "More nonzeros than expected in a group")

                    if nz_count == 0:
                        sparse_local[local_idx * elem // group] = 0
                        meta_local[local_idx // e_factor] |= T.shift_left(0b0100, 4 * (gid % (e_factor // group)))
                    else:
                        sparse_local[local_idx * elem // group] = dense_local[local_idx + nz_idx[0]]
                        meta_local[local_idx // e_factor] |= T.shift_left(
                            T.if_then_else(nz_idx[0] == 0, 0b0100, 0b1110),
                            4 * (gid % (e_factor // group)),
                        )

                sparse_per_thread = elem_per_thread * elem // group
                sparse_base = k_base * elem // group
                meta_base = k_base // e_factor
                T.copy(
                    sparse_local,
                    nonzero[
                        bz * block_M + tm,
                        sparse_base + tn * sparse_per_thread : sparse_base + (tn + 1) * sparse_per_thread,
                    ],
                )
                T.copy(
                    meta_local,
                    meta[
                        bz * block_M + tm,
                        meta_base + tn * (elem_per_thread // e_factor) : meta_base + (tn + 1) * (elem_per_thread // e_factor),
                    ],
                )

        return compress_32bit_ordered_metadata


def torch_compress(dense: torch.Tensor, meta_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: FA100
    """
    Reference 2:4 sparse compressor in pure PyTorch with natural row-major metadata. Modified from https://github.com/pytorch/pytorch/blob/bfa6895a345f6568624a4769238af6a9225e3fb8/torch/sparse/_semi_structured_conversions.py#L47

    Each 4-bit chunk of the metadata integer encodes the two nonzero positions
    within one group of 4 consecutive elements:
        bits [1:0] = index of first  nonzero (0-3)
        bits [3:2] = index of second nonzero (0-3)

    """
    if dense.dim() != 2:
        raise RuntimeError(f"Expected 2D tensor, got {dense.dim()}D")
    m, k = dense.shape

    is_32bit = dense.dtype == torch.float32
    ksparse = 2 if is_32bit else 4
    # int8 uses int32 metadata to match CUTLASS convention; all others use int16
    if meta_dtype is None:
        meta_dtype = torch.int32 if dense.dtype == torch.int8 else torch.int16
    quadbits = meta_dtype.itemsize * 8 // 4  # 4-bit groups that fit in one meta element

    # 8-bit non-integer types (float8 variants) may not support gather; view as int8
    gather_dtype = torch.int8 if (dense.element_size() == 1 and dense.dtype != torch.int8) else None
    work = dense.view(gather_dtype) if gather_dtype is not None else dense

    groups = work.view(-1, k // ksparse, ksparse)
    nz = groups != 0
    if not is_32bit:
        m0, m1, _m2, m3 = nz.unbind(-1)
    else:
        m0, _m2 = m1, m3 = nz.unbind(-1)

    meta_ncols = k // (ksparse * quadbits)

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    idxs0 = expr1.to(torch.int64) | (expr2.to(torch.int64) << 1)
    idxs1 = (expr0 | expr2 | m3).to(torch.int64) | ((expr1 | ~m1).to(torch.int64) << 1)

    if not is_32bit:
        sp0 = groups.gather(-1, idxs0.unsqueeze(-1))
        sp1 = groups.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sp0, sp1), dim=-1).view(m, k // 2)
    else:
        sparse = groups.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)

    if gather_dtype is not None:
        sparse = sparse.view(dense.dtype)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view(-1, meta_ncols, quadbits).to(meta_dtype)
    # Pack 4-bit chunks into each meta element (little-endian)
    meta = meta_n[:, :, 0]
    for i in range(1, quadbits):
        meta = meta | (meta_n[:, :, i] << (4 * i))

    return sparse, meta


def compress(
    A: torch.Tensor,
    meta_dtype: torch.dtype | None = None,  # noqa: FA100
    block_m: int | None = None,  # noqa: FA100
    block_k: int | None = None,  # noqa: FA100
) -> tuple[torch.Tensor, torch.Tensor]:
    assert A.is_contiguous(), "Input must be contiguous"
    assert A.dim() == 2, "Input must be 2D"

    tl_meta_dtype = _to_tl_dtype(meta_dtype) if meta_dtype is not None else _DEFAULT_META_DTYPE
    S, D = A.shape
    block_m = min(_BLOCK_M, S) if block_m is None else block_m
    block_k = min(_BLOCK_K, D) if block_k is None else block_k
    assert block_k % _ELEM_PER_THREAD == 0, f"block_k={block_k} must be divisible by {_ELEM_PER_THREAD}"
    assert D % block_k == 0, f"Last dim D={D} must be divisible by block_k={block_k}"
    assert S % block_m == 0, f"Rows S={S} must be divisible by block_M={block_m}"
    assert D % _ELEM_PER_THREAD == 0, f"Last dim D={D} must be divisible by {_ELEM_PER_THREAD}"

    A_sparse, E = _compress_fn(D, _to_tl_dtype(A.dtype), tl_meta_dtype, block_m, block_k, _ELEM_PER_THREAD)(A)

    return A_sparse, E


def randn_semi_sparse(M: int, K: int, dtype: torch.dtype = torch.float16, device: torch.device = "cuda", transposed: bool = False):
    """
    Generate a random semi-sparse tensor. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = GROUP_CONFIG[_to_tl_dtype(dtype)]
    tensor = torch.randn((M, K), dtype=torch.float, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor.to(dtype)  # dtype like float8 might not have randn kernel


def randint_semi_sparse(
    M: int,
    K: int,
    low: int,
    high: int,
    dtype: torch.dtype = torch.int32,
    device: torch.device = "cuda",
    transposed: bool = False,
):
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
    elem, group = GROUP_CONFIG[_to_tl_dtype(dtype)]
    tensor = torch.randint(low, high, (M, K), dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor


def arange_semi_sparse(M: int, K: int, dtype: torch.dtype = torch.float16, device: torch.device = "cuda", transposed: bool = False):
    """
    Generate a semi-sparse tensor with values from 0 to M*K-1. The generated tensor will have 2:4 sparsity along the K dimension.
    Args:
        M (int): Number of rows
        K (int): Number of columns
        dtype: Data type of the tensor
        device: Device to create the tensor on
        transposed (bool): If True, returns a transposed tensor of shape (K, M)
    """
    elem, group = GROUP_CONFIG[_to_tl_dtype(dtype)]
    tensor = torch.arange(M * K, dtype=dtype, device=device).view(M, -1, group)
    indice = tensor.topk(elem, dim=-1).indices
    tensor.scatter_(-1, indice, 0)
    tensor = tensor.view(M, K)
    if transposed:
        tensor = tensor.t().contiguous()
    return tensor
