from tvm import DataType
from typing import Literal
from .mma_layout import (
    ldmatrix_32x4_to_shared_16x8_layout_a,
    ldmatrix_32x4_to_shared_16x8_layout_b,
    ldmatrix_32x8_to_shared_16x16_layout,
    ldmatrix_trans_32x8_to_shared_16x16_layout,
    ldmatrix_32x16_to_shared_16x32_layout_a,
    ldmatrix_32x16_to_shared_16x32_layout_b,
    mma_store_32x8_to_shared_16x16_layout,
    mma_store_32x2_to_shared_8x8_layout_fp64,
)
from .mfma_layout import thread_id_shared_access_64x4_to_16x16_layout_C_n_m

from .mma_layout import get_swizzle_layout  # noqa: F401
from .mma_layout import make_mma_swizzle_layout  # noqa: F401
from .mfma_layout import make_mfma_swizzle_layout  # noqa: F401


# the original implementation and insight is from the following code snippet
# 3rdparty/tvm/python/tvm/tir/tensor_intrin/cuda.py#get_ldmatrix_intrin
def get_ldmatrix_offset(
    matrix: Literal["A", "B"],
    row_idx,
    col_idx,
    stride,
    dtype: Literal["float16", "int8"] = "float16",
    transposed: bool = False,
):
    assert matrix in ["A", "B"], "matrix should be either A or B"
    dtype_bits = DataType(dtype).bits
    if dtype_bits == 32:
        if matrix == "B" and transposed:
            transform_func = ldmatrix_32x4_to_shared_16x8_layout_b
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        elif matrix == "A" and not transposed:
            transform_func = ldmatrix_32x4_to_shared_16x8_layout_a
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            raise ValueError("ldmatrix only supports B transposed and A non-transposed for int8")
    elif dtype_bits == 16:
        transform_func = ldmatrix_32x8_to_shared_16x16_layout
        transform_func_trans = ldmatrix_trans_32x8_to_shared_16x16_layout
        if transposed:
            new_row_idx, new_col_idx = transform_func_trans(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
    elif dtype_bits == 8:
        if matrix == "B" and transposed:
            transform_func = ldmatrix_32x16_to_shared_16x32_layout_b
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        elif matrix == "A" and not transposed:
            transform_func = ldmatrix_32x16_to_shared_16x32_layout_a
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            raise ValueError("ldmatrix only supports B transposed and A non-transposed for int8")
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def shared_16x16_to_mma_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x32_to_mma_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_mma_32x16_layout(i, j):
    thread_id = (i % 16) // 4 + 4 * (j % 8)
    return thread_id, 8 * (j // 8) + (i // 16) * 4 + i % 4


def mma_store_index_map(thread_id, local_id):
    return mma_store_32x8_to_shared_16x16_layout(thread_id, local_id)


def mma_store_index_map_fp64(thread_id, local_id):
    return mma_store_32x2_to_shared_8x8_layout_fp64(thread_id, local_id)


def mfma_store_index_map(thread_id, local_id):
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)


def get_mma_micro_size(dtype: Literal["float16", "int8"]):
    # TODO(lei): FP8 related precision support.
    # Basic Tensor Core Matrix Multiply operation Unit
    """
    Return the MMA (Tensor Core) micro-tile dimensions for a given data type.

    This function returns the micro tile sizes (x, y, k) used by MMA/Tensor Core operations.
    - x: tile width in the output/result dimension
    - y: tile height in the output/result dimension
    - k: tile depth in the reduction/K dimension

    Accepted dtype strings include "float16", "int8" and some FP8 identifiers ("float8_e4m3", "float8_e5m2"). For FP8 and int8 types the reduction depth (`k`) is 32; for float16 it is 16.

    Returns:
        tuple[int, int, int]: (micro_size_x, micro_size_y, micro_size_k)
    """
    micro_size_x = micro_size_y = 16
    micro_size_k = 16
    if dtype in {"float8_e4m3", "float8_e5m2", "int8"}:
        micro_size_k = 32
    return micro_size_x, micro_size_y, micro_size_k
