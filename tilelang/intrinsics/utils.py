# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tvm import DataType
from typing import Literal
from .mma_layout import (
    ldmatrix_32x8_to_shared_16x16_layout,
    ldmatrix_trans_32x8_to_shared_16x16_layout,
    ldmatrix_16x32_to_shared_16x32_layout_a,
    ldmatrix_16x32_to_shared_16x32_layout_b,
    mma_store_32x8_to_shared_16x16_layout,
)
from .mfma_layout import (thread_id_shared_access_64x4_to_16x16_layout_C_n_m)

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
    if dtype_bits == 16:
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
            transform_func = ldmatrix_16x32_to_shared_16x32_layout_b
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        elif matrix == "A" and not transposed:
            transform_func = ldmatrix_16x32_to_shared_16x32_layout_a
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


def mfma_store_index_map(thread_id, local_id):
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)


def get_mma_micro_size(dtype: Literal["float16", "int8"]):
    # TODO(lei): FP8 related precision support.
    # Basic Tensor Core Matrix Multiply operation Unit
    micro_size_x = micro_size_y = 16
    micro_size_k = 16
    if dtype in {"e4m3_float8", "e5m2_float8", "int8"}:
        micro_size_k = 32
    return micro_size_x, micro_size_y, micro_size_k


def index_to_coordinates(index, shape):
    '''
    General Implementation of:
        vjj = index % (micro_size_k // num_elems_per_byte)
        coordinates[-1] = index % shape[-1]; 
        vii = index // (micro_size_k // num_elems_per_byte) % micro_size_y
        index = index // shape[-1]; coordinates[-2] = index % shape[-2];
        vj = index // (micro_size_k // num_elems_per_byte * micro_size_y) % block_K // (micro_size_k // num_elems_per_byte)
        index = index // shape[-2]; coordinates[-3] = index % shape[-3];
        vi = index // (micro_size_k // num_elems_per_byte * micro_size_y * (block_K // (micro_size_k // num_elems_per_byte))) % block_N // micro_size_y
        index = index // shape[-3]; coordinates[-4] = index % shape[-4];
    '''
    coordinates = []
    dims = len(shape)
    for i in range(dims):
        coordinates.append(index % shape[dims - i - 1])
        index = index // shape[dims - i - 1]
    coordinates.reverse()
    return coordinates
