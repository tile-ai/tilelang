from tvm import DataType
from typing import Literal

from tilelang.intrinsics.mma_layout import (
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_a_32x8_to_shared_16x16_layout,
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_a,
)


def shared_16x16_to_mma_sp_layout_sr_a(i, j):
    return shared_16x8_to_mma_32x4_layout_sr_a(i, j)


def shared_16x16_to_mma_sp_layout_sr_b(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 4 * (i // 8) + (j // 4)


def shared_16x32_to_mma_sp_layout_sr_a(i, j):
    return shared_16x16_to_mma_32x8_layout_sr_a(i, j)


def shared_16x32_to_mma_sp_layout_sr_b(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 8 * (i // 8) + (j // 8) * 2 + (j % 2)


def shared_16x64_to_mma_sp_layout_sr_a(i, j):
    return shared_16x32_to_mma_32x16_layout_sr_a(i, j)


def shared_16x64_to_mma_sp_layout_sr_b(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 16 * (i // 8) + (j // 16) * 4 + j % 4


def mma_sp_load_a_32x4_to_shared_16x16_layout(thread_id, local_id):
    return mma_load_a_32x4_to_shared_16x8_layout(thread_id, local_id)


def mma_sp_load_a_32x8_to_shared_16x32_layout(thread_id, local_id):
    return mma_load_a_32x8_to_shared_16x16_layout(thread_id, local_id)


def mma_sp_load_a_32x16_to_shared_16x64_layout(thread_id, local_id):
    return mma_load_a_32x16_to_shared_16x32_layout(thread_id, local_id)


def mma_sp_load_b_32x8_to_shared_16x16_layout(thread_id, local_id):
    col = 4 * (local_id % 4) + (thread_id % 4)
    row = 8 * (local_id // 4) + (thread_id // 4)
    return row, col


def mma_sp_load_b_32x16_to_shared_16x32_layout(thread_id, local_id):
    col = (thread_id % 4) * 2 + (local_id % 2) + ((local_id % 8) // 2) * 8
    row = (thread_id // 4) + 8 * (local_id // 8)
    return row, col


def mma_sp_load_b_32x32_to_shared_16x64_layout(thread_id, local_id):
    col = (thread_id % 4) * 4 + (local_id % 4) + 16 * ((local_id % 16) // 4)
    row = (thread_id // 4) + 8 * (local_id // 16)
    return row, col


def get_logical_id_32bit(thread_id: int) -> int:
    return (thread_id // 4) * 2 + (thread_id % 4) % 2


def metadata_8bit_load_32x4_to_shared_16x4_layout_32bit(thread_id: int, local_id: int) -> tuple[int, int]:
    logical_id = get_logical_id_32bit(thread_id)
    row = logical_id // 4 + local_id * 8
    col = logical_id % 4
    return row, col


def metadata_16bit_load_32x2_to_shared_16x2_layout_32bit(thread_id: int, local_id: int) -> tuple[int, int]:
    logical_id = get_logical_id_32bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = logical_id % 2
    return row, col


def metadata_8bit_load_32x4_to_shared_16x4_layout_16bit(thread_id: int, local_id: int) -> tuple[int, int]:
    return metadata_8bit_load_32x4_to_shared_16x4_layout_32bit(thread_id, local_id)  # same mapping for 16bit and 32bit


def metadata_16bit_load_32x2_to_shared_16x2_layout_16bit(thread_id: int, local_id: int) -> tuple[int, int]:
    return metadata_16bit_load_32x2_to_shared_16x2_layout_32bit(thread_id, local_id)  # same mapping for 16bit and 32bit


def get_logical_id_8bit(thread_id: int) -> int:
    return thread_id


def metadata_8bit_load_32x4_to_shared_16x4_layout_8bit(thread_id: int, local_id: int) -> tuple[int, int]:
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = (logical_id % 4) // 2 * 4 + local_id
    return row, col


def metadata_16bit_load_32x2_to_shared_16x4_layout_8bit(thread_id: int, local_id: int) -> tuple[int, int]:
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = (logical_id % 4) // 2 * 2 + local_id
    return row, col


def metadata_32bit_load_32x1_to_shared_16x2_layout_8bit(thread_id: int, local_id: int) -> tuple[int, int]:
    # local_id is always 0
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 4 + (logical_id % 2) * 8
    col = (logical_id % 4) // 2
    return row, col


def ldmatrix_trans_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = (local_id // 4) * 8 + thread_id % 8
    col = (thread_id // 8) * 4 + local_id % 4
    return row, col


def ldmatrix_32x16_to_shared_32x16_layout(thread_id, local_id):
    row = thread_id
    col = local_id % 8 + 8 * (local_id // 8)
    return row, col


def ldmatrix_trans_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id // 8) + thread_id % 8
    col = (thread_id // 8) * 8 + local_id % 8
    return row, col


def ldmatrix_trans_32x32_to_shared_shared_16x64_layout(thread_id, local_id):
    row = (local_id // 16) * 8 + thread_id % 8
    col = (thread_id // 8) * 16 + local_id % 16
    return row, col


def get_ldmatrix_offset_b(
    matrix: Literal["B"],
    row_idx,
    col_idx,
    stride,
    dtype: Literal["float16", "int8"] = "float16",
    transposed: bool = False,
):
    assert matrix == "B", "matrix should be B"
    dtype_bits = DataType(dtype).bits
    if dtype_bits == 32:
        if transposed:
            transform_func = ldmatrix_trans_32x8_to_shared_16x16_layout
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            raise ValueError("ldmatrix only supports B transposed for 32-bit dtype")
    elif dtype_bits == 16:
        transform_func = ldmatrix_32x16_to_shared_32x16_layout
        transform_func_trans = ldmatrix_trans_32x16_to_shared_16x32_layout
        if transposed:
            new_row_idx, new_col_idx = transform_func_trans(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
    elif dtype_bits == 8:
        if transposed:
            transform_func = ldmatrix_trans_32x32_to_shared_shared_16x64_layout
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx * stride + new_col_idx
        else:
            raise ValueError("ldmatrix only supports B transposed for 8-bit dtype")
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
