from __future__ import annotations
from tvm import arith, DataType
import tilelang.language as T


def ldmatrix_32x4_to_shared_16x8_layout_a(thread_id, local_id):
    row = thread_id % 16
    col = (thread_id // 16) * 4 + local_id % 4
    return row, col


def ldmatrix_32x4_to_shared_16x8_layout_b(thread_id, local_id):
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = ((thread_id % 16) // 8) * 4 + local_id % 4
    return row, col


def ldmatrix_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = thread_id % 16
    col = 8 * (thread_id // 16) + local_id % 8
    return row, col


def ldmatrix_trans_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (thread_id // 16) + (thread_id % 8)
    col = 8 * ((thread_id % 16) // 8) + local_id % 8
    return row, col


def ldmatrix_32x16_to_shared_16x32_layout_a(thread_id, local_id):
    row = thread_id % 16
    col = local_id + (thread_id // 16) * 16
    return row, col


def ldmatrix_32x16_to_shared_16x32_layout_b(thread_id, local_id):
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = local_id + 16 * ((thread_id % 16) // 8)
    return row, col


def mma_store_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


def mma_store_32x2_to_shared_8x8_layout_fp64(thread_id, local_id):
    row = thread_id // 4
    col = (thread_id % 4) * 2 + local_id
    return row, col


# sr represents spatial + reduction layout
# the first axis is spatial while the second axis is reduction
# mma.sync matrix A layout, if wanna trans, please apply map_indices
def shared_16x8_to_mma_a_32x4_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 2 * (j // 4) + (i // 8)


def shared_16x8_to_mma_a_32x4_layout_trans(i, j):
    return shared_16x8_to_mma_a_32x4_layout(j, i)


# mma.sync matrix B layout, if wanna trans, please apply map_indices
def shared_16x8_to_mma_b_32x4_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 2 * (i // 8) + (j // 4)


def shared_16x8_to_mma_b_32x4_layout_trans(i, j):
    return shared_16x8_to_mma_b_32x4_layout(j, i)


shared_16x8_to_mma_32x4_layout_sr_a = shared_16x8_to_mma_a_32x4_layout
shared_16x8_to_mma_32x4_layout_sr_b = shared_16x8_to_mma_b_32x4_layout
shared_16x8_to_mma_32x4_layout_rs_a = shared_16x8_to_mma_a_32x4_layout_trans
shared_16x8_to_mma_32x4_layout_rs_b = shared_16x8_to_mma_b_32x4_layout_trans


def shared_16x16_to_mma_a_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x16_to_mma_a_32x8_layout_trans(i, j):
    return shared_16x16_to_mma_a_32x8_layout(j, i)


def shared_16x16_to_mma_b_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (i // 8) + (j // 8) * 2 + (j % 2)


def shared_16x16_to_mma_b_32x8_layout_trans(i, j):
    return shared_16x16_to_mma_b_32x8_layout(j, i)


shared_16x16_to_mma_32x8_layout_sr_a = shared_16x16_to_mma_a_32x8_layout
shared_16x16_to_mma_32x8_layout_sr_b = shared_16x16_to_mma_b_32x8_layout
shared_16x16_to_mma_32x8_layout_rs_a = shared_16x16_to_mma_a_32x8_layout_trans
shared_16x16_to_mma_32x8_layout_rs_b = shared_16x16_to_mma_b_32x8_layout_trans


def shared_16x32_to_mma_a_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_mma_a_32x16_layout_trans(i, j):
    return shared_16x32_to_mma_a_32x16_layout(j, i)


def shared_16x32_to_mma_b_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (i // 8) + (j // 16) * 4 + j % 4


def shared_32x16_to_mma_b_32x16_layout_trans(i, j):
    return shared_16x32_to_mma_b_32x16_layout(j, i)


shared_16x32_to_mma_32x16_layout_sr_a = shared_16x32_to_mma_a_32x16_layout
shared_16x32_to_mma_32x16_layout_sr_b = shared_16x32_to_mma_b_32x16_layout
shared_16x32_to_mma_32x16_layout_rs_a = shared_32x16_to_mma_a_32x16_layout_trans
shared_16x32_to_mma_32x16_layout_rs_b = shared_32x16_to_mma_b_32x16_layout_trans


def mma_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


def mma_load_a_32x4_to_shared_16x8_layout(thread_id, local_id):
    row = 8 * (local_id % 2) + (thread_id // 4)
    col = 4 * (local_id // 2) + (thread_id % 4)
    return row, col


def mma_load_b_32x4_to_shared_16x8_layout(thread_id, local_id):
    row = 8 * (local_id // 2) + (thread_id // 4)
    col = 4 * (local_id % 2) + (thread_id % 4)
    return row, col


def mma_load_a_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id % 8 // 4) + (thread_id // 4)
    col = 16 * (local_id // 8) + (thread_id % 4) * 4 + (local_id % 4)
    return row, col


def mma_load_a_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
            groupID + 8         Otherwise

    col =  (threadID_in_group * 2) + (i & 0x1)          for ai where i <  4
    (threadID_in_group * 2) + (i & 0x1) + 8      for ai where i >= 4
    """
    row = (thread_id // 4) + 8 * (local_id % 4 // 2)
    col = (thread_id % 4) * 2 + (local_id % 2) + 8 * (local_id // 4)
    return row, col


def mma_load_b_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id // 8) + (thread_id // 4)
    col = 16 * (local_id % 8 // 4) + (thread_id % 4) * 4 + (local_id % 4)
    return row, col


def mma_load_b_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =  (threadID_in_group * 2) + (i & 0x1)           for bi where i <  2
        (threadID_in_group * 2) + (i & 0x1) + 8       for bi where i >= 2

    col = groupID
    """
    col = (thread_id % 4) * 2 + ((local_id % 4) % 2) + ((local_id % 4) // 2) * 8
    row = (thread_id // 4) + 8 * (local_id // 4)
    return row, col


def shared_16x16_to_mma_32x8_smoothlayout(i, j):
    return (i * 2 + j // 8, j % 8)


def shared_16x32_to_mma_32x16_smoothlayout(i, j):
    return (i * 2 + j // 16, j % 16)


def shared_32x16_to_mma_32x16_smoothlayout(i, j):
    return (i * 2 + j // 16, j % 16)


def get_swizzle_layout(row_idx, col_idx, row_size, dtype: DataType | str, swizzle_bytes=None):
    ana = arith.Analyzer()
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    row_bytes = dtype.bits * row_size // 8
    assert row_bytes % 32 == 0, "Row size must be multiple of 32B."
    if swizzle_bytes is None:
        swizzle_bytes = min(128, row_bytes)
    # 128B swizzle
    #   Use 8 * 8 permuted layout
    #   Every number below corresponds to 16B
    #   0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #   0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    #   0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
    #   0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
    #   0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
    #   0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
    #   0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
    #   0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
    # 64B swizzle
    #  Use 8 * 4 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
    #  0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
    #  0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
    #  0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
    # 32B swizzle
    #  Use 8 * 2 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #  0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    elem_per_16B = 128 // dtype.bits
    col_idx_16B = col_idx // elem_per_16B
    col_idx_in_16B = col_idx % elem_per_16B
    new_col_idx_16B = col_idx_16B ^ (row_idx % (swizzle_bytes // 16))
    return row_idx, ana.simplify(new_col_idx_16B * elem_per_16B + col_idx_in_16B)


def make_mma_swizzle_layout(shared_buf, is_smooth: bool = False):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits % 512 == 0
    if is_smooth or (not can_swizzle):
        return T.Layout(shape, lambda *args: args)

    def transform_func(*args):
        i, j = args[-2:]
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [*args[:-2], new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)
