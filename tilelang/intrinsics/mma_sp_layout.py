from typing import Tuple

from tilelang.intrinsics.mma_layout import (
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_a_32x8_to_shared_16x16_layout,
)

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

def metadata_8bit_load_32x4_to_shared_16x4_layout_32bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    logical_id = get_logical_id_32bit(thread_id)
    row = logical_id // 4 + local_id * 8
    col = logical_id % 4
    return row, col

def metadata_16bit_load_32x2_to_shared_16x2_layout_32bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    logical_id = get_logical_id_32bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = logical_id % 2
    return row, col

def metadata_8bit_load_32x4_to_shared_16x4_layout_16bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    return metadata_8bit_load_32x4_to_shared_16x4_layout_32bit(thread_id, local_id)  # same mapping for 16bit and 32bit

def metadata_16bit_load_32x2_to_shared_16x2_layout_16bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    return metadata_16bit_load_32x2_to_shared_16x2_layout_32bit(thread_id, local_id)  # same mapping for 16bit and 32bit

def get_logical_id_8bit(thread_id: int) -> int:
    return thread_id

def metadata_8bit_load_32x4_to_shared_16x4_layout_8bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = (logical_id % 4) // 2 * 4 + local_id
    return row, col

def metadata_16bit_load_32x2_to_shared_16x4_layout_8bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 2 + local_id * 8
    col = (logical_id % 4) // 2 * 2 + local_id
    return row, col

def metadata_32bit_load_32x1_to_shared_16x2_layout_8bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    # local_id is always 0
    logical_id = get_logical_id_8bit(thread_id)
    row = logical_id // 4 + (logical_id % 2) * 8
    col = (logical_id % 4) // 2
    return row, col
