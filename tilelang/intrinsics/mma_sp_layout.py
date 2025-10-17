from typing import Tuple

from .mma_layout import (
    mma_load_a_32x8_to_shared_16x16_layout,
    mma_load_b_32x4_to_shared_16x8_layout_16bit,

)

def mma_sp_load_a_32x8_to_shared_16x32_layout(thread_id, local_id):
    return mma_load_a_32x8_to_shared_16x16_layout(thread_id, local_id)

def mma_sp_load_b_32x8_to_shared_32x8_layout(thread_id, local_id):
    return mma_load_b_32x4_to_shared_16x8_layout_16bit(thread_id, local_id)

def mma_sp_load_b_32x16_to_shared_32x16_layout(thread_id, local_id):
    row, col = mma_load_b_32x4_to_shared_16x8_layout_16bit(thread_id, local_id % 8)
    return row, col + 8 * (local_id // 8) 


def get_logical_id(thread_id: int) -> int:
    return (thread_id // 4) * 2 + (thread_id % 4) % 2

def metadata_load_32x4_to_shared_16x4_layout_8bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    """
    For 16 bit mma dtype, 8 bit mma dtype
    32x4 // 2 == 16x4, For consecutive 4 threads, only 2 (lower or higher depends on selector) are needed to load metadata.
    Args:
        thread_id (int): The thread id in the warp, range [0, 31]
        local_id (int): The local id in the warp, range [0, 3] (u8 * 4)
    Returns:
        row (int): The row index in the shared memory
    """
    logical_id = get_logical_id(thread_id)
    thread_row = logical_id // 2
    thread_col = logical_id % 2
    local_row = local_id // 2
    local_col = local_id % 2
    row = thread_row + local_row * 8
    col = thread_col * 2 + local_col
    return row, col


def metadata_load_32x2_to_shared_16x2_layout_16bit(thread_id: int, local_id: int) -> Tuple[int, int]:
    """
    For 16 bit mma dtype, 16 bit mma dtype
    32x2 // 2 == 16x2, For consecutive 4 threads, only 2 (lower or higher depends on selector) are needed to load metadata.
    Args:
        thread_id (int): The thread id in the warp, range [0, 31]
        local_id (int): The local id in the warp, range [0, 1] (u16 * 2)
    Returns:
        row (int): The row index in the shared memory
    """
    logical_id = get_logical_id(thread_id)
    thread_row = logical_id // 2
    thread_col = logical_id % 2
    row = thread_row + local_id * 8
    col = thread_col
    return row, col

if __name__ == "__main__":
    # for thread_id in range(32):
    #     print(f"thread_id: {thread_id}, logical_id: {get_logical_id(thread_id)}")
    #     for local_id in range(4):
    #         row, col = metadata_load_32x4_to_shared_16x4_layout_8bit(thread_id, local_id)
    #         print(f"thread_id: {thread_id}, local_id: {local_id} => row: {row}, col: {col}")

    # for tid in range(32):
    #     print(f"thread_id: {tid}, logical_id: {get_logical_id(tid)}")
    #     for lid in range(2):
    #         row, col = metadata_load_32x2_to_shared_16x2_layout_16bit(tid, lid)
    #         print(f"thread_id: {tid}, local_id: {lid} => row: {row}, col: {col}")

    def mma_load_b_32x4_to_shared_16x8_layout_16bit(thread_id, local_id):
        """
            groupID           = %laneid >> 2
            threadID_in_group = %laneid % 4

            row =  (threadID_in_group * 2) + (i & 0x1)           for bi where i <  2
                (threadID_in_group * 2) + (i & 0x1) + 8       for bi where i >= 2

            col = groupID
        """
        row = (thread_id % 4) * 2 + (local_id % 2) + (local_id // 2) * 8
        col = (thread_id // 4)
        return row, col

    def mma_load_b_32x8_to_shared_16x16_layout_16bit_replicate_b(thread_id, local_id):
        row, col = mma_load_b_32x4_to_shared_16x8_layout_16bit(thread_id, local_id % 4)
        return row, col + 8 * (local_id // 4)

    for tid in range(32):
        for lid in range(8):
            row, col = mma_load_b_32x8_to_shared_16x16_layout_16bit_replicate_b(tid, lid)
            print(f"thread_id: {tid}, local_id: {lid} => row: {row}, col: {col}")