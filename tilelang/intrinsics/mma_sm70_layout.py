def shared_16x4_to_mma_a_32x4_layout(row, col, rep):
    tid = (row % 4) + 16 * ((row // 4) % 2) + 4 * (row // 8) + 8 * rep
    local_id = col
    return tid, local_id


def shared_4x16_to_mma_b_32x4_layout(row, col, rep):
    thread_id = row + 8 * col // 4 + 4 * rep
    local_id = col % 4
    return thread_id, local_id


def shared_16x4_to_mma_b_32x4_layout_trans(row, col, rep):
    thread_id = row % 4 + 4 * rep + 8 * ((row % 8) // 4) + 16 * (row // 8)
    local_id = col
    return thread_id, local_id


def mma_32x8_to_shared_16x16_layout_fp32(thread_id, local_id):
    row = (thread_id % 2) + ((local_id // 2 % 2) * 2) + 4 * (thread_id // 16) + (thread_id % 16 // 4) % 2 * 8
    col = (thread_id % 4 // 2) * 2 + (thread_id % 16 // 8) * 4 + (local_id % 2) + (local_id // 4) * 8
    return row, col


def mma_32x8_to_shared_16x16_layout_fp16(thread_id, local_id):
    row = (thread_id % 4) + (thread_id // 16) * 4 + (thread_id % 8) // 4 * 8
    col = local_id % 4 + ((thread_id % 16) // 8) * 4 + (local_id // 4) * 8
    return row, col


def mma_load_a_32x4_to_shared_16x4_layout(thread_id, local_id):
    row = (thread_id % 4) + (4 * ((thread_id // 16 + thread_id % 16 // 4 * 2) % 4))
    col = local_id
    return row, col


def mma_load_b_32x4_to_shared_16x4_layout_trans(thread_id, local_id):
    row = (thread_id % 4) + 8 * (thread_id // 16) + 4 * ((thread_id // 8) % 2)
    col = local_id
    return row, col


def mma_load_b_32x4_to_shared_4x16_layout(thread_id, local_id):
    row = thread_id % 4
    col = local_id + (4 * (thread_id // 8))
    return row, col
