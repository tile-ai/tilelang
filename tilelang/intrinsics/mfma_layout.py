from tvm import DataType
from tvm.runtime import convert
import tilelang.language as T


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    return thread_id, convert(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_m = shared_16x16_to_local_64x4_layout_B
shared_16x16_to_local_64x4_layout_k_n = shared_16x16_to_local_64x4_layout_B


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return i, j


def thread_id_shared_access_64x8_to_16x32_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 8 + local_id
    return i, j


def shared_16x32_to_local_64x8_layout_A(i, j):
    thread_id = i + 16 * (j // 8)
    local = j % 8
    return thread_id, local


def thread_id_shared_access_64x8_to_16x32_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 8
    j = thread_id % 16
    return i, j


def shared_16x32_to_local_64x8_layout_B(i, j):
    thread_id = j + (i // 8) * 16
    local = i % 8
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 16
    return i, j


def shared_16x64_to_local_64x16_layout_A(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 16
    j = thread_id % 16
    return i, j


def shared_16x64_to_local_64x16_layout_B(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def make_mfma_swizzle_layout(shared_buf, vecSize=8):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    numBanks = 32
    bankBitWidth = 32
    SIMDWidth = 16

    innerDimLength = shape[-1]
    typeWidthInBit = DataType(dtype).bits

    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeWidthInBit
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)

    def transform(row, col):
        phase = (row // perPhase) % maxPhase
        colOffSwizzled = ((col // vecSize) ^ phase) * vecSize
        colOffOrdered = col % vecSize
        colOff = colOffSwizzled + colOffOrdered
        return row, colOff

    return T.Layout(shape, transform)
