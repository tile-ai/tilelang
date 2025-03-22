# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import math
from typing import Optional

import torch

import tilelang
import tilelang.language as T
from tilelang.cache import clear_cache

clear_cache()


def _is_power_of_two(n: int):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def gpu_2d_continuous_cumsum(
    M: int,
    N: int,
    ty_len: int = 4,
    tx_len: int = 32,
    in_dtype: str = "int32",
    out_dtype: Optional[str] = None,
):
    """Generate GPU kernel for 2D continuous cumsum, i.e. The cumsum axis is -1

    Parameters
    ----------
    M : int
        The number of rows of the input tensor

    N : int
        The number of columns of the input tensor

    ty_len : int
        The length of thread.y

    tx_len : int
        The length of thread.x

    in_dtype : str
        The input data type

    out_dtype : Optional[str]
        The output data type, if None, it will be the same as in_dtype

    Returns
    -------
    cumsum : PrimFunc
        The generated cumsum kernel
    """

    out_dtype = out_dtype or in_dtype

    # Configuration for GPU kernel
    TX = T.int32(tx_len)  # thread.x
    TY = T.int32(ty_len)  # thread.y
    thread_elem = N  # number of elements in single thread

    if not _is_power_of_two(TX) or not _is_power_of_two(TY) or not _is_power_of_two(N):
        raise ValueError("Configuration of TX, TY, N must be power of 2")

    # number of elements to be processed by single warp
    warp_elem = T.int32(tx_len * thread_elem)
    # number of elements to be processed by single block(SM)
    block_elem = T.int32(tx_len * ty_len * thread_elem)

    LOG_TX = T.int32(int(math.log2(tx_len)))
    LOG_BLOCK_N = T.int32(int(math.log2(tx_len * ty_len * thread_elem)))

    @T.macro
    def block_inclusive_inside_block(
        batch: T.int32,
        cur_len: T.int32,
        source: T.Buffer,
        output: T.Buffer,
        tmp_buf: T.Buffer,
        src_offset: T.int32,
        tmp_offset: T.int32,
    ):
        local_buf = T.alloc_buffer((thread_elem,), out_dtype, scope="local")
        shared_buf = T.alloc_buffer((block_elem,), out_dtype, scope="shared")
        bx = T.get_block_binding(0)
        by = T.get_block_binding(1)
        tx = T.get_thread_binding(0)
        ty = T.get_thread_binding(1)

        tx_idx = bx * block_elem + ty * warp_elem + tx * thread_elem
        # Load data from global memory
        for i in T.vectorized(N):
            local_buf[i] = T.if_then_else(
                tx_idx + i < cur_len,
                T.Cast(out_dtype, source[by, src_offset + tx_idx + i]),
                T.Cast(out_dtype, 0),
            )
        # Inclusive scan inside thread
        for i in T.serial(1, N):
            local_buf[i] += local_buf[i - 1]
        # Store data to shared memory
        for i in T.vectorized(N):
            shared_buf[ty * warp_elem + tx * thread_elem + i] = local_buf[i]
        # Inclusive scan inside warp
        for i in T.serial(LOG_TX):
            for j in T.vectorized(N):
                idx: T.int32 = ty * warp_elem + tx * thread_elem
                if tx >= (1 << i):
                    shared_buf[idx + j] += shared_buf[idx - (1 << i) * thread_elem + N - 1]
        # Inclusive scan inside block
        for i in T.serial(1, TY):
            for j in T.vectorized(N):
                if ty == 0:
                    idx: T.int32 = i * warp_elem + tx * thread_elem
                    shared_buf[idx + j] += shared_buf[i * warp_elem - 1]
        # Write sum of block to global memory
        for i in T.vectorized(N):
            idx: T.int32 = ty * warp_elem + tx * thread_elem + i
            if bx * block_elem + idx < cur_len:
                output[by, src_offset + bx * block_elem + idx] = shared_buf[idx]
        if tx == 0 and ty == 0:
            for i in T.vectorized(N):  # noqa: B007
                tmp_buf[by, tmp_offset + bx] = shared_buf[block_elem - 1]

    @T.macro
    def update_cross_block(
        batch: T.int32,
        cur_len: T.int32,
        source: T.Buffer,
        output: T.Buffer,
        src_offset: T.int32,
        out_offset: T.int32,
    ):
        bx = T.get_block_binding(0)
        by = T.get_block_binding(1)
        tx = T.get_thread_binding(0)
        ty = T.get_thread_binding(1)
        for i in T.serial(N):
            idx: T.int32 = bx * block_elem + ty * warp_elem + i * TX + tx
            if idx < cur_len:
                output[by, out_offset + idx] += T.if_then_else(bx > 0,
                                                               source[by, src_offset + bx - 1], 0)

    @T.prim_func
    def cumsum(A: T.Buffer((M, N), dtype="int32"), Out: T.Buffer((M, N), dtype="int32"),
               Tmp: T.Buffer((M, N), dtype="int32")):
        ceil_log2 = T.Cast("int32", T.ceil(T.log2(T.Cast("float32", N))))
        total_rounds = ceil_log2 // LOG_BLOCK_N

        with T.Kernel(T.ceildiv(N, block_elem), M, threads=[tx_len, ty_len]) as (bx, by):
            block_inclusive_inside_block(
                M, N, A, Out, Tmp, src_offset=T.int32(0), tmp_offset=T.int32(0))

        for i in range(total_rounds):
            cur_len = T.ceildiv(N, 1 << (LOG_BLOCK_N * (i + 1)))
            with T.Kernel(T.ceildiv(cur_len, block_elem), M) as (bx, by):
                block_inclusive_inside_block(
                    M,
                    cur_len,
                    Tmp,
                    Tmp,
                    Tmp,
                    src_offset=i * T.ceildiv(N, block_elem),
                    tmp_offset=(i + 1) * T.ceildiv(N, block_elem),
                )

        for i in range(total_rounds - 1):
            real_idx = total_rounds - 1 - i - 1
            cur_len = T.ceildiv(N, 1 << (LOG_BLOCK_N * (real_idx + 1)))
            with T.Kernel(T.ceildiv(cur_len, block_elem), M) as (bx, by):
                update_cross_block(
                    M,
                    cur_len,
                    Tmp,
                    Tmp,
                    src_offset=(real_idx + 1) * T.ceildiv(N, block_elem),
                    out_offset=real_idx * T.ceildiv(N, block_elem),
                )

        with T.Kernel(T.ceildiv(N, block_elem), M) as (bx, by):
            update_cross_block(M, N, Tmp, Out, src_offset=0, out_offset=0)

    return cumsum


def torch_cumsum(A: torch.Tensor, dim: int = -1):
    return torch.cumsum(A, dim=dim)


if __name__ == "__main__":

    M = 128
    N = 32
    program = gpu_2d_continuous_cumsum(M, N)
    kernel = tilelang.compile(program, execution_backend="dlpack", out_idx=[1])
    code = kernel.get_kernel_source()

    A = torch.randint(0, 10, (M, N)).cuda().to(torch.int32)
    tmp = torch.zeros_like(A).cuda().to(torch.int32)
    tilelang_output = kernel(A, tmp)
    torch_output = torch_cumsum(A).cuda().to(torch.int32)
    torch.testing.assert_close(tilelang_output, torch_output, atol=1e-2, rtol=1e-2)
