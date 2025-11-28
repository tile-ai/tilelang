# For intranode only
# This op is distributed
### TILELANG_USE_DISTRIBUTED=1 python dispatch.py

import os, sys
from torch.types import Number
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from argparse import ArgumentParser
from typing import Optional, Tuple
from tilelang.distributed.utils import init_dist
from utils import Config, create_moe_recv_counters, gen_inputs  # noqa: F403

from get_dispatch_layout import get_dispatch_layout

# tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def notify_dispatch_kernel(
    rank: int,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_channels: int,
    expert_alignment: int,
):
    threads = 128
    num_local_experts = num_experts // num_ranks
    num_warps = threads // 32

    @T.prim_func
    def notify_dispatch_main(
        num_tokens_per_rank: T.Tensor((num_ranks,), 'int32'),
        num_tokens_per_expert: T.Tensor((num_experts,), 'int32'),
        is_token_in_rank: T.Tensor((num_tokens, num_ranks), 'bool'),
        moe_recv_counter_mapped: T.Tensor((1,), 'int32'),
        moe_recv_expert_counter_mapped: T.Tensor((num_local_experts,), 'int32'),
        per_rank_buffer: T.Tensor((num_ranks, num_ranks), 'int32'),
        per_expert_buffer: T.Tensor((num_ranks, num_local_experts), 'int32'),
        barrier_signal: T.Tensor((num_ranks,), 'int32'),
        rank_prefix_matrix: T.Tensor((num_ranks, num_ranks), 'int32'),
        channel_prefix_matrix: T.Tensor((num_ranks, num_channels), 'int32'),
    ):
        with T.Kernel(num_ranks+1, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id, warp_id = tx % 32, tx // 32

            if bx == 0:
                # Barrier first 
                T.sync_blocks(barrier_signal)

                # `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
                # `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
                if tx < num_ranks:
                    T.st(per_rank_buffer[rank, tx], num_tokens_per_rank[tx], dst_pe=tx)
                    for i in T.serial(num_local_experts):
                        T.st(per_expert_buffer[rank, i], num_tokens_per_expert[tx * num_local_experts + i], dst_pe=tx)
                
                T.barrier_blocks(barrier_signal)

                # Sum per-rank cnts and pre-compute the prefix sum for data sending
                if tx < num_ranks:
                    for i in T.serial(1, num_ranks):
                        per_rank_buffer[i, tx] += per_rank_buffer[i-1, tx]
                    if tx == rank:
                        moe_recv_counter_mapped[0] = per_rank_buffer[num_ranks-1, rank]
                
                # Sum per-expert cnts
                if tx < num_local_experts:
                    sum = T.alloc_local([1], 'int32')
                    sum[0] = 0
                    for i in T.serial(0, num_ranks):
                        sum[0] += per_expert_buffer[i, tx]
                    sum[0] = T.ceildiv(sum[0], expert_alignment) * expert_alignment  # align up
                    moe_recv_expert_counter_mapped[tx] = sum[0]
                T.sync_threads()

                # Copy rank size prefix matrix to another tensor   
                # TODO: simply returns per_rank_buffer as rank_prefix_matrix             
                T.copy(per_rank_buffer, rank_prefix_matrix)

                # NOTE: We don't cleanup the buffer for later use
                T.barrier_blocks(barrier_signal)
            else:
                dst_rank = bx - 1
                for channel_id in T.serial(warp_id, num_channels, num_warps):
                    num_tokens_per_channel = T.ceildiv(num_tokens, num_channels)
                    token_start_idx = T.min(num_tokens_per_channel * channel_id, num_tokens)
                    token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_tokens)
                    cnt = T.alloc_var('int32')
                    cnt = 0
                    for i in T.serial(token_start_idx + lane_id, token_end_idx, 32):
                        cnt += is_token_in_rank[i, dst_rank]
                    cnt = T.warp_reduce_sum(cnt)
                    if T.elect_one_sync():
                        channel_prefix_matrix[dst_rank, channel_id] = cnt
                T.sync_threads()

                if tx == 0:
                    for i in T.serial(1, num_channels):
                        channel_prefix_matrix[dst_rank, i] += channel_prefix_matrix[dst_rank, i-1]

    return notify_dispatch_main


# TileScale notify-dispatch op
def notify_dispatch(
    # meta
    rank: int, 
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_channels: int,
    expert_alignment: int,
    # dispatch layout
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    # counter
    moe_recv_counter_mapped: torch.Tensor,
    moe_recv_expert_counter_mapped: torch.Tensor,
    # symm buffers
    per_rank_buffer: torch.Tensor,
    per_expert_buffer: torch.Tensor,
    barrier_signal: torch.Tensor,
    # allocator
    allocator,
):
    """
    TileScale notify-dispatch op.

    Args:
        rank (int): The current rank (process or device index).
        num_ranks (int): Total number of participating ranks (nodes).
        num_experts (int): Global number of experts in the MoE layer.
        num_tokens (int): Number of tokens being dispatched.
        num_channels (int): Number of communication channels.
        expert_alignment (int): Alignment constraint for expert buffer.

        num_tokens_per_rank (torch.Tensor): [num_ranks] 
            - For each rank r, num_tokens_per_rank[r] is the number of tokens assigned for dispatch to rank r across the cluster.
        num_tokens_per_expert (torch.Tensor): [num_experts] 
            - For each expert e, num_tokens_per_expert[e] is the number of tokens rank r will send to global expert e.
        is_token_in_rank (torch.Tensor): [num_tokens, num_ranks]
            - For each (token t, rank r), is_token_in_rank[t, r] indicates (bool) whether token t belongs to rank r after dispatch.
        
        moe_recv_counter_mapped (torch.Tensor): [1] 
            - The number of tokens received by the current rank from other ranks.
        moe_recv_expert_counter_mapped (torch.Tensor): [num_local_experts]
            - The number of tokens received by the current rank for its local experts.
        
        per_rank_buffer (torch.Tensor): num_ranks * [num_ranks, num_ranks], symm tensor, should be zeroed before use
            - Symmetric buffer for per-rank communication; [src_rank, dst_rank] region.
        per_expert_buffer (torch.Tensor): num_ranks * [num_ranks, num_local_experts], symm tensor, should be zeroed before use
            - Buffer for per-expert communication; [rank, local_expert] region.
        barrier_signal (torch.Tensor): num_ranks * [num_ranks], symm_tensor, should be zeroed before use
            - Synchronization tensor used as a system-wide barrier.

        allocator: TileScale allocator for symm tensors

    Returns
        rank_prefix_matrix (torch.Tensor): [num_ranks, num_ranks] 
            - For each (rank r, other_rank), rank_prefix_matrix[r, other_rank] records prefix sums/statistics for token dispatch between r and other_rank.
        channel_prefix_matrix (torch.Tensor): [num_ranks, num_channels]
            - For each (rank r, channel c), channel_prefix_matrix[r, c] records prefix sums/statistics for tokens on communication channel c for rank r.
    """
    kernel = notify_dispatch_kernel(
        rank,
        num_ranks,
        num_experts,
        num_tokens,
        num_channels,
        expert_alignment,
    )
    kernel.initialize(allocator=allocator)

    rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
    channel_prefix_matrix = torch.empty([num_ranks, num_channels], dtype=torch.int32, device='cuda')

    kernel(
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        moe_recv_counter_mapped,
        moe_recv_expert_counter_mapped,
        per_rank_buffer,
        per_expert_buffer,
        barrier_signal,
        rank_prefix_matrix,
        channel_prefix_matrix,
    )

    return rank_prefix_matrix, channel_prefix_matrix

# NOTE: We don't need cached_notify_dispatch, as per-rank-buffer is for one-time use


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # enable TMA later
        "tl.disable_warp_specialized": True})
def dispatch_kernel(
    rank, num_ranks, 
    num_tokens, 
    num_max_send_tokens,  # config.num_max_nvl_chunked_send_tokens
    num_recv_buffer_tokens,  # config.num_max_nvl_chunked_recv_tokens
    hidden, 
    num_topk, 
    num_experts,
    num_sms,
    dtype: str = 'bfloat16',
):
    threads = 768  # 24 warps
    TMABytesPerWarp = 8192
    smem_size = TMABytesPerWarp * threads // 32
    
    num_threads_per_rank = threads // num_ranks  # 96 (3 warps for each rank)
    num_channels = num_sms // 2  # 10 (2 SMs for each channel)
    num_local_experts = num_experts // num_ranks

    num_warps = threads // 32  # 24
    num_warps_per_rank = num_warps // num_ranks  # 3

    num_recv_tokens = T.dynamic('num_recv_tokens')

    @T.prim_func
    def dispatch_main(
        # output
        recv_x: T.Tensor((num_recv_tokens, hidden), dtype),
        recv_src_idx: T.Tensor((num_recv_tokens,), 'int32'),
        recv_topk_idx: T.Tensor((num_recv_tokens, num_topk), 'int64'),
        recv_topk_weights: T.Tensor((num_recv_tokens, num_topk), 'float'),
        recv_channel_offset: T.Tensor([num_ranks, num_channels], "int32"),
        send_head: T.Tensor([num_tokens, num_ranks], "int32"),
        # input
        x: T.Tensor([num_tokens, hidden], dtype),
        topk_idx: T.Tensor([num_tokens, num_topk], "int64"),
        topk_weights: T.Tensor([num_tokens, num_topk], "float32"),
        is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),
        rank_prefix_matrix: T.Tensor([num_ranks, num_ranks], "int32"),
        channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
        ###### below are symm buffers, one on each rank ######
        # channel buffer metadatas, stored on the receiver side
        # senders are responsible for tails, and receivers are responsible for heads
        channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
        # channel data buffers, stored on the receiver side
        channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden], dtype),
        channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
        channel_topk_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "int64"),
        channel_topk_weights_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "float32"),
        # channel_x_scales_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_scales], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id = tx % 32
            responsible_rank = tx // num_threads_per_rank
            responsible_channel = bx // 2

            if bx % 2 == 0:  # sender
                send_warp_id_in_rank = (tx % num_threads_per_rank) // 32

                # send offset by `-value-1` e.g. 0->-1, 1->-2
                # this is for distinguishing zero tokens
                if send_warp_id_in_rank == 0 and T.elect_one_sync():
                    value = T.alloc_var('int32')
                    value = T.if_then_else(
                        responsible_channel > 0,
                        channel_prefix_matrix[responsible_rank, responsible_channel - 1],
                        0)
                    T.st(channel_start_offset[responsible_channel, rank], -value-1, 
                        scope='sys', sem='relaxed', dst_pe=responsible_rank)
                    value = channel_prefix_matrix[responsible_rank, responsible_channel]
                    T.st(channel_end_offset[responsible_channel, rank], -value-1,
                        scope='sys', sem='relaxed', dst_pe=responsible_rank)
                T.sync_warp()

                # get task
                num_tokens_per_channel = T.alloc_var('int32', init=T.ceildiv(num_tokens, num_channels))
                token_start_idx = T.alloc_var('int32')
                token_start_idx = T.min(num_tokens_per_channel * responsible_channel, num_tokens)
                token_end_idx = T.alloc_var('int32')
                token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_tokens)
                
                # sender mainloop: iterate over all tokens and send by trunk
                cached_channel_tail_idx = T.alloc_var('int32')
                cached_channel_tail_idx = 0
                token_idx = T.alloc_var('int32')
                token_idx = token_start_idx
                with T.While(token_idx < token_end_idx):
                    if T.elect_one_sync():
                        T.wait_ge(channel_head_idx[responsible_channel, rank], 
                            num_max_send_tokens+cached_channel_tail_idx-num_recv_buffer_tokens,
                            responsible_rank)
                    T.sync_warp()

                    chunk_token_idx = T.alloc_var('int32')
                    chunk_token_idx = 0
                    while chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx:
                        # for the same token, the warp assigned to save `send_head` may be different from the warp 
                        # assigned to send the following data
                        if token_idx % num_warps_per_rank == send_warp_id_in_rank and T.elect_one_sync():
                            send_head[token_idx, responsible_rank] = T.if_then_else(
                                is_token_in_rank[token_idx, responsible_rank],
                                cached_channel_tail_idx,
                                -1
                            )
                        
                        # skip if not selected
                        if not is_token_in_rank[token_idx, responsible_rank]:
                            token_idx += 1
                            T.loop_continue()

                        # selected, get an empty slot
                        dst_slot_idx = T.alloc_var('int32')
                        dst_slot_idx = cached_channel_tail_idx % num_recv_buffer_tokens
                        cached_channel_tail_idx += 1
                        if cached_channel_tail_idx % num_warps_per_rank == send_warp_id_in_rank:
                            # copy data, all are remote copy
                            # 1. copy data (why useless???)
                            T.put_warp(T.address_of(x[token_idx, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=responsible_rank, unroll_factor=4)
                           
                            # 2. copy src idx
                            if T.elect_one_sync():
                                T.st(channel_src_idx_buffers[responsible_channel, rank, dst_slot_idx], token_idx,
                                    dst_pe=responsible_rank)

                            # 3. copy `topk_idx` and `topk_weights` with transformed index
                            if lane_id < num_topk:
                                # topk_idx
                                recv_expert_begin = responsible_rank * num_local_experts
                                recv_expert_end = recv_expert_begin + num_local_experts
                                
                                idx_value = T.alloc_var('int64')
                                T.ld(topk_idx[token_idx, lane_id], idx_value, nc=True)
                                idx_value = T.if_then_else(
                                    recv_expert_begin <= T.cast(idx_value, 'int32') < recv_expert_end,
                                    idx_value - recv_expert_begin,
                                    -1
                                )
                                T.st(channel_topk_idx_buffers[responsible_channel, rank, dst_slot_idx, lane_id], idx_value,
                                    dst_pe=responsible_rank)

                                # topk_weights
                                weight_value = T.alloc_var('float32')
                                T.ld(topk_weights[token_idx, lane_id], weight_value, nc=True)
                                weight_value = T.if_then_else(idx_value >= 0, weight_value, 0)
                                T.st(channel_topk_weights_buffers[responsible_channel, rank, dst_slot_idx, lane_id], weight_value,
                                    dst_pe=responsible_rank)

                            # 4. copy scale (support fp8 later)

                        chunk_token_idx += 1
                        token_idx += 1
                    
                    # move tail index
                    # here all warps should share the same new tail
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    if send_warp_id_in_rank == 0 and T.elect_one_sync():
                        T.st(channel_tail_idx[responsible_channel, rank], cached_channel_tail_idx,
                            scope='sys', sem='release',
                            dst_pe=responsible_rank)
                
            else:  # receiver
                recv_thread_id_in_rank = tx % num_threads_per_rank
                recv_warp_id_in_rank = recv_thread_id_in_rank // 32

                # calculate offset first
                rank_offset = T.if_then_else(responsible_rank > 0, rank_prefix_matrix[responsible_rank-1, rank], 0)

                # receive channel offset
                total_offset = T.alloc_var('int32')
                num_tokens_to_recv = T.alloc_var('int32')
                if T.elect_one_sync():
                    T.wait_ne(channel_start_offset[responsible_channel, responsible_rank], 0)
                    T.ld(channel_start_offset[responsible_channel, responsible_rank], total_offset, sem='volatile')
                    T.wait_ne(channel_end_offset[responsible_channel, responsible_rank], 0)
                    T.ld(channel_end_offset[responsible_channel, responsible_rank], num_tokens_to_recv, sem='volatile')
                    total_offset = -total_offset - 1
                    num_tokens_to_recv = -num_tokens_to_recv - 1
                    if recv_warp_id_in_rank == 0:
                        recv_channel_offset[responsible_rank, responsible_channel] = total_offset
                    num_tokens_to_recv -= total_offset
                total_offset = T.tvm_warp_shuffle(-1, total_offset, 0, 32, 32)
                total_offset += rank_offset
                num_tokens_to_recv = T.tvm_warp_shuffle(-1, num_tokens_to_recv, 0, 32, 32)

                # Shared tail indices for different warps
                shared_channel_tail_idx = T.alloc_shared([num_ranks], 'int32')

                cached_channel_head_idx = T.alloc_var('int32') 
                cached_channel_head_idx = 0
                cached_channel_tail_idx = T.alloc_var('int32')
                cached_channel_tail_idx = 0
                with T.While(num_tokens_to_recv > 0):
                    with T.While(recv_thread_id_in_rank == 0):
                        T.ld(channel_tail_idx[responsible_channel, responsible_rank], cached_channel_tail_idx, sem='acquire', scope='sys')
                        
                        # read to copy
                        if cached_channel_head_idx != cached_channel_tail_idx:
                            shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx 
                            T.loop_break()

                    # sync queue tail
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank]

                    # copy data
                    # 1. recv x
                    num_cur_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx
                    for chunk_idx in T.serial(recv_warp_id_in_rank, num_cur_recv_tokens, num_warps_per_rank):
                        token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens
                        # T.copy(channel_x_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, :], recv_x[total_offset+chunk_idx, :])  # todo: add ld_nc and st_na
                        #! T.copy will cause layout inference error
                        T.put_warp(T.address_of(channel_x_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, 0]),
                            T.address_of(recv_x[total_offset+chunk_idx, 0]),
                            hidden,
                            -1, 
                            5)
                    
                    # 2. recv src_idx
                    for chunk_idx in T.serial(cached_channel_head_idx+recv_thread_id_in_rank,
                        cached_channel_tail_idx,
                        num_threads_per_rank):
                        local_src_idx = T.alloc_var('int32')
                        T.ld(channel_src_idx_buffers[responsible_channel, responsible_rank, chunk_idx % num_recv_buffer_tokens], local_src_idx, nc=True)
                        recv_src_idx[total_offset+chunk_idx-cached_channel_head_idx] = local_src_idx
                        
                    # 3. recv topk_idx and topk_weights
                    for idx in T.serial(recv_thread_id_in_rank, num_cur_recv_tokens*num_topk, num_threads_per_rank):
                        chunk_idx = idx // num_topk
                        token_topk_idx = idx % num_topk
                        token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens
                        recv_topk_idx[total_offset+chunk_idx, token_topk_idx] = channel_topk_idx_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, token_topk_idx]
                        recv_topk_weights[total_offset+chunk_idx, token_topk_idx] = channel_topk_weights_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, token_topk_idx]

                    # 4. recv scale (support fp8 later)

                    # Move queue
                    cached_channel_head_idx += num_cur_recv_tokens
                    total_offset += num_cur_recv_tokens
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    if recv_warp_id_in_rank == num_warps_per_rank - 1 and T.elect_one_sync():
                        T.st(channel_head_idx[responsible_channel, responsible_rank], cached_channel_head_idx,
                            scope='sys', sem='relaxed')
                    
                    # Exit
                    num_tokens_to_recv -= num_cur_recv_tokens
                    
    return dispatch_main


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # enable TMA later
        "tl.disable_warp_specialized": True})
def cached_dispatch_kernel(
    rank, num_ranks, 
    num_tokens, 
    num_max_send_tokens,  # config.num_max_nvl_chunked_send_tokens
    num_recv_buffer_tokens,  # config.num_max_nvl_chunked_recv_tokens
    hidden, 
    num_topk, 
    num_experts,
    num_sms,
    dtype: str = 'bfloat16',
):
    threads = 768  # 24 warps
    TMABytesPerWarp = 8192
    smem_size = TMABytesPerWarp * threads // 32
    
    num_threads_per_rank = threads // num_ranks  # 96 (3 warps for each rank)
    num_channels = num_sms // 2  # 10 (2 SMs for each channel)
    num_local_experts = num_experts // num_ranks

    num_warps = threads // 32  # 24
    num_warps_per_rank = num_warps // num_ranks  # 3

    num_recv_tokens = T.dynamic('num_recv_tokens')

    @T.prim_func
    def cached_dispatch_main(
        # output
        recv_x: T.Tensor((num_recv_tokens, hidden), dtype),
        recv_src_idx: T.Tensor((num_recv_tokens,), 'int32'),
        recv_channel_offset: T.Tensor([num_ranks, num_channels], "int32"),
        send_head: T.Tensor([num_tokens, num_ranks], "int32"),
        # input
        x: T.Tensor([num_tokens, hidden], dtype),
        is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),
        rank_prefix_matrix: T.Tensor([num_ranks, num_ranks], "int32"),
        channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
        ###### below are symm buffers, one on each rank ######
        # channel buffer metadatas, stored on the receiver side
        # senders are responsible for tails, and receivers are responsible for heads
        channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
        # channel data buffers, stored on the receiver side
        channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden], dtype),
        channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
        # channel_x_scales_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_scales], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id = tx % 32
            responsible_rank = tx // num_threads_per_rank
            responsible_channel = bx // 2

            if bx % 2 == 0:  # sender
                send_warp_id_in_rank = (tx % num_threads_per_rank) // 32

                # send offset by `-value-1` e.g. 0->-1, 1->-2
                # this is for distinguishing zero tokens
                if send_warp_id_in_rank == 0 and T.elect_one_sync():
                    value = T.alloc_var('int32')
                    value = T.if_then_else(
                        responsible_channel > 0,
                        channel_prefix_matrix[responsible_rank, responsible_channel - 1],
                        0)
                    T.st(channel_start_offset[responsible_channel, rank], -value-1, 
                        scope='sys', sem='relaxed', dst_pe=responsible_rank)
                    value = channel_prefix_matrix[responsible_rank, responsible_channel]
                    T.st(channel_end_offset[responsible_channel, rank], -value-1,
                        scope='sys', sem='relaxed', dst_pe=responsible_rank)
                T.sync_warp()

                # get task
                num_tokens_per_channel = T.alloc_var('int32', init=T.ceildiv(num_tokens, num_channels))
                token_start_idx = T.alloc_var('int32')
                token_start_idx = T.min(num_tokens_per_channel * responsible_channel, num_tokens)
                token_end_idx = T.alloc_var('int32')
                token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_tokens)
                
                # sender mainloop: iterate over all tokens and send by trunk
                cached_channel_tail_idx = T.alloc_var('int32')
                cached_channel_tail_idx = 0
                token_idx = T.alloc_var('int32')
                token_idx = token_start_idx
                with T.While(token_idx < token_end_idx):
                    if T.elect_one_sync():
                        T.wait_ge(channel_head_idx[responsible_channel, rank], 
                            num_max_send_tokens+cached_channel_tail_idx-num_recv_buffer_tokens,
                            responsible_rank)
                    T.sync_warp()

                    chunk_token_idx = T.alloc_var('int32')
                    chunk_token_idx = 0
                    while chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx:
                        # for the same token, the warp assigned to save `send_head` may be different from the warp 
                        # assigned to send the following data
                        if token_idx % num_warps_per_rank == send_warp_id_in_rank and T.elect_one_sync():
                            send_head[token_idx, responsible_rank] = T.if_then_else(
                                is_token_in_rank[token_idx, responsible_rank],
                                cached_channel_tail_idx,
                                -1
                            )
                        
                        # skip if not selected
                        if not is_token_in_rank[token_idx, responsible_rank]:
                            token_idx += 1
                            T.loop_continue()

                        # selected, get an empty slot
                        dst_slot_idx = T.alloc_var('int32')
                        dst_slot_idx = cached_channel_tail_idx % num_recv_buffer_tokens
                        cached_channel_tail_idx += 1
                        if cached_channel_tail_idx % num_warps_per_rank == send_warp_id_in_rank:
                            # copy data, all are remote copy
                            # 1. copy data (why useless???)
                            T.put_warp(T.address_of(x[token_idx, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=responsible_rank, unroll_factor=4)
                        
                            # 2. copy src idx
                            if T.elect_one_sync():
                                T.st(channel_src_idx_buffers[responsible_channel, rank, dst_slot_idx], token_idx,
                                    dst_pe=responsible_rank)

                            # 4. copy scale (support fp8 later)

                        chunk_token_idx += 1
                        token_idx += 1
                    
                    # move tail index
                    # here all warps should share the same new tail
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    if send_warp_id_in_rank == 0 and T.elect_one_sync():
                        T.st(channel_tail_idx[responsible_channel, rank], cached_channel_tail_idx,
                            scope='sys', sem='release',
                            dst_pe=responsible_rank)
                
            else:  # receiver
                recv_thread_id_in_rank = tx % num_threads_per_rank
                recv_warp_id_in_rank = recv_thread_id_in_rank // 32

                # calculate offset first
                rank_offset = T.if_then_else(responsible_rank > 0, rank_prefix_matrix[responsible_rank-1, rank], 0)

                # receive channel offset
                total_offset = T.alloc_var('int32')
                num_tokens_to_recv = T.alloc_var('int32')
                if T.elect_one_sync():
                    T.wait_ne(channel_start_offset[responsible_channel, responsible_rank], 0)
                    T.ld(channel_start_offset[responsible_channel, responsible_rank], total_offset, sem='volatile')
                    T.wait_ne(channel_end_offset[responsible_channel, responsible_rank], 0)
                    T.ld(channel_end_offset[responsible_channel, responsible_rank], num_tokens_to_recv, sem='volatile')
                    total_offset = -total_offset - 1
                    num_tokens_to_recv = -num_tokens_to_recv - 1
                    if recv_warp_id_in_rank == 0:
                        recv_channel_offset[responsible_rank, responsible_channel] = total_offset
                    num_tokens_to_recv -= total_offset
                total_offset = T.tvm_warp_shuffle(-1, total_offset, 0, 32, 32)
                total_offset += rank_offset
                num_tokens_to_recv = T.tvm_warp_shuffle(-1, num_tokens_to_recv, 0, 32, 32)

                # Shared tail indices for different warps
                shared_channel_tail_idx = T.alloc_shared([num_ranks], 'int32')

                cached_channel_head_idx = T.alloc_var('int32') 
                cached_channel_head_idx = 0
                cached_channel_tail_idx = T.alloc_var('int32')
                cached_channel_tail_idx = 0
                with T.While(num_tokens_to_recv > 0):
                    with T.While(recv_thread_id_in_rank == 0):
                        T.ld(channel_tail_idx[responsible_channel, responsible_rank], cached_channel_tail_idx, sem='acquire', scope='sys')
                        
                        # read to copy
                        if cached_channel_head_idx != cached_channel_tail_idx:
                            shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx 
                            T.loop_break()

                    # sync queue tail
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank]

                    # copy data
                    # 1. recv x
                    num_cur_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx
                    for chunk_idx in T.serial(recv_warp_id_in_rank, num_cur_recv_tokens, num_warps_per_rank):
                        token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens
                        # T.copy(channel_x_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, :], recv_x[total_offset+chunk_idx, :])  # todo: add ld_nc and st_na
                        #! T.copy will cause layout inference error
                        T.put_warp(T.address_of(channel_x_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, 0]),
                            T.address_of(recv_x[total_offset+chunk_idx, 0]),
                            hidden,
                            -1, 
                            5)
                    
                    # 2. recv src_idx
                    for chunk_idx in T.serial(cached_channel_head_idx+recv_thread_id_in_rank,
                        cached_channel_tail_idx,
                        num_threads_per_rank):
                        local_src_idx = T.alloc_var('int32')
                        T.ld(channel_src_idx_buffers[responsible_channel, responsible_rank, chunk_idx % num_recv_buffer_tokens], local_src_idx, nc=True)
                        recv_src_idx[total_offset+chunk_idx-cached_channel_head_idx] = local_src_idx

                    # 4. recv scale (support fp8 later)

                    # Move queue
                    cached_channel_head_idx += num_cur_recv_tokens
                    total_offset += num_cur_recv_tokens
                    T.sync_threads(responsible_rank, num_threads_per_rank)
                    if recv_warp_id_in_rank == num_warps_per_rank - 1 and T.elect_one_sync():
                        T.st(channel_head_idx[responsible_channel, responsible_rank], cached_channel_head_idx,
                            scope='sys', sem='relaxed')
                    
                    # Exit
                    num_tokens_to_recv -= num_cur_recv_tokens
                    
            # todo: support num_worst_tokens > 0 later
            
    return cached_dispatch_main


# todo: support cached-mode via handle
def intranode_dispatch(
    rank: int,
    allocator,
    # data
    x: torch.Tensor,  # todo: support fp8 quant
    # handle
    handle: Optional[Tuple] = None,
    # meta
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    is_token_in_rank: Optional[torch.Tensor] = None,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    expert_alignment: int = 1,
    # todo: support num_worst_tokens
    # tuning cfg
    config: Optional[Config] = None,
    # todo: support async functionality
    
):
    """
    Dispatch tokens to different intranode ranks.
    Intranode kernels require all the ranks should be visible via NVLink.

    Arguments:
        x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
            and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
            `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                (requiring divisible) with type `torch.float`.
        num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
        is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
        num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            Returns None for cached-mode.
        topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices
            selected by each token, `-1` means no selections.
        topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        config: the performance tuning config.
        allocator: TileScale allocator for symm tensors

    Returns:
        recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
            received token count.
        recv_topk_idx: received expert indices.
        recv_topk_weights: received expert weights.
        num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
            each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
            will be empty.
        handle: the handle for combine, has `(rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)`.
    """

    if handle is None:
        assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None, \
        "num_tokens_per_rank, is_token_in_rank, and num_tokens_per_expert must be provided in non-cached mode"
    else:
        rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle

    num_tokens, hidden = x.shape
    num_experts = num_tokens_per_expert.shape[0] if handle is None else 0
    num_ranks = num_tokens_per_rank.shape[0]
    num_local_experts = num_experts // num_ranks
    num_topk = topk_idx.shape[1] if handle is None else 0

    # Default config
    config = Config.get_dispatch_config(num_ranks) if config is None else config

    # Alloc public barrier
    barrier_signal = tilelang.tensor((num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()

    if handle is None:
        # Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        # Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
        channel_prefix_matrix = torch.empty([num_ranks, config.num_channels], dtype=torch.int32, device='cuda')

        moe_recv_counter_mapped, moe_recv_expert_counter_mapped = create_moe_recv_counters(num_ranks, num_experts // num_ranks)[3:5]

        per_rank_buffer = tilelang.tensor((num_ranks, num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()
        per_expert_buffer = tilelang.tensor((num_ranks, num_local_experts), dtype=torch.int32, device='cuda', allocator=allocator).zero_()        

        rank_prefix_matrix, channel_prefix_matrix = notify_dispatch(
            rank,
            num_ranks,
            num_experts,
            num_tokens,
            config.num_channels,
            expert_alignment,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            moe_recv_counter_mapped,
            moe_recv_expert_counter_mapped,
            per_rank_buffer,
            per_expert_buffer,
            barrier_signal,
            allocator,
        )
        torch.cuda.synchronize()  # todo: replace it with host-side wait_ne

        num_recv_tokens = moe_recv_counter_mapped.item()
        num_recv_tokens_per_expert_list = moe_recv_expert_counter_mapped.tolist()
    else:
        num_recv_tokens = recv_src_idx.size(0)
        num_recv_tokens_per_expert_list = None

    # create normal buffers
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device='cuda')
    recv_src_idx = torch.empty((num_recv_tokens,), dtype=torch.int32, device='cuda')
    if handle is None:
        recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=torch.int64, device='cuda')
        recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device='cuda')
    recv_channel_prefix_matrix = torch.empty((num_ranks, config.num_channels), dtype=torch.int32, device='cuda')
    send_head = torch.empty((num_tokens, num_ranks), dtype=torch.int32, device='cuda')

    # create symm buffers
    channel_start_offset = tilelang.tensor(
        [config.num_channels, num_ranks], dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    channel_end_offset = tilelang.tensor(
        [config.num_channels, num_ranks], dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    channel_head_idx = tilelang.tensor(
        [config.num_channels, num_ranks], dtype=torch.int32, device='cuda', allocator=allocator).zero_()    
    channel_tail_idx = tilelang.tensor(
        shape=[config.num_channels, num_ranks], dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    channel_x_buffers = tilelang.tensor(
        [config.num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens, hidden], dtype=torch.bfloat16, device='cuda', allocator=allocator)
    channel_src_idx_buffers = tilelang.tensor(
        [config.num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens], dtype=torch.int32, device='cuda', allocator=allocator)
    
    if handle is None:
        channel_topk_idx_buffers = tilelang.tensor(
            [config.num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens, num_topk], dtype=torch.int64, device='cuda', allocator=allocator)
        channel_topk_weights_buffers = tilelang.tensor(
            [config.num_channels, num_ranks, config.num_max_nvl_chunked_recv_tokens, num_topk], dtype=torch.float32, device='cuda', allocator=allocator)
    else: 
        channel_topk_idx_buffers = None  # todo: double-check this (may affect combine)
        channel_topk_weights_buffers = None

    # get dispatch 
    _kernel = dispatch_kernel if handle is None else cached_dispatch_kernel
    kernel = _kernel(
        rank, 
        num_ranks,
        num_tokens,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        hidden,
        num_topk,
        num_experts,
        config.num_sms,
        'bfloat16'
    )
    kernel.initialize(allocator=allocator)
    
    # run dispatch
    if rank == 0:
        print('Start running dispatch kernel...')
    if handle is None:
        args = (recv_x, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_prefix_matrix, send_head, x, topk_idx, topk_weights, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers, channel_topk_idx_buffers, channel_topk_weights_buffers)
    else:
        args = (recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head, x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers)
    kernel(*args)

    handle = (rank_prefix_matrix, channel_prefix_matrix, 
        recv_channel_prefix_matrix, recv_src_idx,
        is_token_in_rank, send_head
    )
    symm_buffers = (channel_head_idx, channel_tail_idx, barrier_signal, channel_x_buffers, channel_src_idx_buffers, channel_topk_weights_buffers)
    
    if handle is not None:
        recv_topk_idx = recv_topk_weights = None
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, symm_buffers  # todo: reconsider hierachy


def test_intranode_dispatch(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
    num_ranks: int,
    expert_alignment: int,
    cached: bool,
    group: torch.distributed.ProcessGroup,
):
    try: 
        import deep_ep  # noqa: F403
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install DeepEP to run this test.")

    allocator = tilelang.get_allocator(
        size=2**30,
        device="cuda",
        is_distributed=True,
        local_rank=rank,
        num_local_ranks=num_ranks,
        group=group)

    x, topk_idx, topk_weights, rank_idx = gen_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks)
    buffer = deep_ep.Buffer(group, num_nvl_bytes=2**30)

    if rank == 0: 
        print(f'get dispatch layout ...')
    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank = get_dispatch_layout(topk_idx, num_experts, num_ranks)
    assert torch.equal(num_tokens_per_expert, ref_num_tokens_per_expert), \
        f"num_tokens_per_expert mismatch, max err: {(num_tokens_per_expert - ref_num_tokens_per_expert).abs().max()}"
    assert torch.equal(is_token_in_rank, ref_is_token_in_rank), \
        "is_token_in_rank mismatch"
    assert torch.equal(num_tokens_per_rank, ref_num_tokens_per_rank), \
        f"num_tokens_per_rank mismatch, max err: {(num_tokens_per_rank - ref_num_tokens_per_rank).abs().max()}"

    if rank == 0: 
        print('notify dispatch and intranode dispatch ...')
    
    # golden
    ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_num_recv_tokens_per_expert_list, ref_handle, _ = \
        buffer.dispatch(x, None, ref_num_tokens_per_rank, None, ref_is_token_in_rank, ref_num_tokens_per_expert, topk_idx, topk_weights, expert_alignment)

    # ours
    if cached:
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
            intranode_dispatch(rank, allocator, x, ref_handle, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, None, None, expert_alignment, None)
    else:
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
            intranode_dispatch(rank, allocator, x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, None)

    # check dispatch output
    assert torch.equal(recv_x, ref_recv_x), f'recv_x mismatch, max err: {(recv_x - ref_recv_x).abs().max()}'
    if not cached:
        assert torch.equal(recv_topk_idx, ref_recv_topk_idx), f'recv_topk_idx mismatch, max err: {(recv_topk_idx - ref_recv_topk_idx).abs().max()}'
        assert torch.equal(recv_topk_weights, ref_recv_topk_weights), f'recv_topk_weights mismatch, max err: {(recv_topk_weights - ref_recv_topk_weights).abs().max()}'
        assert num_recv_tokens_per_expert_list == ref_num_recv_tokens_per_expert_list, 'num_recv_tokens_per_expert_list mismatch'
    
    # check handle
    if not cached:
        rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
        ref_rank_prefix_matrix, ref_channel_prefix_matrix, ref_recv_channel_prefix_matrix, ref_recv_src_idx, ref_is_token_in_rank, ref_send_head = ref_handle
        assert torch.equal(rank_prefix_matrix, ref_rank_prefix_matrix), f'rank_prefix_matrix mismatch, max err: {(rank_prefix_matrix - ref_rank_prefix_matrix).abs().max()}'
        assert torch.equal(channel_prefix_matrix, ref_channel_prefix_matrix), f'channel_prefix_matrix mismatch, max err: {(channel_prefix_matrix - ref_channel_prefix_matrix).abs().max()}'
        assert torch.equal(recv_channel_prefix_matrix, ref_recv_channel_prefix_matrix), f'recv_channel_prefix_matrix mismatch, max err: {(recv_channel_prefix_matrix - ref_recv_channel_prefix_matrix).abs().max()}'
        assert torch.equal(recv_src_idx, ref_recv_src_idx), f'recv_src_idx mismatch, max err: {(recv_src_idx - ref_recv_src_idx).abs().max()}'
        assert torch.equal(is_token_in_rank, ref_is_token_in_rank), f'is_token_in_rank mismatch, max err: {(is_token_in_rank - ref_is_token_in_rank).abs().max()}'
        assert torch.equal(send_head, ref_send_head), f'send_head mismatch, max err: {(send_head - ref_send_head).abs().max()}'
    
    print(f'[rank {rank}] All checks passed for {'cached' if cached else 'non-cached'} TileScale intranode_dispatch. ✅')


def main(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_intranode_dispatch(
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        rank,
        num_ranks,
        args.expert_alignment,
        args.cached,
        group,
    )

    torch.distributed.destroy_process_group(group)
    torch.distributed.destroy_process_group()


def parse_args():
    parser = ArgumentParser(description="Test dispatch")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden size")
    parser.add_argument("--num_topk", type=int, default=8, help="Number of top-k experts to select for each token")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--expert_alignment", type=int, default=1, help="Expert alignment")
    parser.add_argument("-cached", action="store_true", default=False, help="Use cached mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_ranks = args.num_ranks
    torch.multiprocessing.spawn(main, args=(num_ranks, args), nprocs=num_ranks)