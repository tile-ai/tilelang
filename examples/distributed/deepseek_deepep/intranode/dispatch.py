# For intranode only
# This op is distributed

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from typing import Optional, Tuple
from utils import Config, ep_ext  # noqa: F403

tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


# notify_dispatch is responible for:
# 1. Pre-compute rank/channel prefix for dispatch
# 2. Zero 4 symm buffers before a system-level barrier
@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def notify_dispatch_kernel(
    rank: int,
    num_ranks: int,
    num_experts: int,
    num_channels: int,
    expert_alignment: int,
):
    threads = 128
    num_local_experts = num_experts // num_ranks
    num_warps = threads // 32

    num_tokens = T.dynamic('num_tokens')

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
        # 4 symm buffers to be zeroed
        channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
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

                # Clear 4 symm buffers  for later use 
                T.clear(channel_start_offset)
                T.clear(channel_end_offset)
                T.clear(channel_head_idx)
                T.clear(channel_tail_idx)

                T.barrier_blocks(barrier_signal)
            else:
                dst_rank = bx - 1
                for channel_id in T.serial(warp_id, num_channels, num_warps):
                    num_tokens_per_channel = T.truncdiv(num_tokens+num_channels-1, num_channels)
                    # todo: this is a workaround, as TVM has a bug when calculating safe ceildiv for tir.Var
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
    moe_recv_counter: torch.Tensor,
    moe_recv_expert_counter: torch.Tensor,
    moe_recv_counter_mapped: torch.Tensor,
    moe_recv_expert_counter_mapped: torch.Tensor,
    # symm buffers
    per_rank_buffer: torch.Tensor,
    per_expert_buffer: torch.Tensor,
    barrier_signal: torch.Tensor,
    channel_start_offset: torch.Tensor,
    channel_end_offset: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    # allocator
    allocator,
    comm_stream=None,
):
    kernel = notify_dispatch_kernel(
        rank,
        num_ranks,
        num_experts,
        num_channels,
        expert_alignment,
    )
    kernel.initialize(allocator=allocator, stream=comm_stream.cuda_stream)

    rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
    channel_prefix_matrix = torch.empty([num_ranks, num_channels], dtype=torch.int32, device='cuda')

    # clear buffers and counters
    moe_recv_counter.fill_(-1)
    moe_recv_expert_counter.fill_(-1)

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
        channel_start_offset,
        channel_end_offset,
        channel_head_idx,
        channel_tail_idx,
        stream=comm_stream.cuda_stream,
        skip_tensor_validation=True  # reduce runtime overhead
    )

    num_recv_tokens, num_recv_tokens_per_expert_list = ep_ext.wait_for_counters_ready(moe_recv_counter, moe_recv_expert_counter)
    return num_recv_tokens, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix


# cached_notify_dispatch only needs to clear symm buffers
@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def cached_notify_dispatch_kernel(num_ranks: int, num_channels: int):
    @T.prim_func
    def cached_notify_dispatch_main(
        barrier_signal: T.Tensor((num_ranks,), 'int32'),
        # 4 symm buffers to be zeroed
        channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
    ):
        with T.Kernel(1, threads=128):
            T.sync_blocks(barrier_signal)

            T.clear(channel_start_offset)
            T.clear(channel_end_offset)
            T.clear(channel_head_idx)
            T.clear(channel_tail_idx)

            T.barrier_blocks(barrier_signal)

    return cached_notify_dispatch_main


def cached_notify_dispatch(
    num_ranks: int,
    num_channels: int,
    # symm buffers to be cleared
    channel_start_offset: torch.Tensor,
    channel_end_offset: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    # barrier
    barrier_signal: torch.Tensor,
    # allocator
    allocator,
    comm_stream=None,
):
    kernel = cached_notify_dispatch_kernel(num_ranks, num_channels)
    kernel.initialize(allocator=allocator, stream=comm_stream.cuda_stream)  # we still comm on barrier_signal
    kernel(barrier_signal, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, stream=comm_stream.cuda_stream,
        skip_tensor_validation=True) # reduce runtime overhead


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # enable TMA later
        "tl.disable_warp_specialized": True})
def dispatch_kernel(
    rank, 
    num_ranks,
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

    num_tokens = T.dynamic('num_tokens')
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
                num_tokens_per_channel = T.truncdiv(num_tokens+num_channels-1, num_channels) 
                # todo: this is a workaround, as TVM has a bug when calculating safe ceildiv for tir.Var
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
                            # 1. copy data
                            T.put_warp(T.address_of(x[token_idx, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=responsible_rank, unroll_factor=4, enable_aggresive_vectorize=True)
                           
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
                            5,
                            enable_aggresive_vectorize=True)
                    
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
                            # 1. copy data
                            T.put_warp(T.address_of(x[token_idx, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=responsible_rank, unroll_factor=4, enable_aggresive_vectorize=True)
                        
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
                        #! T.copy will cause layout inference error
                        T.put_warp(T.address_of(channel_x_buffers[responsible_channel, responsible_rank, token_idx_in_buffer, 0]),
                            T.address_of(recv_x[total_offset+chunk_idx, 0]),
                            hidden,
                            -1, 
                            5,
                            enable_aggresive_vectorize=True)
                    
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


def intranode_dispatch(
    rank: int,
    allocator,
    symm_buffers,
    moe_recv_counter,
    moe_recv_expert_counter,
    moe_recv_counter_mapped,
    moe_recv_expert_counter_mapped,
    x: torch.Tensor,  # todo: support fp8 quant
    config: Config,
    handle: Optional[Tuple] = None,
    num_tokens_per_rank: Optional[torch.Tensor] = None,
    is_token_in_rank: Optional[torch.Tensor] = None,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    topk_idx: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    expert_alignment: int = 1,
    kernel = None,
    comm_stream = None,
    # todo: support num_worst_tokens
    # todo: support async functionality
):
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

    barrier_signal, per_rank_buffer, per_expert_buffer, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, \
        channel_x_buffers, channel_src_idx_buffers, channel_topk_idx_buffers, channel_topk_weights_buffers = symm_buffers

    if handle is None:
        num_recv_tokens, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix = notify_dispatch(
            rank,
            num_ranks,
            num_experts,
            num_tokens,
            config.num_channels,
            expert_alignment,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            moe_recv_counter,
            moe_recv_expert_counter,
            moe_recv_counter_mapped,
            moe_recv_expert_counter_mapped,
            per_rank_buffer,
            per_expert_buffer,
            barrier_signal,
            channel_start_offset,
            channel_end_offset,
            channel_head_idx,
            channel_tail_idx,
            allocator,
            comm_stream=comm_stream,
        )
    else:
        cached_notify_dispatch(num_ranks, config.num_channels, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, barrier_signal, allocator, comm_stream=comm_stream)
        num_recv_tokens = recv_src_idx.size(0)

    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device='cuda')
    recv_src_idx = torch.empty((num_recv_tokens,), dtype=torch.int32, device='cuda')
    if handle is None:
        recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=torch.int64, device='cuda')
        recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device='cuda')
    recv_channel_prefix_matrix = torch.empty((num_ranks, config.num_channels), dtype=torch.int32, device='cuda')
    send_head = torch.empty((num_tokens, num_ranks), dtype=torch.int32, device='cuda')

    # run dispatch
    if handle is None:
        # kernel = dispatch_kernel(rank, num_ranks, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, hidden, num_topk, num_experts, config.num_sms, 'bfloat16')
        # kernel.initialize(allocator=allocator)
        kernel(recv_x, recv_src_idx, recv_topk_idx, recv_topk_weights, recv_channel_prefix_matrix, send_head, x, topk_idx, topk_weights, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers, channel_topk_idx_buffers, channel_topk_weights_buffers, stream=comm_stream.cuda_stream,
            skip_tensor_validation=True) # reduce runtime overhead
        handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle
    else:
        kernel = cached_dispatch_kernel(rank, num_ranks, num_tokens, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, hidden, num_topk, num_experts, config.num_sms, 'bfloat16')
        kernel.initialize(allocator=allocator, stream=comm_stream.cuda_stream)
        kernel(recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head, x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, channel_start_offset, channel_end_offset, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers, stream=comm_stream.cuda_stream,
            skip_tensor_validation=True) # reduce runtime overhead
        return recv_x
