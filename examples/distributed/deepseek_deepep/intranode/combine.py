# For intranode only
# This op is distributed

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T

# tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def cached_notify_combine_kernel(
    num_ranks,
    num_sms,
):
    num_channels = num_sms // 2
    threads = max(128, 32 * num_ranks)

    num_recv_tokens = T.dynamic('num_recv_tokens')

    @T.prim_func
    def cached_notify_combine_main(
        send_head: T.Tensor([num_recv_tokens, num_ranks], "int32"),
        ##### symm buffers #####
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
        barrier_signal: T.Tensor((num_ranks,), 'int32'),
    ):
        with T.Kernel(num_channels + 1, threads=threads) as bx:
            tx = T.get_thread_binding()

            if bx == 0:  # clearing channel_head/tail_idx buffers
                T.sync_blocks(barrier_signal)
                T.clear(channel_head_idx)
                T.clear(channel_tail_idx)
                T.barrier_blocks(barrier_signal)
            else:  # calculate send_head
                channel_id = bx - 1
                rank_id = tx // 32
                lane_id = tx % 32
                if rank_id >= num_ranks:
                    T.thread_return()

                tokens_per_channel = T.ceildiv(num_recv_tokens, num_channels)
                token_start_idx = T.min(tokens_per_channel * channel_id, num_recv_tokens)
                token_end_idx = T.min(token_start_idx + tokens_per_channel, num_recv_tokens)
               
                last_head = T.alloc_var('int32', init=2**25)  # a heuristic large number
                # todo: tilelang doesn't support reverse loop, we simulate this
                for i in T.serial(0, token_end_idx-token_start_idx, 32):
                    token_idx_tail = token_end_idx - i - 1
                    token_idx = token_idx_tail - lane_id
                    current_head = T.alloc_var('int32')
                    if token_idx >= token_start_idx:
                        T.ld(send_head[token_idx, rank_id], current_head, nc=True)
                    else:
                        current_head = -1
                    expected_head = T.alloc_var('int32')
                    expected_head = 0
                    for j in T.serial(T.min(32, token_idx_tail-token_start_idx + 1)):
                        head = T.tvm_warp_shuffle(-1, current_head, j, 32, 32)
                        if head < 0:
                            if lane_id == j:
                                expected_head = -last_head - 1
                        else:
                            last_head = head
                    if current_head < 0 and token_idx >= token_start_idx:
                        send_head[token_idx, rank_id] = expected_head
                
    return cached_notify_combine_main


def cached_notify_combine(
    num_ranks,
    num_sms,
    ##### symm buffers #####
    send_head: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    barrier_signal: torch.Tensor,    
    allocator
):
    kernel = cached_notify_combine_kernel(num_ranks, num_sms)
    kernel.initialize(allocator=allocator)

    kernel(send_head, channel_head_idx, channel_tail_idx, barrier_signal)


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # use TMA later
        "tl.disable_warp_specialized": True})
def combine_kernel(
    rank, 
    num_ranks,
    num_max_send_tokens,  # config.num_max_nvl_chunked_send_tokens
    num_recv_buffer_tokens,  # config.num_max_nvl_chunked_recv_tokens
    hidden, 
    num_topk, 
    num_sms,
    dtype: str = 'bfloat16',
):
    num_tokens = T.dynamic('num_tokens')
    num_recv_tokens = T.dynamic('num_recv_tokens')

    num_channels = num_sms // 2
    threads = 768  # 24 warps
    warps = threads // 32
    warps_per_rank = warps // num_ranks  # 3
    threads_per_rank = threads // num_ranks  # 96
    TMABytesPerWarp = 4096
    smem_size = TMABytesPerWarp * (threads // 32)
    num_stages = 8

    assert hidden % 8 == 0  # manual vectorize on recv-side

    @T.prim_func
    def combine_main(
        # inputs
        x: T.Tensor([num_tokens, hidden], dtype),
        topk_weights: T.Tensor([num_tokens, num_topk], "float32"),
        src_idx: T.Tensor([num_tokens], "int32"),
        # todo: support bias as inputs
        # outputs
        recv_x: T.Tensor([num_recv_tokens, hidden], dtype),
        recv_topk_weights: T.Tensor([num_recv_tokens, num_topk], "float32"),
        # metadata
        rank_prefix_matrix: T.Tensor([num_ranks, num_ranks], "int32"),
        channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
        send_head: T.Tensor([num_recv_tokens, num_ranks], "int32"),
        # symm buffers
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),  # reuse, already zeroed
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),  # reuse, already zeroed
        channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden], dtype),
        channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
        channel_topk_weights_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id = tx % 32
            warp_id = tx // 32
            responsible_channel = bx // 2


            if bx % 2 == 0:  # sender
                send_rank_id = (responsible_channel + warp_id) % num_ranks
                send_warp_id_in_rank = warp_id // num_ranks

                # get tasks
                rank_offset = T.if_then_else(send_rank_id > 0, rank_prefix_matrix[send_rank_id-1, rank], 0)
                num_rank_tokens = rank_prefix_matrix[send_rank_id, rank] - rank_offset
                channel_offset = channel_prefix_matrix[send_rank_id, responsible_channel]
                num_channel_tokens=  T.if_then_else(
                    responsible_channel == num_channels - 1,
                    num_rank_tokens,
                    channel_prefix_matrix[send_rank_id, responsible_channel + 1]
                ) - channel_offset
                token_start_idx = rank_offset + channel_offset
                token_end_idx = token_start_idx + num_channel_tokens

                # Iterate over all tokens and send by trunk
                current_channel_tail_idx = T.alloc_var('int32')
                current_channel_tail_idx = 0
                token_idx = T.alloc_var('int32')
                token_idx = token_start_idx
                with T.While(token_idx < token_end_idx):
                    # Check destination queue emptiness, or wait a buffer to be released (rare cases)
                    num_round_tokens = T.min(num_max_send_tokens, token_end_idx - token_idx)
                    if T.elect_one_sync():
                        T.wait_ge(channel_head_idx[responsible_channel, rank], current_channel_tail_idx + num_round_tokens - num_recv_buffer_tokens, peer=send_rank_id)
                    T.sync_warp()

                    # Send by trunk
                    for i in T.serial(send_warp_id_in_rank, num_round_tokens, warps_per_rank):
                        # Get an empty slot
                        dst_slot_idx = T.alloc_var('int32')
                        dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens

                        # 1. copy data
                        T.put_warp(T.address_of(x[token_idx + i, 0]), 
                            T.address_of(channel_x_buffers[responsible_channel, rank, dst_slot_idx, 0]), 
                            hidden, dst_pe=send_rank_id, unroll_factor=4, enable_aggresive_vectorize=True)
                            
                        # 2. send src idx
                        idx = T.alloc_var('int32')
                        if T.elect_one_sync():
                            T.ld(src_idx[token_idx + i], idx, nc=True)
                            T.st(channel_src_idx_buffers[responsible_channel, rank, dst_slot_idx], idx,
                                dst_pe=send_rank_id)

                        # 3. send topk_weights
                        if num_topk > 0 and lane_id < num_topk:
                            weight = T.alloc_var('float32')
                            T.ld(topk_weights[token_idx + i, lane_id], weight, nc=True)
                            T.st(channel_topk_weights_buffers[responsible_channel, rank, dst_slot_idx, lane_id], weight,
                                dst_pe=send_rank_id)

                    token_idx += num_round_tokens
                    current_channel_tail_idx += num_round_tokens

                    # move tail index
                    T.sync_threads(send_rank_id, threads_per_rank)
                    if send_warp_id_in_rank == 0 and T.elect_one_sync():
                        T.st(channel_tail_idx[responsible_channel, rank], current_channel_tail_idx,
                            scope='sys', sem='release',
                            dst_pe=send_rank_id)
            
            else:  # receiver
                #? Why we must need scope='shared', not 'shared.dynamic' here?
                warp_channel_head_idx = T.alloc_shared([warps, num_ranks], 'int32', scope='shared')
                shared_channel_tail_idx = T.alloc_shared([32], 'int32', scope='shared')  #! workaround for illegal address
                warp_retired = T.alloc_shared([warps], 'bool', scope='shared')
                if tx < warps:
                    warp_retired[tx] = False
                if lane_id < num_ranks:
                    warp_channel_head_idx[warp_id, lane_id] = 0
                if tx < 32:
                    shared_channel_tail_idx[tx] = 0
                T.sync_threads()

                if tx < 32:  # one warp for moving the queue head
                    last_head = T.alloc_var('int32')
                    last_head = 0
                    with T.While(lane_id < num_ranks):
                        # check retired
                        retired = T.alloc_var('bool')
                        retired = True
                        for i in T.serial(1, warps):
                            retired = retired and warp_retired[i]
                        if retired:
                            T.loop_break()
                        
                        # Update queue tail
                        new_tail = T.alloc_var('int32')
                        T.ld(channel_tail_idx[responsible_channel, lane_id], new_tail, sem="acquire", scope="sys")
                        # Use release semantics to ensure receiver warps see the update
                        T.st(shared_channel_tail_idx[lane_id], new_tail, sem="release", scope="cta")  # todo: weaker sem pair

                        # Update minimum head
                        min_head = T.alloc_var('int32')
                        min_head = 2**31 - 1  # int32 max
                        for i in T.serial(1, warps):
                            if not warp_retired[i]:
                                min_head = T.min(min_head, warp_channel_head_idx[i, lane_id])
                        if min_head != 2**31 - 1 and min_head > last_head:
                            last_head = min_head
                            T.st(channel_head_idx[responsible_channel, lane_id], min_head, sem="relaxed", scope="sys")
                else:  # other warps for reduction
                    # All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`

                    # The same tokens as the dispatch process
                    num_tokens_per_channel = T.truncdiv(num_recv_tokens+num_channels-1, num_channels)
                    # todo: this is a workaround, as TVM has a bug when calculating safe ceildiv for tir.Var
                    token_start_idx = T.min(num_tokens_per_channel * responsible_channel, num_recv_tokens)
                    token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_recv_tokens)

                    # Iterate over all tokens and combine
                    for token_idx in T.serial(token_start_idx+warp_id-1, token_end_idx, warps-1):
                        # Read expected head
                        expected_head = T.alloc_var('int32')
                        expected_head = -1
                        if lane_id < num_ranks:
                            T.ld(send_head[token_idx, lane_id], expected_head, nc=True)

                        condvar = T.alloc_var('int32')
                        T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                        with T.While(T.warp_any(condvar <= expected_head and expected_head >= 0)):
                            T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                            T.loop_continue()
                        # can we simplify this ?
                        T.sync_warp()

                        # Broadcast current heads
                        num_topk_ranks = T.alloc_var('int32')
                        num_topk_ranks = 0
                        topk_ranks= T.alloc_local([num_ranks], 'int32')
                        slot_indices = T.alloc_local([num_ranks], 'int32')
                        for i in T.serial(num_ranks):
                            expected_head_i = T.tvm_warp_shuffle(-1, expected_head, i, 32, 32)
                            if expected_head_i >= 0:
                                slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens
                                topk_ranks[num_topk_ranks] = i
                                num_topk_ranks += 1

                        # Reduce data with pipeline
                        # todo: vectorize
                        recv_value = T.alloc_local([num_ranks, 8], dtype)
                        values = T.alloc_local([8], "float32")
        
                        for i in T.serial(lane_id, hidden // 8, 32):
                            T.clear(values)
                            for j in T.serial(num_topk_ranks):
                                for k in T.vectorized(8):
                                    T.ld(channel_x_buffers[responsible_channel, topk_ranks[j], slot_indices[j], i*8+k], recv_value[j, k], nc=True)
                                
                            # todo: support bias

                            # Reduce a2a results
                            for j in T.serial(num_topk_ranks):
                                for k in T.vectorized(8):
                                    values[k] += recv_value[j, k]
                            for j in T.vectorized(8):
                                recv_x[token_idx, i*8+j] = values[j]

                        # Reduce topk_weights
                        if lane_id < num_topk:
                            weight_sum = T.alloc_var('float32')
                            weight_sum = 0
                            for i in T.serial(num_topk_ranks):
                                weight = T.alloc_var('float32')
                                T.ld(channel_topk_weights_buffers[responsible_channel, topk_ranks[i], slot_indices[i], lane_id], weight, nc=True)
                                weight_sum += weight
                            recv_topk_weights[token_idx, lane_id] = weight_sum

                        # Update head
                        if lane_id < num_ranks:
                            warp_channel_head_idx[warp_id, lane_id] = T.if_then_else(
                                expected_head < 0,
                                -expected_head - 1,
                                expected_head + 1)

                    # Retired
                    T.sync_warp()
                    if T.elect_one_sync():
                        warp_retired[warp_id] = True

    return combine_main


def intranode_combine(
    rank: int, 
    allocator, 
    symm_buffers,
    x,
    config,
    handle,
    topk_weights, 
):
    assert handle is not None
    rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, _, send_head = handle  
    barrier_signal, _, _, _, _, channel_head_idx, channel_tail_idx, channel_x_buffers, channel_src_idx_buffers, _, channel_topk_weights_buffers = symm_buffers

    # acquire_shapes
    num_tokens, hidden = x.shape
    _, num_topk = topk_weights.shape
    num_ranks, num_channels = channel_prefix_matrix.shape
    num_recv_tokens = send_head.shape[0]
    
    # notify combine
    cached_notify_combine(num_ranks, config.num_sms, send_head, channel_head_idx, channel_tail_idx, barrier_signal, allocator)

    # combine
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device='cuda')
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device='cuda')

    kernel = combine_kernel(
        rank, 
        num_ranks,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        hidden,
        num_topk,
        config.num_sms,
        dtype='bfloat16'
    )
    kernel.initialize(allocator=allocator)
    kernel(
        x,
        topk_weights,
        recv_src_idx,
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        channel_head_idx,
        channel_tail_idx,
        channel_x_buffers,
        channel_src_idx_buffers,
        channel_topk_weights_buffers,
    )
    return recv_x, recv_topk_weights
