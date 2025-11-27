# For intranode only
# This op is distributed
### TILELANG_USE_DISTRIBUTED=1 python combine.py

from asyncio import Handle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.distributed.utils import init_dist
from utils import Config, create_moe_recv_counters, gen_inputs  # noqa: F403
from argparse import ArgumentParser

from get_dispatch_layout import get_dispatch_layout
from notify_dispatch import notify_dispatch
from dispatch import intranode_dispatch

# tilelang.disable_cache()
os.environ['NCCL_DEBUG'] = 'WARN'  # silence NCCL log


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True},
        debug_root_path='/root/workspace/wt/debug/notify_combine')
def cached_notify_combine_kernel(
    num_recv_tokens,
    num_ranks,
    num_sms,
):
    num_channels = num_sms // 2
    threads = max(128, 32 * num_ranks)

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

            if bx == 0:
                # block 0 is responsible for clearing channel_head/tail_idx buffers
                # note that the buffer layout is slightly different from original DeepEP logic
                T.sync_blocks(barrier_signal)
                T.clear(channel_head_idx)
                T.clear(channel_tail_idx)
                T.barrier_blocks(barrier_signal)
            else:
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
    num_recv_tokens,  #! means the original #tokens on each rank here
    ##### symm buffers #####
    send_head: torch.Tensor,
    channel_head_idx: torch.Tensor,
    channel_tail_idx: torch.Tensor,
    barrier_signal: torch.Tensor,    
    allocator
):
    kernel = cached_notify_combine_kernel(num_recv_tokens, num_ranks, num_sms)
    kernel.initialize(allocator=allocator)

    kernel(
        send_head,
        channel_head_idx,
        channel_tail_idx,
        barrier_signal,
    )


@tilelang.jit(
    pass_configs={"tl.disable_tma_lower": True,  # use TMA later
        "tl.disable_warp_specialized": True}, debug_root_path='/root/workspace/wt/debug/combine')
def combine_kernel(
    rank, num_ranks,
    num_recv_tokens,
    num_max_send_tokens,  # config.num_max_nvl_chunked_send_tokens
    num_recv_buffer_tokens,  # config.num_max_nvl_chunked_recv_tokens
    hidden, 
    num_topk, 
    num_sms,
    dtype: str = 'bfloat16',
):
    num_tokens = T.dynamic('num_tokens')

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
                    channel_prefix_matrix[send_rank_id, responsible_channel + 1] - channel_offset,
                )
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
                            hidden, dst_pe=send_rank_id, unroll_factor=4)
                            
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
                warp_channel_head_idx = T.alloc_shared([warps, num_ranks], 'int32')
                shared_channel_tail_idx = T.alloc_shared([32], 'int32')  #! workaround for illegal address
                warp_retired = T.alloc_shared([warps], 'bool')
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
                        T.st(shared_channel_tail_idx[lane_id], new_tail, sem="release", scope="cta")

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
                    # for *_buffers[i] channel_rank_offset = responsible_channel * kNumRanks + i;

                    # The same tokens as the dispatch process
                    num_tokens_per_channel = T.ceildiv(num_recv_tokens, num_channels)
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
                        if bx == 1 and tx == 32:
                            T.print(condvar)
                        T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                        with T.While(T.warp_any(condvar <= expected_head and expected_head >= 0)):
                            T.ld(shared_channel_tail_idx[lane_id], condvar, sem="acquire", scope="cta")
                            T.print(condvar-expected_head)
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
                        if bx == 0 and tx == 32:
                            T.print(num_topk_ranks, 'broadcast finished')

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
                            
                        if bx == 1 and tx == 32:
                            T.print(warp_channel_head_idx[warp_id, lane_id])

                    # Retired
                    T.sync_warp()
                    if T.elect_one_sync():
                        warp_retired[warp_id] = True
                    if bx == 1 and tx == 32:
                        T.print(warp_channel_head_idx, 'retired')

    return combine_main


def intranode_combine(rank: int, allocator, x, topk_weights, src_idx, 
    rank_prefix_matrix, channel_prefix_matrix, send_head, 
    channel_head_idx, channel_tail_idx, barrier_signal, channel_x_buffers, channel_src_idx_buffers, channel_topk_weights_buffers,
    config=None):

    # acquire_shapes
    num_tokens, hidden = x.shape
    _, num_topk = topk_weights.shape
    num_ranks, num_channels = channel_prefix_matrix.shape
    num_recv_tokens = send_head.shape[0]
    
    # Default config
    config = Config.get_combine_config(num_ranks) if config is None else config
    
    ### notify combine ###
    kernel1 = cached_notify_combine_kernel(num_recv_tokens, num_ranks, config.num_sms)
    kernel1.initialize(allocator=allocator)
    kernel1(
        send_head,
        channel_head_idx,
        channel_tail_idx,
        barrier_signal,
    )

    ### combine ###
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device='cuda')
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=torch.float32, device='cuda')

    kernel2 = combine_kernel(
        rank, num_ranks,
        num_recv_tokens,
        config.num_max_nvl_chunked_send_tokens,
        config.num_max_nvl_chunked_recv_tokens,
        hidden,
        num_topk,
        config.num_sms,
        dtype='bfloat16'
    )
    kernel2.initialize(allocator=allocator)
    kernel2(
        x,
        topk_weights,
        src_idx,
        recv_x,
        recv_topk_weights,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        channel_head_idx,
        channel_tail_idx,
        channel_x_buffers,
        channel_src_idx_buffers,
        channel_topk_weights_buffers,
    )

    return recv_x, recv_topk_weights


def test_intranode_combine(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
    num_ranks: int,
    expert_alignment: int,
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
        print('get dispatch layout...')
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx.to(torch.int64), num_experts)  # DeepEP requires int64 topk_idx

    if rank == 0: 
        print('intranode dispatch...')
    
    ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_num_recv_tokens_per_expert_list, ref_handle, event = \
        buffer.dispatch(x, None, num_tokens_per_rank, None, is_token_in_rank, num_tokens_per_expert, topk_idx.to(torch.int64), topk_weights, expert_alignment)  # DeepEP requires int64 topk_idx``

    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, symm_buffers = \
        intranode_dispatch(rank, allocator, x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, None)

    assert torch.equal(recv_x, ref_recv_x), f'recv_x mismatch, max err: {(recv_x - ref_recv_x).abs().max()}'
    assert torch.equal(recv_topk_idx, ref_recv_topk_idx), f'recv_topk_idx mismatch, max err: {(recv_topk_idx - ref_recv_topk_idx).abs().max()}'
    assert torch.equal(recv_topk_weights, ref_recv_topk_weights), f'recv_topk_weights mismatch, max err: {(recv_topk_weights - ref_recv_topk_weights).abs().max()}'
    assert num_recv_tokens_per_expert_list == ref_num_recv_tokens_per_expert_list, 'num_recv_tokens_per_expert_list mismatch'
    
    if rank == 0:
        print('Start combine...')

    ref_combine_x, ref_combine_topk_weights, _, ref_send_head = buffer.combine(ref_recv_x, ref_handle, ref_recv_topk_weights, previous_event=event)

    rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
    channel_head_idx, channel_tail_idx, barrier_signal, channel_x_buffers, channel_src_idx_buffers, channel_topk_weights_buffers = symm_buffers
    combine_x, combine_topk_weights = intranode_combine(rank, allocator, recv_x, recv_topk_weights, recv_src_idx, rank_prefix_matrix, recv_channel_prefix_matrix, send_head, channel_head_idx, channel_tail_idx, barrier_signal, channel_x_buffers, channel_src_idx_buffers, channel_topk_weights_buffers)

    assert torch.equal(combine_x, ref_combine_x), f'combine_x mismatch, max err: {(combine_x - ref_combine_x).abs().max()}'
    assert torch.equal(combine_topk_weights, ref_combine_topk_weights), f'combine_topk_weights mismatch, max err: {(combine_topk_weights - ref_combine_topk_weights).abs().max()}'
    print(f'[rank {rank}] All checks passed for TileScale intranode_combine. ✅')


def main(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_intranode_combine(
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        rank,
        num_ranks,
        args.expert_alignment,
        group,
    )

def parse_args():
    parser = ArgumentParser(description="Test notify_dispatch")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden size")
    parser.add_argument("--num_topk", type=int, default=8, help="Number of top-k experts to select for each token")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--expert_alignment", type=int, default=1, help="Expert alignment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_ranks = args.num_ranks
    torch.multiprocessing.spawn(main, args=(num_ranks, args), nprocs=num_ranks)