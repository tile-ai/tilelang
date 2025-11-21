# For intranode only
# This op is distributed
### TILELANG_USE_DISTRIBUTED=1 python notify_dispatch.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import tilelang
import tilelang.language as T
import torch
from argparse import ArgumentParser
from tilelang.distributed.utils import init_dist
from utils import gen_inputs  # noqa: F403

from get_dispatch_layout import get_dispatch_layout


# TileScale notify-dispatch kernel for non-cached mode
# Check: DeepEP/csrc/kernels/intranode.cu::notify_dispatch
@tilelang.jit
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
        moe_recv_counter_mapped: T.Tensor((1,), 'int64'),
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
                T.barrier_all_blocks_sys(barrier_signal)

                # `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
                # `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
                if tx < num_ranks:
                    T.st(per_rank_buffer[rank, tx], num_tokens_per_rank[tx], dst_pe=tx)
                    for i in T.serial(num_local_experts):
                        T.st(per_expert_buffer[rank, i], num_tokens_per_expert[tx * num_local_experts + i], dst_pe=tx)
                
                T.barrier_all_blocks_sys(barrier_signal)

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
                T.copy(per_rank_buffer, rank_prefix_matrix)

                #? We don't cleanup the buffer for later use, as it is one time used?
                T.barrier_all_blocks_sys(barrier_signal)
            else:
                dst_rank = bx - 1
                for channel_id in T.serial(warp_id, num_channels, num_warps):
                    num_tokens_per_channel = T.ceildiv(num_tokens, num_channels)
                    token_start_idx = T.min(num_tokens_per_channel * channel_id, num_tokens)
                    token_end_idx = T.min(token_start_idx + num_tokens_per_channel, num_tokens)
                    cnt = T.alloc_local([1], 'int32')
                    cnt[0] = 0
                    for i in T.serial(token_start_idx + lane_id, token_end_idx, 32):
                        cnt[0] += is_token_in_rank[i, dst_rank]
                    cnt[0] = T.warp_reduce_sum(cnt[0])
                    if lane_id == 0:  # todo: replace with elect_one_sync() for sm90
                        channel_prefix_matrix[dst_rank, channel_id] = cnt[0]
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


# todo: impl cached_notify_dispatch


def test_notify_dispatch(
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

    num_local_experts = num_experts // num_ranks

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
        print(f'get dispatch layout...')
    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank = get_dispatch_layout(topk_idx, num_experts, num_ranks)
    assert torch.equal(num_tokens_per_expert, ref_num_tokens_per_expert), \
        f"num_tokens_per_expert mismatch, max err: {(num_tokens_per_expert - ref_num_tokens_per_expert).abs().max()}"
    assert torch.equal(is_token_in_rank, ref_is_token_in_rank), \
        "is_token_in_rank mismatch"
    assert torch.equal(num_tokens_per_rank, ref_num_tokens_per_rank), \
        f"num_tokens_per_rank mismatch, max err: {(num_tokens_per_rank - ref_num_tokens_per_rank).abs().max()}"

    if rank == 0: 
        print(f'notify dispatch...')
    handle = buffer.dispatch(x, None, ref_num_tokens_per_rank, None, ref_is_token_in_rank, ref_num_tokens_per_expert, topk_idx, topk_weights)[-2]
    ref_rank_prefix_matrix, ref_channel_prefix_matrix = handle[:2]

    # create buffers in need
    moe_recv_counter_mapped = torch.empty([1], dtype=torch.int64, device='cuda')
    moe_recv_counter_mapped[0] = -1
    moe_recv_expert_counter_mapped = torch.empty([num_local_experts], dtype=torch.int32, device='cuda')
    moe_recv_expert_counter_mapped.fill_(-1)

    per_rank_buffer = tilelang.tensor((num_ranks, num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    per_expert_buffer = tilelang.tensor((num_ranks, num_local_experts), dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    barrier_signal = tilelang.tensor((num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()

    rank_prefix_matrix, channel_prefix_matrix = notify_dispatch(
        rank,
        num_ranks,
        num_experts,
        num_tokens,
        10,  # 20 sms by default
        expert_alignment,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        moe_recv_counter_mapped,
        moe_recv_expert_counter_mapped,
        per_rank_buffer,
        per_expert_buffer,
        barrier_signal,
        allocator
    )

    assert torch.allclose(rank_prefix_matrix, ref_rank_prefix_matrix), \
        f"rank_prefix_matrix mismatch, max err: {(rank_prefix_matrix - ref_rank_prefix_matrix).abs().max()}"
    assert torch.allclose(channel_prefix_matrix, ref_channel_prefix_matrix), \
        f"channel_prefix_matrix mismatch, max err: {(channel_prefix_matrix - ref_channel_prefix_matrix).abs().max()}"
    print(f'[rank {rank}] All checks passed for TileScale notify_dispatch. ✅')

    # todo: benchmark


def main(
    local_rank: int, num_local_ranks: int, args
):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_notify_dispatch(
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
