# For intranode only
# This op is distributed
### TILELANG_USE_DISTRIBUTED=1 python dispatch.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from typing import Optional, Tuple, List
from utils import Config, create_moe_recv_counters  # noqa: F403

from get_dispatch_layout import get_dispatch_layout
from notify_dispatch import notify_dispatch


@tilelang.jit
def dispatch_kernel(
    rank, num_ranks, 
    num_tokens, 
    num_recv_tokens,
    hidden, 
    num_topk, 
    num_experts,
    num_sms,
    dtype: str = 'bfloat16',
):
    threads = 768  # 24 warps1
    TMABytesPerWarp = 8192
    smem_size = TMABytesPerWarp * threads // 32
    
    num_threads_per_rank = threads // num_ranks  # 96 (3 warps for each rank)
    num_channels = num_sms // 2  # 10 (2 SMs for each channel)
    num_channels_total = num_channels * num_ranks  # 80
    num_local_experts = num_experts // num_ranks

    num_send_warps = num_threads_per_rank // 32  # 24
    num_send_warps_per_rank = num_send_warps // num_ranks  # 3


    @T.prim_func
    def dispatch_main(
        # output
        recv_x: T.Tensor((num_recv_tokens, hidden), 'bfloat16'),
        recv_src_idx: T.Tensor((num_recv_tokens,), 'int32'),
        recv_topk_idx: T.Tensor((num_recv_tokens, num_topk), 'int64'),
        recv_topk_weights: T.Tensor((num_recv_tokens, num_topk), 'float'),
        recv_channel_offset: T.Tensor([num_ranks, num_channels], "int32"),
        send_head: T.Tensor([num_tokens, num_ranks], "int32"),
        # input
        x: T.Tensor([num_tokens, hidden], "int32"),
        topk_idx: T.Tensor([num_tokens, num_topk], "int64"),
        topk_weights: T.Tensor([num_tokens, num_topk], "float32"),
        is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),
        channel_prefix_matrix: T.Tensor([num_ranks, num_channels], "int32"),
        # For now we use NVSHMEM to allocate buffer
        # instead of using CUDA IPC on the host side
        # buffer_ptrs: T.Tensor([...], "int32"),
        # channel metadatas (for local rank)
        channel_start_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_end_offset: T.Tensor([num_channels, num_ranks], "int32"),
        channel_head_idx: T.Tensor([num_channels, num_ranks], "int32"),
        channel_tail_idx: T.Tensor([num_channels, num_ranks], "int32"),
        # channel buffers (for remote ranks)
        channel_x_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, hidden_int4], "int4"),
        channel_src_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens], "int32"),
        channel_topk_idx_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "uint64"),
        channel_topk_weights_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_topk], "float32"),
        channel_x_scales_buffers: T.Tensor([num_channels, num_ranks, num_recv_buffer_tokens, num_scales], "float32"),
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()
            lane_id = tx // 32
            responsible_rank = tx // num_threads_per_rank
            responsible_channel = bx // 2
            tgt_rank = rank if bx % 2 == 0 else (rank + 1) % num_ranks
            channel_rank_offset = responsible_channel * num_ranks + tgt_rank

            if bx % 2 == 0:  # sender
                send_warp_id_in_rank = (tx % num_send_warps_per_rank) // 32

                # send offset by `-value-1` e.g. 0->-1, 1->-2
                if send_warp_id_in_rank == 0 and T.elect_one_sync():
                    T.st

            

# todo: support cached-mode via handle
def intranode_dispatch(
    # data
    x: torch.Tensor,  # todo: support fp8 quant
    # handle
    handle: Optional[Tuple] = None,
    # meta
    rank: int,
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
    allocator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], Tuple]:
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
        topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
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
        handle: the returned communication handle.
    """

    assert handle is None  # Currently only support non-cached mode
    assert num_tokens_per_rank is not None or is_token_in_rank is not None and num_tokens_per_expert is not None, \
        "num_tokens_per_rank, is_token_in_rank, and num_tokens_per_expert must be provided in non-cached mode"

    # acquire shapes
    num_tokens, hidden = x.shape
    num_experts = num_tokens_per_expert.shape[0]
    num_ranks = num_tokens_per_rank.shape[0]
    num_local_experts = num_experts // num_ranks
    num_topk = topk_idx.shape[1]

    # Default config
    config = Config.get_dispatch_config(num_ranks) if config is None else config

    # Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    # Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
    rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
    channel_prefix_matrix = torch.empty([num_ranks, config.num_channels], dtype=torch.int32, device='cuda')

    moe_recv_counter_mapped, moe_recv_expert_counter_mapped = create_moe_recv_counters(num_ranks)[3:5]

    per_rank_buffer = tilelang.tensor((num_ranks, num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    per_expert_buffer = tilelang.tensor((num_ranks, num_local_experts), dtype=torch.int32, device='cuda', allocator=allocator).zero_()
    barrier_signal = tilelang.tensor((num_ranks), dtype=torch.int32, device='cuda', allocator=allocator).zero_()

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

