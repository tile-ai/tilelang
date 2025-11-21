# For intranode only
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
from typing import Optional, Tuple, List
from utils import Config


# todo: support cached-mode via handle
def intranode_dispatch(
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
    # todo: support expert alignment and num_worst_tokens
    # tuning cfg
    config: Optional[Config] = None,
    # todo: support async functionality
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

    num_memset_int = config.num_channels * num_ranks * 4

    # Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    # Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
    rank_prefix_matrix = torch.empty([num_ranks, num_ranks], dtype=torch.int32, device='cuda')
    channel_prefix_matrix = torch.empty([num_ranks, config.num_channels], dtype=torch.int32, device='cuda')

    notify_dispatch(
        num_tokens_per_rank,
        
    )

