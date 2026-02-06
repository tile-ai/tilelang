# For intranode only
# This op is non-distributed
### python get_dispatch_layout.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from typing import Tuple


# TODO(wt): Add async functionality
def get_dispatch_layout(
    topk_idx: torch.Tensor, num_experts: int, num_ranks: int
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Calculate the layout required for later communication.

    Arguments:
        topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
            `-1` means no selections.
        num_experts: the number of experts.
        num_ranks: the number of ranks.

    Returns:
        num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
        num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
            rank (with the same GPU index), return `None` for intranode settings.
        num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
        is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
    """

    # Check inputs
    assert topk_idx.dtype == torch.int64, "topk_idx must be of dtype torch.int64"
    assert topk_idx.ndim == 2, "topk_idx must be a 2D tensor"
    assert topk_idx.is_contiguous(), "topk_idx must be a contiguous tensor"
    assert num_experts > 0, "num_experts must be greater than 0"

    # Allocate tensors
    # TODO(wt): Wait on previous events and allocate on comm stream when adding async functionality
    num_tokens, num_topk = topk_idx.shape
    num_tokens_per_rank = torch.empty(num_ranks, dtype=torch.int32, device="cuda")
    num_tokens_per_expert = torch.empty(num_experts, dtype=torch.int32, device="cuda")
    is_token_in_rank = torch.empty((num_tokens, num_ranks), dtype=torch.bool, device="cuda")

    # Launch the kernel
    kernel = get_dispatch_layout_kernel(num_topk, num_experts, num_ranks)
    kernel(
        topk_idx,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    )

    # TODO(wt): Wait streams when adding async functionality

    return num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank


@tilelang.jit(pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def get_dispatch_layout_kernel(
    num_topk: int,
    num_experts: int,
    num_ranks: int,
) -> tilelang.JITKernel:
    threads = 256
    experts_per_sm = 4
    ranks_per_sm = 8
    num_sms = T.ceildiv(num_experts, experts_per_sm) + T.ceildiv(num_ranks, ranks_per_sm)
    experts_per_rank = num_experts // num_ranks

    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def get_dispatch_layout_main(
        topk_idx: T.Tensor([num_tokens, num_topk], "int64"),  # type: ignore
        num_tokens_per_rank: T.Tensor([num_ranks], "int32"),  # type: ignore
        num_tokens_per_expert: T.Tensor([num_experts], "int32"),  # type: ignore
        is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),  # type: ignore
    ):
        with T.Kernel(num_sms, threads=threads) as bx:
            tx = T.get_thread_binding()

            # Calculate expert statistics
            tokens_per_expert_per_thread = T.alloc_shared([threads, experts_per_sm], "int32")
            T.clear(tokens_per_expert_per_thread)
            expert_begin_idx = T.alloc_var("int32")
            expert_begin_idx = bx * experts_per_sm
            expert_end_idx = T.alloc_var("int32")
            expert_end_idx = T.min(expert_begin_idx + experts_per_sm, num_experts)

            if expert_begin_idx < expert_end_idx:
                for i in T.serial(tx, num_tokens, threads):
                    for j in T.serial(num_topk):
                        expert_idx = T.alloc_var("int32")
                        expert_idx = topk_idx[i, j]
                        if expert_begin_idx <= expert_idx and expert_idx < expert_end_idx:
                            tokens_per_expert_per_thread[tx, expert_idx - expert_begin_idx] += 1

                if expert_begin_idx + tx < expert_end_idx:
                    sum = T.alloc_var("int32")
                    sum = 0
                    for i in T.serial(threads):
                        sum += tokens_per_expert_per_thread[i, tx]
                    num_tokens_per_expert[expert_begin_idx + tx] = sum

            # Calculate rank statistics
            sm_begin = T.alloc_var("int32")
            sm_begin = T.ceildiv(num_experts, experts_per_sm)
            rank_begin_idx = T.alloc_var("int32")
            rank_begin_idx = (bx - sm_begin) * ranks_per_sm
            rank_end_idx = T.alloc_var("int32")
            rank_end_idx = T.min(rank_begin_idx + ranks_per_sm, num_ranks)

            if rank_begin_idx >= 0 and rank_begin_idx < rank_end_idx:
                tokens_per_rank_per_thread = T.alloc_shared([threads, ranks_per_sm], "int32")
                T.clear(tokens_per_rank_per_thread)

                expert_begin = T.alloc_var("int32")
                expert_begin = rank_begin_idx * experts_per_rank
                expert_end = T.alloc_var("int32")
                expert_end = rank_end_idx * experts_per_rank

                for i in T.serial(tx, num_tokens, threads):
                    is_in_rank = T.alloc_local([ranks_per_sm], "int32")
                    T.clear(is_in_rank)

                    for j in T.serial(num_topk):
                        expert_idx = T.alloc_var("int32")
                        rank_idx = T.alloc_var("int32")
                        expert_idx = topk_idx[i, j]
                        if expert_begin <= expert_idx and expert_idx < expert_end:
                            rank_idx = expert_idx // experts_per_rank - rank_begin_idx

                            is_in_rank[rank_idx] += 1

                    for j in T.serial(rank_begin_idx, rank_end_idx):
                        if is_in_rank[j - rank_begin_idx] > 0:
                            is_token_in_rank[i, j] = True
                            tokens_per_rank_per_thread[tx, j - rank_begin_idx] += 1
                        else:
                            is_token_in_rank[i, j] = False

                if rank_begin_idx + tx < rank_end_idx:
                    sum = T.alloc_var("int32")
                    sum = 0
                    for i in T.serial(threads):
                        sum += tokens_per_rank_per_thread[i, tx]
                    num_tokens_per_rank[rank_begin_idx + tx] = sum

    return get_dispatch_layout_main
