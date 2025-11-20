# For intranode only
# This op is non-distributed
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from typing import Tuple
from argparse import ArgumentParser
from utils import gen_inputs  # noqa: F403


# TODO(wt): Add async functionality
def get_dispatch_layout(
        topk_idx: torch.Tensor, num_experts: int,
        num_ranks: int) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
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
    num_tokens_per_rank = torch.empty(num_ranks, dtype=torch.int32, device='cuda')
    num_tokens_per_rdma_rank = None  # No RDMA ranks in intranode settings
    num_tokens_per_expert = torch.empty(num_experts, dtype=torch.int32, device='cuda')
    is_token_in_rank = torch.empty((num_tokens, num_ranks), dtype=torch.bool, device='cuda')

    # Launch the kernel
    kernel = get_dispatch_layout_kernel(num_tokens, num_topk, num_experts, num_ranks)
    kernel(
        topk_idx,
        num_tokens_per_rank,
        # num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    )

    # TODO(wt): Wait streams when adding async functionality

    return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank


@tilelang.jit
def get_dispatch_layout_kernel(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
) -> tilelang.JITKernel:
    """Kernel to compute the dispatch layout."""

    # Work partition from DeepEP/csrc/kernels/layout.cu:get_dispatch_layout
    threads = 256
    experts_per_sm = 4
    ranks_per_sm = 8
    num_sms = T.ceildiv(num_experts, experts_per_sm) + T.ceildiv(num_ranks, ranks_per_sm)
    experts_per_rank = num_experts // num_ranks

    @T.prim_func
    def main(
            topk_idx: T.Tensor([num_tokens, num_topk], "int64"),  # type: ignore
            num_tokens_per_rank: T.Tensor([num_ranks], "int32"),  # type: ignore
            num_tokens_per_expert: T.Tensor([num_experts], "int32"),  # type: ignore
            is_token_in_rank: T.Tensor([num_tokens, num_ranks], "bool"),  # type: ignore
    ):
        with T.Kernel(num_sms, threads=threads) as bid:
            tid = T.get_thread_binding()

            # Calculate expert statistics
            tokens_per_expert_per_thread = T.alloc_shared([threads, experts_per_sm], "int32")
            T.clear(tokens_per_expert_per_thread)
            expert_begin_idx = T.alloc_local([1], "int32")
            expert_begin_idx[0] = bid * experts_per_sm
            expert_end_idx = T.alloc_local([1], "int32")
            expert_end_idx[0] = T.min(expert_begin_idx[0] + experts_per_sm, num_experts)

            if expert_begin_idx[0] < expert_end_idx[0]:
                for i in T.serial(tid, num_tokens, threads):
                    for j in T.serial(0, num_topk):
                        expert_idx = T.alloc_local([1], "int32")
                        expert_idx[0] = T.cast(topk_idx[i, j], "int32")
                        if expert_begin_idx[0] <= expert_idx[0] and expert_idx[0] < expert_end_idx[
                                0]:
                            tokens_per_expert_per_thread[tid,
                                                         expert_idx[0] - expert_begin_idx[0]] += 1

                if expert_begin_idx[0] + tid < expert_end_idx[0]:
                    sum = T.alloc_local([1], "int32")
                    sum[0] = 0
                    for i in T.serial(0, threads):
                        sum[0] += tokens_per_expert_per_thread[i, tid]
                    num_tokens_per_expert[expert_begin_idx[0] + tid] = sum[0]

            # Calculate rank statistics
            sm_begin = T.alloc_local([1], "int32")
            sm_begin[0] = T.ceildiv(num_experts, experts_per_sm)
            rank_begin_idx = T.alloc_local([1], "int32")
            rank_begin_idx[0] = (bid - sm_begin[0]) * ranks_per_sm
            rank_end_idx = T.alloc_local([1], "int32")
            rank_end_idx[0] = T.min(rank_begin_idx[0] + ranks_per_sm, num_ranks)

            if rank_begin_idx[0] >= 0 and rank_begin_idx[0] < rank_end_idx[0]:
                tokens_per_rank_per_thread = T.alloc_shared([threads, ranks_per_sm], "int32")
                T.clear(tokens_per_rank_per_thread)

                expert_begin = T.alloc_local([1], "int32")
                expert_begin[0] = rank_begin_idx[0] * experts_per_rank
                expert_end = T.alloc_local([1], "int32")
                expert_end[0] = rank_end_idx[0] * experts_per_rank

                for i in T.serial(tid, num_tokens, threads):
                    is_in_rank = T.alloc_local([ranks_per_sm], "int32")
                    T.clear(is_in_rank)

                    for j in T.serial(0, num_topk):
                        expert_idx = T.alloc_local([1], "int32")
                        rank_idx = T.alloc_local([1], "int32")
                        expert_idx[0] = T.cast(topk_idx[i, j], "int32")
                        if expert_begin[0] <= expert_idx[0] and expert_idx[0] < expert_end[0]:
                            rank_idx[0] = expert_idx[0] // experts_per_rank - rank_begin_idx[0]

                            is_in_rank[rank_idx[0]] += 1

                    for j in T.serial(rank_begin_idx[0], rank_end_idx[0]):
                        if is_in_rank[j - rank_begin_idx[0]] > 0:
                            is_token_in_rank[i, j] = True
                            tokens_per_rank_per_thread[tid, j - rank_begin_idx[0]] += 1
                        else:
                            is_token_in_rank[i, j] = False

                if rank_begin_idx[0] + tid < rank_end_idx[0]:
                    sum = T.alloc_local([1], "int32")
                    sum[0] = 0
                    for i in T.serial(0, threads):
                        sum[0] += tokens_per_rank_per_thread[i, tid]
                    num_tokens_per_rank[rank_begin_idx[0] + tid] = sum[0]

    return main


def test_get_dispatch_layout(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
):
    try:
        import deep_ep_cpp  # noqa: F403
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install DeepEP to run this test.")

    # Validate correctness
    topk_idx = gen_inputs(num_tokens, 1, num_topk, num_experts, num_ranks)[1]
    buffer = deep_ep_cpp.Buffer(
        0, # rank
        num_ranks, 
        0, # num_nvl_bytes
        0, # num_rdma_bytes
        False, # low_latency_mode
        False, # explicit_destroy
        False, # enable_shrink
        False, # use fabric
    )

    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts, None, False, False)

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank = get_dispatch_layout(topk_idx, num_experts, num_ranks)

    assert torch.allclose(num_tokens_per_expert, ref_num_tokens_per_expert), \
        f"num_tokens_per_expert mismatch, max err: {(num_tokens_per_expert - ref_num_tokens_per_expert).abs().max()}"

    assert torch.equal(is_token_in_rank, ref_is_token_in_rank), \
        "is_token_in_rank mismatch"

    assert torch.equal(num_tokens_per_rank, ref_num_tokens_per_rank), \
        f"num_tokens_per_rank mismatch, max err: {(num_tokens_per_rank - ref_num_tokens_per_rank).abs().max()}"

    print("All checks passed.✅")

    # Benchmark
    t1 = do_bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts, None, False, False))
    t2 = do_bench(lambda: get_dispatch_layout(topk_idx, num_experts, num_ranks))
    print(f"DeepEP: {t1:.3f} ms")
    print(f"TileScale: {t2:.3f} ms")
    print(f"Speedup: {t1 / t2:.2f}x")


def parse_args():
    parser = ArgumentParser(description="Test get_dispatch_layout")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument(
        "--num_topk", type=int, default=8, help="Number of top-k experts to select for each token")
    parser.add_argument("--num_experts", type=int, default=256, help="Number of experts")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_get_dispatch_layout(
        num_tokens=args.num_tokens,
        num_topk=args.num_topk,
        num_experts=args.num_experts,
        num_ranks=args.num_ranks)
