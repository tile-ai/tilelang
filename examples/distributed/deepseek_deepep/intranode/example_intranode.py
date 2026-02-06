### TILELANG_USE_DISTRIBUTED=1 python test_intranode.py (--cached, optionally)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add parent folder to path

import torch
from argparse import ArgumentParser
from tilelang.distributed.utils import init_dist

from buffer import EPBuffer
from deepep_utils import gen_inputs, ep_bench

# tilelang.disable_cache()
os.environ["NCCL_DEBUG"] = "WARN"  # silence NCCL log


def test_intranode(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
    num_ranks: int,
    expert_alignment: int,
    cached_dispatch: bool,
    group: torch.distributed.ProcessGroup,
):
    try:
        import deep_ep  # noqa: F403
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install DeepEP to run this test.") from None

    # Create interface buffers
    ts_buffer = EPBuffer(group, 2**30, num_topk, num_experts, hidden)
    deepep_buffer = deep_ep.Buffer(group, num_nvl_bytes=2**30)

    # Generate inputs for testing
    x, topk_idx, topk_weights, rank_idx = gen_inputs(num_tokens, hidden, num_topk, num_experts, num_ranks)

    # 1. test get_dispatch_layout
    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = deepep_buffer.get_dispatch_layout(
        topk_idx, num_experts
    )
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = ts_buffer.get_dispatch_layout(topk_idx)

    assert torch.equal(num_tokens_per_expert, ref_num_tokens_per_expert), (
        f"[rank {rank}] num_tokens_per_expert mismatch, max err: {(num_tokens_per_expert - ref_num_tokens_per_expert).abs().max()}"
    )
    assert torch.equal(is_token_in_rank, ref_is_token_in_rank), f"[rank {rank}] is_token_in_rank mismatch"
    assert torch.equal(num_tokens_per_rank, ref_num_tokens_per_rank), (
        f"[rank {rank}] num_tokens_per_rank mismatch, max err: {(num_tokens_per_rank - ref_num_tokens_per_rank).abs().max()}"
    )

    group.barrier()
    if rank == 0:
        print("Check passed for get_dispatch_layout. ✅")

    # 2. test dispatch
    # ref
    ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, ref_num_recv_tokens_per_expert_list, ref_handle, event = deepep_buffer.dispatch(
        x, None, ref_num_tokens_per_rank, None, ref_is_token_in_rank, ref_num_tokens_per_expert, topk_idx, topk_weights, expert_alignment
    )
    # ours
    if cached_dispatch:
        recv_x = ts_buffer.dispatch(
            x, ref_handle, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, None, None, expert_alignment
        )
    else:
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle = ts_buffer.dispatch(
            x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment
        )

    # check dispatch output
    assert torch.equal(recv_x, ref_recv_x), f"[rank {rank}] recv_x mismatch, max err: {(recv_x - ref_recv_x).abs().max()}"
    if not cached_dispatch:
        assert torch.equal(recv_topk_idx, ref_recv_topk_idx), (
            f"[rank {rank}] recv_topk_idx mismatch, max err: {(recv_topk_idx - ref_recv_topk_idx).abs().max()}"
        )
        assert torch.equal(recv_topk_weights, ref_recv_topk_weights), (
            f"[rank {rank}] recv_topk_weights mismatch, max err: {(recv_topk_weights - ref_recv_topk_weights).abs().max()}"
        )
        assert num_recv_tokens_per_expert_list == ref_num_recv_tokens_per_expert_list, (
            f"[rank {rank}] num_recv_tokens_per_expert_list mismatch"
        )

        # check handle
        rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
        (
            ref_rank_prefix_matrix,
            ref_channel_prefix_matrix,
            ref_recv_channel_prefix_matrix,
            ref_recv_src_idx,
            ref_is_token_in_rank,
            ref_send_head,
        ) = ref_handle
        assert torch.equal(rank_prefix_matrix, ref_rank_prefix_matrix), (
            f"[rank {rank}] rank_prefix_matrix mismatch, max err: {(rank_prefix_matrix - ref_rank_prefix_matrix).abs().max()}"
        )
        assert torch.equal(channel_prefix_matrix, ref_channel_prefix_matrix), (
            f"[rank {rank}] channel_prefix_matrix mismatch, max err: {(channel_prefix_matrix - ref_channel_prefix_matrix).abs().max()}"
        )
        assert torch.equal(recv_channel_prefix_matrix, ref_recv_channel_prefix_matrix), (
            f"[rank {rank}] recv_channel_prefix_matrix mismatch, max err: {(recv_channel_prefix_matrix - ref_recv_channel_prefix_matrix).abs().max()}"
        )
        assert torch.equal(recv_src_idx, ref_recv_src_idx), (
            f"[rank {rank}] recv_src_idx mismatch, max err: {(recv_src_idx - ref_recv_src_idx).abs().max()}"
        )
        assert torch.equal(is_token_in_rank, ref_is_token_in_rank), (
            f"[rank {rank}] is_token_in_rank mismatch, max err: {(is_token_in_rank - ref_is_token_in_rank).abs().max()}"
        )
        assert torch.equal(send_head, ref_send_head), (
            f"[rank {rank}] send_head mismatch, max err: {(send_head - ref_send_head).abs().max()}"
        )

    group.barrier()
    if rank == 0:
        print(f"Check passed for {'cached' if cached_dispatch else 'non-cached'} dispatch. ✅")

    # 3. test combine
    ref_combined_x, ref_combined_topk_weights, _ = deepep_buffer.combine(recv_x, ref_handle, ref_recv_topk_weights)
    if cached_dispatch:  # acquire handle first
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle = ts_buffer.dispatch(
            x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment
        )
    combined_x, combined_topk_weights = ts_buffer.combine(recv_x, handle, recv_topk_weights)
    assert torch.equal(combined_x, ref_combined_x), (
        f"[rank {rank}] combined_x mismatch, max err: {(combined_x - ref_combined_x).abs().max()}"
    )
    assert torch.equal(combined_topk_weights, ref_combined_topk_weights), (
        f"[rank {rank}] combined_topk_weights mismatch, max err: {(combined_topk_weights - ref_combined_topk_weights).abs().max()}"
    )

    group.barrier()
    if rank == 0:
        print("Check passed for combine. ✅")

    if rank == 0:
        print("All checks passed for TileScale intranode DeepEP. ✅")

    # benchmark
    if rank == 0:
        print(f"========== Benchmarking {'cached' if cached_dispatch else 'non-cached'} dispatch ==========")
    if not cached_dispatch:
        group.barrier()
        deepep_dispatch_time = ep_bench(
            lambda: deepep_buffer.dispatch(
                x,
                None,
                ref_num_tokens_per_rank,
                None,
                ref_is_token_in_rank,
                ref_num_tokens_per_expert,
                topk_idx,
                topk_weights,
                expert_alignment,
            ),
            warmup=50,
            rep=50,
        )
        print(f"[rank {rank}] DeepEP dispatch time: {deepep_dispatch_time:.4f}ms")
        group.barrier()
        ts_dispatch_time = ep_bench(
            lambda: ts_buffer.dispatch(
                x, None, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, topk_idx, topk_weights, expert_alignment
            ),
            warmup=50,
            rep=50,
        )
        print(f"[rank {rank}] TileScale dispatch time: {ts_dispatch_time:.4f}ms")
        group.barrier()
    else:
        group.barrier()
        deepep_dispatch_time = ep_bench(
            lambda: deepep_buffer.dispatch(
                x, ref_handle, ref_num_tokens_per_rank, None, ref_is_token_in_rank, ref_num_tokens_per_expert, None, None, expert_alignment
            ),
            warmup=50,
            rep=50,
        )
        print(f"[rank {rank}] DeepEP dispatch time: {deepep_dispatch_time:.4f}ms")
        group.barrier()
        ts_dispatch_time = ep_bench(
            lambda: ts_buffer.dispatch(
                x, ref_handle, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, None, None, expert_alignment
            ),
            warmup=50,
            rep=50,
        )
        print(f"[rank {rank}] TileScale dispatch time: {ts_dispatch_time:.4f}ms")
        group.barrier()

    if rank == 0:
        print("========== Benchmarking combine ==========")
    group.barrier()
    deepep_combine_time = ep_bench(lambda: deepep_buffer.combine(recv_x, ref_handle, ref_recv_topk_weights), warmup=50, rep=50)
    print(f"[rank {rank}] DeepEP combine time: {deepep_combine_time:.4f}ms")

    group.barrier()
    ts_combine_time = ep_bench(lambda: ts_buffer.combine(recv_x, handle, recv_topk_weights), warmup=50, rep=50)
    print(f"[rank {rank}] TileScale combine time: {ts_combine_time:.4f}ms")
    group.barrier()

    if rank == 0:
        print("========== Benchmarking report ==========")
    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
    if rank == 0:
        print(
            f"DeepEP dispatch time: {deepep_dispatch_time:.4f}ms, bandwidth: {dispatch_bf16_nvl_recv_bytes / deepep_dispatch_time / 1e6:.2f} GB/s (NVL)"
        )
        print(
            f"TileScale dispatch time: {ts_dispatch_time:.4f}ms, bandwidth: {dispatch_bf16_nvl_recv_bytes / ts_dispatch_time / 1e6:.2f} GB/s (NVL)"
        )
        print(
            f"DeepEP combine time: {deepep_combine_time:.4f}ms, bandwidth: {combine_bf16_nvl_send_bytes / deepep_combine_time / 1e6:.2f} GB/s (NVL)"
        )
        print(
            f"TileScale combine time: {ts_combine_time:.4f}ms, bandwidth: {combine_bf16_nvl_send_bytes / ts_combine_time / 1e6:.2f} GB/s (NVL)"
        )


def run(local_rank: int, num_local_ranks: int, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    test_intranode(
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

    torch.distributed.destroy_process_group()


def parse_args():
    parser = ArgumentParser(description="Test dispatch")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden size")
    parser.add_argument("--num_topk", type=int, default=8, help="Number of top-k experts to select for each token")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--expert_alignment", type=int, default=1, help="Expert alignment")
    parser.add_argument("--cached", action="store_true", default=False, help="Whether to use cached dispatch")
    return parser.parse_args()


def main():
    args = parse_args()

    num_ranks = args.num_ranks
    torch.multiprocessing.spawn(run, args=(num_ranks, args), nprocs=num_ranks)


if __name__ == "__main__":
    main()
