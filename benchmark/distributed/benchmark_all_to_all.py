from __future__ import annotations

import torch
import tilelang
import tilelang.language as T
from tilelang.distributed import init_distributed, dtype_map
import argparse
import random
from triton_dist.kernels.nvidia import fast_all_to_all, all_to_all_post_process
from benchmark.distributed.utils import create_all_to_all_context, AllToAllContext

tilelang.disable_cache()


def all_to_all(max_m, hidden, num_tot_experts, WORLD_SIZE, threads=128, dtype="float16"):
    scale_dtype = "float"
    EXPERTS_PER_RANK = num_tot_experts // WORLD_SIZE

    @T.prim_func
    def main(
        send_buf: T.Tensor((max_m, hidden), dtype),  # type: ignore
        recv_buf: T.Tensor((WORLD_SIZE * max_m * 2, hidden), dtype),  # type: ignore
        scale_send_buf: T.Tensor((max_m), scale_dtype),  # type: ignore
        scale_recv_buf: T.Tensor((WORLD_SIZE * max_m * 2), scale_dtype),  # type: ignore
        split_send_buf: T.Tensor((num_tot_experts), "int32"),  # type: ignore
        split_recv_buf: T.Tensor((num_tot_experts * 2), "int32"),  # type: ignore
        signal_buf: T.Tensor((WORLD_SIZE * 2), "uint64"),  # type: ignore
    ):
        with T.Kernel(WORLD_SIZE, threads=threads) as (bx):
            peer = bx
            tx = T.thread_binding(threads, thread="threadIdx.x")

            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            m_start = T.alloc_local([1], "int32")
            m_end = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()
            m_start[0] = split_send_buf[peer * EXPERTS_PER_RANK]
            m_end[0] = split_send_buf[(peer + 1) * EXPERTS_PER_RANK]

            # T.putmem_nbi_block(
            #     T.address_of(recv_buf[0, 0]), T.address_of(send_buf[m_start[0], 0]),
            #     (m_end[0] - m_start[0]) * hidden * 2, peer)

            T.fence()

            if tx == 0:
                T.signal_op(
                    T.address_of(signal_buf[mype[0]]),
                    99,
                    T.Amo.SIGNAL_SET,
                    peer,
                )
                T.signal_wait_until(
                    T.address_of(signal_buf[peer]),
                    T.CmpType.EQ,
                    99,
                )

    return main


class TilelangAllToAll:
    def __init__(self, ctx: AllToAllContext):
        self.ctx = ctx
        self.func = all_to_all(ctx.max_m, ctx.hidden, ctx.num_tot_experts, ctx.WORLD_SIZE, threads=128)
        self.kernel = tilelang.compile(self.func, pass_configs={"tl.disable_tma_lower": True})
        if self.ctx.rank == 0:
            print(self.kernel.get_kernel_source())

    def __call__(self, send_tensor: torch.Tensor, send_split_cumsum: torch.Tensor, send_scale: torch.Tensor | None):
        """
        low-latency all-to-all communication
        """
        with_scale = send_scale is not None

        act_pos = self.ctx.call_count % 2

        split_buf_st = act_pos * self.ctx.num_tot_experts
        split_buf_ed = split_buf_st + self.ctx.num_tot_experts

        data_buf_st = act_pos * self.ctx.WORLD_SIZE * self.ctx.max_m
        data_buf_ed = data_buf_st + self.ctx.WORLD_SIZE * self.ctx.max_m

        scale_buf_st = act_pos * self.ctx.WORLD_SIZE * self.ctx.max_m
        scale_buf_ed = scale_buf_st + self.ctx.WORLD_SIZE * self.ctx.max_m

        num_tokens = send_tensor.shape[0]
        assert num_tokens <= self.ctx.max_m
        self.ctx.send_buf[:num_tokens, :] = send_tensor
        if with_scale:
            self.ctx.scale_send_buf[:num_tokens] = send_scale

        self.kernel(
            self.ctx.send_buf,
            self.ctx.recv_buf,
            self.ctx.scale_send_buf,
            self.ctx.scale_recv_buf,
            self.ctx.split_send_buf,
            self.ctx.split_recv_buf,
            self.ctx.signal_buf,
        )

        self.ctx.call_count = (self.ctx.call_count + 1) % self.ctx.MOD_VALUE
        out_lis: list[torch.Tensor] = []
        out_lis.append(self.ctx.split_recv_buf[split_buf_st:split_buf_ed])
        out_lis.append(self.ctx.recv_buf[data_buf_st:data_buf_ed, :])
        if with_scale:
            out_lis.append(self.ctx.scale_recv_buf[scale_buf_st:scale_buf_ed])
        else:
            out_lis.append(None)
        return out_lis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=3584)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument("--dtype", default="float16", help="data type")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with_scale", action="store_true")
    parser.add_argument("--print_source", action="store_true")
    parser.add_argument("--threads", type=int, default=128)
    return parser.parse_args()


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for _ in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


import torch.distributed
import triton
import triton.language as tl


def calc_gather_index(
    scatter_index: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):
    @triton.jit
    def _kernel(
        scatter_index: torch.Tensor,
        gather_index: torch.Tensor,
        topk_index: torch.Tensor,
        ntokens: int,
        topk: int,
        row_start: int,
        row_end: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < ntokens * topk
        scatter_idx = tl.load(scatter_index + offset, mask=mask, other=-1)
        token_idx = offset // topk
        topk_idx = offset % topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start, token_idx, mask=token_idx_mask)
        tl.store(topk_index + scatter_idx - row_start, topk_idx, mask=token_idx_mask)

    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    topk_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]),)  # noqa: E731
    _kernel[grid](
        scatter_index,
        gather_index,
        topk_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return gather_index, topk_index


def calc_scatter_index_stable(choosed_experts: torch.Tensor):
    return choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape)


def main():
    WORLD_SIZE, RANK, LOCAL_RANK = init_distributed()

    args = parse_args()
    token_num = args.M // 2
    experts_per_rank = args.G // WORLD_SIZE

    all_to_all_ctx = create_all_to_all_context(
        # set max_m to 2 * M * topk to avoid bug in combine for now
        # TODO: Check this
        args.M * args.topk * 2,
        args.N,
        RANK,
        args.G,
        WORLD_SIZE,
        experts_per_rank,
        dtype_map[args.dtype],
        torch.float,
    )

    def perf_triton(input: torch.Tensor, scale_tensor: torch.Tensor, exp_indices: torch.Tensor):
        # prepare the indexes
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
        split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

        # calculate the scatter idx
        scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
        # calculate the gather idx accordingly
        gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * args.topk)
        # use torch native scatter forward(will not be included in the e2e time measurement)
        scattered_input = torch.empty(input.size(0) * args.topk, input.size(1), dtype=input.dtype, device=input.device)
        scattered_scale_tensor = torch.empty(
            (scale_tensor.size(0) * args.topk),
            dtype=scale_tensor.dtype,
            device=scale_tensor.device,
        )
        scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
        scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))

        def fwd():
            return fast_all_to_all(all_to_all_ctx, scattered_input, split_cumsum, scattered_scale_tensor if args.with_scale else None)

        torch.cuda._sleep(1000000000)
        # warmup
        for _ in range(20):
            fwd()

        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        # bench
        st.record()
        for _ in range(args.bench_iters):
            _ = fwd()
        ed.record()
        torch.cuda.synchronize()
        avg_time = st.elapsed_time(ed) / args.bench_iters

        # 1. dispatch
        dispatch_splits, dispatch_token, dispatch_scale = fast_all_to_all(
            all_to_all_ctx, scattered_input, split_cumsum, scattered_scale_tensor if args.with_scale else None
        )
        dispatch_token, dispatch_scale = all_to_all_post_process(
            all_to_all_ctx, dispatch_splits, dispatch_token, dispatch_scale if args.with_scale else None
        )

        # 2. compute: moe_compute(dispatch_token, dispatch_scale, moe_weight, ...)
        # ...

        # 3. combine
        combine_splits, combine_token, combine_scale = fast_all_to_all(
            all_to_all_ctx, dispatch_token, splits_to_cumsum(dispatch_splits), dispatch_scale
        )
        combine_token, combine_scale = all_to_all_post_process(
            all_to_all_ctx, combine_splits, combine_token, combine_scale if args.with_scale else None
        )

        # 3.1. reduce: [num_tokens_local_rank * topk] => [num_tokens_local_rank]
        combine_reduced_out = torch.zeros_like(input)
        combine_reduced_out.index_add_(0, gather_idx_cur_rank, combine_token)

        # check the output of `dispatch => => combine`
        torch.testing.assert_close(combine_reduced_out, input * args.topk, rtol=1e-2, atol=1e-2)

        tilelang_all_to_all = TilelangAllToAll(all_to_all_ctx)
        tilelang_all_to_all(scattered_input, split_cumsum, scattered_scale_tensor if args.with_scale else None)

        # torch.testing.assert_close(tilelang_out[1], dispatch_token, rtol=1e-2, atol=1e-2)
        # torch.testing.assert_close(tilelang_scale, dispatch_scale, rtol=1e-2, atol=1e-2)

        return dispatch_token, dispatch_scale, avg_time

    # random simulate token received from dataloader
    print(f"Rank-{RANK}: Received {token_num} tokens")

    exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
    assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
    exp_indices = exp_indices.to("cuda")
    input = torch.rand(token_num, args.N, dtype=torch.float32).to(dtype_map[args.dtype]).to("cuda")
    scale_tensor = torch.rand(token_num, dtype=torch.float32).to("cuda")

    torch.cuda.synchronize()
    triton_out, triton_scale, triton_time = perf_triton(input, scale_tensor, exp_indices)
    torch.cuda.synchronize()
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
