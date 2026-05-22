"""Intranode allreduce with NVSwitch multimem instructions.

Strategies:
  one_shot: every rank computes the full output with multimem.ld_reduce.
  two_shot: each rank computes one shard with multimem.ld_reduce, then
            broadcasts that shard with multimem.st. This matches the TK
            high-throughput allreduce pattern and reduces per-rank work by
            world_size.
  two_shot_tma_copy: experimental Hopper path that uses ordinary TMA store
            to the multicast VA instead of multimem.st.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing

import tilelang
import tilelang.language as T
from tilelang.distributed.bench import do_bench
from tilelang.distributed.host import init_dist
from tilelang.distributed.allocator import get_allocator

os.environ.setdefault("NCCL_DEBUG", "ERROR")


_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_TL_DTYPES = {
    "bf16": T.bfloat16,
    "fp16": T.float16,
    "fp32": T.float32,
}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def multimem_allreduce_one_shot_kernel(N, block_N, threads, dtype=T.bfloat16):
    @T.prim_func
    def main(
        mcast_buf: T.Tensor((N,), dtype),
        result: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            acc = T.alloc_fragment((block_N,), dtype)
            T.multimem_ld_reduce(
                mcast_buf[bx * block_N : (bx + 1) * block_N],
                acc,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(acc, result[bx * block_N : (bx + 1) * block_N])

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def multimem_allreduce_two_shot_kernel(N, num_ranks, block_N, threads, dtype=T.bfloat16):
    N_per_rank = N // num_ranks

    @T.prim_func
    def main(
        mcast_buf: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N_per_rank, block_N), threads=threads) as bx:
            local_rank = T.get_rank()
            offset = local_rank * N_per_rank + bx * block_N
            acc = T.alloc_fragment((block_N,), dtype)
            T.multimem_ld_reduce(
                mcast_buf[offset : offset + block_N],
                acc,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.multimem_st(acc, mcast_buf[offset : offset + block_N])

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
    compile_once=True,
)
def multimem_allreduce_two_shot_tma_copy_kernel(N, num_ranks, block_N, threads, dtype=T.bfloat16):
    N_per_rank = N // num_ranks

    @T.prim_func
    def main(
        mcast_buf: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N_per_rank, block_N), threads=threads) as bx:
            local_rank = T.get_rank()
            offset = local_rank * N_per_rank + bx * block_N
            acc = T.alloc_fragment((block_N,), dtype)
            shard = T.alloc_shared((block_N,), dtype)
            T.multimem_ld_reduce(
                mcast_buf[offset : offset + block_N],
                acc,
                reduce_op=T.MultimemReduceOp.ADD,
            )
            T.copy(acc, shard)
            # TMA store to the multicast VA.
            T.tma_copy(shard, mcast_buf[offset : offset + block_N])
            # The CTA does not read this global data after issuing the store.
            T.tma_store_wait(0, False)

    return main


def torch_allreduce(group: torch.distributed.ProcessGroup, local_data: torch.Tensor) -> torch.Tensor:
    expected = local_data.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=group)
    return expected


def _bandwidth_gbs(nbytes: int, duration_ms: float, num_ranks: int, strategy: str) -> tuple[float, float]:
    algo_bw = nbytes * 1e-9 / (duration_ms * 1e-3) * 2
    if strategy == "one_shot":
        hw_bw = algo_bw * num_ranks / 2
    else:
        hw_bw = algo_bw * (num_ranks - 1) / num_ranks
    return algo_bw, hw_bw


def _check_result(
    group: torch.distributed.ProcessGroup,
    local_rank: int,
    strategy: str,
    local_data: torch.Tensor,
    result: torch.Tensor,
    args: argparse.Namespace,
    *,
    start: int = 0,
    end: int | None = None,
):
    expected = torch_allreduce(group, local_data)
    if end is not None:
        result = result[start:end]
        expected = expected[start:end]
    max_diff = (result.float() - expected.float()).abs().max().item()
    passed = torch.allclose(result, expected, atol=args.atol, rtol=args.rtol)
    print(f"rank {local_rank} {strategy} check {'passed' if passed else 'failed'}. max_diff={max_diff}")
    dist.barrier(group)


def main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    assert args.N % num_local_ranks == 0, "N must be divisible by num-processes"
    assert args.N % args.block_n == 0, "N must be divisible by block-n"
    assert (args.N // num_local_ranks) % args.block_n == 0, "N_per_rank must be divisible by block-n"
    assert args.threads == 256, "this example is tuned and validated for 256 threads"
    assert args.block_n % 512 == 0, "block-n must be a multiple of 512 for the current multimem vectorization"

    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert rank == local_rank and num_ranks == num_local_ranks, "only support single-node launch for now"

    torch_dtype = _TORCH_DTYPES[args.dtype]
    tl_dtype = _TL_DTYPES[args.dtype]
    dtype_bytes = torch.empty((), dtype=torch_dtype).element_size()
    allocator = get_allocator(
        size=max(args.N * dtype_bytes * 4, 2**22),
        device=f"cuda:{local_rank}",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_local_ranks,
        group=group,
        mcast_size=args.N * dtype_bytes,
    )

    one_shot_kernel = None
    two_shot_kernel = None
    two_shot_tma_copy_kernel = None
    if args.strategy in ("one_shot", "both", "all"):
        one_shot_kernel = multimem_allreduce_one_shot_kernel(args.N, args.block_n, args.threads, tl_dtype)
        one_shot_kernel.compile_group = group
        one_shot_kernel.initialize(allocator=allocator)
    if args.strategy in ("two_shot", "both", "all"):
        two_shot_kernel = multimem_allreduce_two_shot_kernel(args.N, num_local_ranks, args.block_n, args.threads, tl_dtype)
        two_shot_kernel.compile_group = group
        two_shot_kernel.initialize(allocator=allocator)
    if args.strategy in ("two_shot_tma_copy", "all"):
        two_shot_tma_copy_kernel = multimem_allreduce_two_shot_tma_copy_kernel(
            args.N, num_local_ranks, args.block_n, args.threads, tl_dtype
        )
        two_shot_tma_copy_kernel.compile_group = group
        two_shot_tma_copy_kernel.initialize(allocator=allocator)

    if local_rank == 0 and args.print_source:
        if one_shot_kernel is not None:
            print(one_shot_kernel.get_kernel_source())
        if two_shot_kernel is not None:
            print(two_shot_kernel.get_kernel_source())
        if two_shot_tma_copy_kernel is not None:
            print(two_shot_tma_copy_kernel.get_kernel_source())

    torch.manual_seed(42 + local_rank)
    local_data = torch.randn(args.N, dtype=torch_dtype, device=f"cuda:{local_rank}")
    local_data = local_data / (num_local_ranks**0.5)
    mcast_buf, local_buf = allocator._allocate_mcast_tensor((args.N,), torch_dtype)
    local_buf.copy_(local_data)
    one_shot_out = tilelang.tensor((args.N,), torch_dtype, allocator=allocator)

    torch.cuda.synchronize()
    dist.barrier(group)

    def run_one_shot():
        one_shot_kernel(mcast_buf, one_shot_out)
        return one_shot_out

    def run_two_shot():
        two_shot_kernel(mcast_buf)
        return local_buf

    def run_two_shot_tma_copy():
        two_shot_tma_copy_kernel(mcast_buf)
        return local_buf

    def bench_two_shot_variant(strategy_name, run_fn):
        # Two-shot variants write only this rank's shard back through the
        # multicast VA, so the local physical buffer must be restored between
        # iterations.
        local_buf.copy_(local_data)
        torch.cuda.synchronize()
        dist.barrier(group)
        run_fn()
        torch.cuda.synchronize()
        dist.barrier(group)
        if args.check:
            N_per_rank = args.N // num_local_ranks
            shard_start = local_rank * N_per_rank
            _check_result(
                group,
                local_rank,
                strategy_name,
                local_data,
                local_buf,
                args,
                start=shard_start,
                end=shard_start + N_per_rank,
            )

        def reset():
            local_buf.copy_(local_data)
            torch.cuda.synchronize()
            dist.barrier(group)

        reset()
        tl_t = do_bench(
            run_fn,
            warmup=args.warmup,
            rep=args.rep,
            post_fn=reset,
            group=group,
        )
        if local_rank == 0:
            algo_bw, hw_bw = _bandwidth_gbs(args.N * dtype_bytes, tl_t, num_local_ranks, "two_shot")
            print(
                f"tilelang {strategy_name} allreduce time: {tl_t * 1000:.2f} us, "
                f"algo BW: {algo_bw:.2f} GB/s, HW BW: {hw_bw:.2f} GB/s"
            )

    if one_shot_kernel is not None:
        run_one_shot()
        torch.cuda.synchronize()
        dist.barrier(group)
        if args.check:
            _check_result(group, local_rank, "one_shot", local_data, one_shot_out, args)
        tl_t = do_bench(run_one_shot, warmup=args.warmup, rep=args.rep, group=group)
        if local_rank == 0:
            algo_bw, hw_bw = _bandwidth_gbs(args.N * dtype_bytes, tl_t, num_local_ranks, "one_shot")
            print(
                f"tilelang one_shot allreduce time: {tl_t * 1000:.2f} us, "
                f"algo BW: {algo_bw:.2f} GB/s, HW BW: {hw_bw:.2f} GB/s"
            )

    if two_shot_kernel is not None:
        bench_two_shot_variant("two_shot", run_two_shot)

    if two_shot_tma_copy_kernel is not None:
        bench_two_shot_variant("two_shot_tma_copy", run_two_shot_tma_copy)

    allocator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--N", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--block-n", type=int, default=512)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--dtype", choices=tuple(_TORCH_DTYPES), default="bf16")
    parser.add_argument(
        "--strategy",
        choices=("one_shot", "two_shot", "two_shot_tma_copy", "both", "all"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args.num_processes, args), nprocs=args.num_processes, join=True)
