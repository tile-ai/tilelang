# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import torch
import torch.distributed as dist
import pynvshmem  
import tilelang
import tilelang.language as T
from tilelang.distributed.utils import init_distributed, dtype_map, perf_fn, CUDA_CHECK
from typing import List
from cuda import cuda

tilelang.disable_cache()


# Copied from Triton-distributed/tutorials/02-intra-node-allgather.py
# This is the default AllGather impl. in Triton-dist given full-mesh NVLink
def cp_engine_producer_all_gather_full_mesh_pull(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    ag_stream: torch.cuda.Stream,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, _ = local_tensor.shape

    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    with torch.cuda.stream(ag_stream):
        for src_rank in rank_orders:
            if src_rank == rank:
                continue
            # peer: src_rank, offset src_rank[src_rank] -> rank[src_rank]
            dst = remote_tensor_buffers[rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank, :]
            src = remote_tensor_buffers[src_rank][src_rank * M_per_rank:(src_rank + 1) *
                                                  M_per_rank, :]
            dst.copy_(src)
            (err,) = cuda.cuStreamWriteValue32(
                ag_stream.cuda_stream,
                barrier_buffers[rank][src_rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


def allgather(PE_num, M, N, dtype="float16", threads=128):
    M_per_rank = M // PE_num
    block_M = 4
        
    @T.prim_func
    def a2a_pull(
            A: T.Tensor((M_per_rank, N), dtype),  # type: ignore
            B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(M_per_rank // block_M, PE_num - 1, threads=threads) as (bx, by):
            mype = T.get_pe()
            npes = T.get_pe_num()
            peer = (mype + by + 1) % npes
            
            T.getmem_nbi_block(
                T.address_of(B[peer*M_per_rank+bx*block_M, 0]), T.address_of(A[bx*block_M, 0]),
                block_M * N * dtype_map[dtype].itemsize, peer)
            # We don't need a barrier for the pull mode

    return a2a_pull


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M", type=int,
        default=8192)  # Follow Triton-setting, we benchmark on (M, N) = (8192, 12288)
    parser.add_argument("--N", type=int, default=12288)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=128, help="number of threads in a block")
    parser.add_argument("--print_source", action="store_true", help="print kernel source code")
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == '__main__':
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    assert WORLD_SIZE <= 8, "This benchmark is designed for intra-node communication"

    args = parse_args()
    M, N, dtype, threads, warmup, repeat = args.M, args.N, args.dtype, args.threads, args.warmup, args.repeat
    PE_num = WORLD_SIZE
    assert M % PE_num == 0, "M must be divisible by PE_num"
    M_per_rank = M // PE_num
    torch_dtype = dtype_map[dtype]
    nelems = M * PE_num

    func = allgather(PE_num, M, N, dtype=dtype, threads=threads)
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True})

    # Get CUDA Source
    if RANK == 0 and args.print_source:
        print(kernel.get_kernel_source())

    local_data = torch.randn([M_per_rank, N], dtype=torch_dtype).cuda()

    # Benchmark Torch
    def torch_ag():
        out = torch.empty((M, N), dtype=torch_dtype).cuda()
        dist.all_gather_into_tensor(out, local_data, group=TP_GROUP)
        return out

    dist.barrier(TP_GROUP)
    torch_out, torch_t = perf_fn(torch_ag, warmup, repeat)
    print(f"rank {RANK} torch all_gather avg time: {torch_t} ms")

    # Benchmark Triton-dist
    def triton_ag():
        ag_buffer_ptrs = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [M, N], torch_dtype)  # buffer for dist-triton allgather
        signal = pynvshmem.nvshmem_create_tensor_list_intra_node(
            ([PE_num]), torch.uint64)  # each rank corresponds to one barrier
        ag_buffer_ptrs[RANK][
            RANK * M_per_rank:(RANK + 1) * M_per_rank,
        ].copy_(local_data)
        signal[RANK].zero_()
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        cp_engine_producer_all_gather_full_mesh_pull(
            RANK, PE_num, local_data, ag_buffer_ptrs, torch.cuda.current_stream(), signal
        )  # Here we use current stream for allgather, we can pass any other stream for comm-comp fusion.
        return ag_buffer_ptrs[RANK]

    dist.barrier(TP_GROUP)
    tt_out, tt_t = perf_fn(triton_ag, warmup, repeat)
    print(f"rank {RANK} triton all_gather avg time: {tt_t} ms")

    # Benchmark Tilelang-dist
    def tilelang_ag():
        ag_buffer = pynvshmem.nvshmem_create_tensor([M_per_rank, N], torch_dtype)
        ag_buffer.copy_(local_data)
        out = pynvshmem.nvshmem_create_tensor([M, N], torch_dtype)
        out[RANK * M_per_rank:(RANK + 1) * M_per_rank, :].copy_(local_data)
        kernel(ag_buffer, out)
        
        return out

    dist.barrier(TP_GROUP)
    tl_out, tl_t = perf_fn(tilelang_ag, warmup, repeat)
    print(f"rank {RANK} tilelang all_gather avg time: {tl_t} ms")
    # Tested on 4A100 with full-mesh NVLink, comparable with Triton-dist and ~20x faster than Torch

    # Check correctness
    assert torch.allclose(tt_out, torch_out, atol=0, rtol=0), f'max error: {(tt_out - torch_out).abs().max()}'
    print(f"rank {RANK} check passed.✅")
    
    dist.destroy_process_group()
