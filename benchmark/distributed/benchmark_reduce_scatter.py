# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# This benchmark requires GPU arch sm_90 or above.

import argparse
import torch
import torch.distributed as dist
import triton_dist
from triton_dist.kernels.nvidia.reduce_scatter import (
    reduce_scatter_ring_push_1d_intra_node_ce,
    reduce_scatter_ring_push_1d_intra_node_sm,
    reduce_scatter_ring_push_1d_intra_node_sm_rma
)
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.distributed.utils import init_distributed, dtype_map, perf_fn, dist_print

tilelang.disable_cache()

#TODO: Bench on 4/8 H100
#TODO: split N?

'''
Note: Minor numerical differences exist between Triton/TileLang and Torch (~1e-2) 
due to the way reductions are handled in different implementations.
(No error when #PE = 2)
'''

def reducescatter(PE_num, M, N, dtype="float16", threads=128):
    M_per_rank = M // PE_num
    block_M = 1
    acc_dtype = "float32"
    
    @T.prim_func
    def pull_reduce(
            A: T.Tensor((M, N), dtype),  # type: ignore
            B: T.Tensor((M_per_rank, N), dtype),  # type: ignore
    ):
        with T.Kernel(M_per_rank // block_M, threads=threads) as (bx):
            mype = T.get_pe()
            
            A_shared = T.alloc_shared((PE_num, block_M, N), dtype)
            A_local = T.alloc_fragment((PE_num, block_M, N), dtype)
            A_local_sum = T.alloc_fragment((block_M, N), acc_dtype)
            
            for i in T.serial(PE_num - 1):
                peer = (mype + i + 1) % PE_num
                T.getmem_nbi_block(
                    T.address_of(A_shared[peer, 0, 0]),
                    T.address_of(A[mype * M_per_rank + bx * block_M, 0]),
                    block_M * N * dtype_map[dtype].itemsize,
                    peer
                )
            base = mype * M_per_rank + bx * block_M
            T.copy(A[base:base+block_M, :], A_shared[mype, :, :])

            T.fence() # Ensure reduce happens after all IO
            
            T.copy(A_shared, A_local)
            T.reduce_sum(A_local, A_local_sum, dim=0)
            T.copy(A_local_sum, B[bx*block_M:bx*block_M+block_M, :])

    return pull_reduce


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192) 
    parser.add_argument("--N", type=int, default=16384)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=128, help="number of threads in a block")
    parser.add_argument("--print_source", action="store_true", help="print kernel source code")
    parser.add_argument("--warmup", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=10, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == '__main__':
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
    torch.cuda.set_device(RANK)
    assert WORLD_SIZE <= 8, "This benchmark is designed for intra-node RS"

    args = parse_args()
    M, N, dtype, threads, warmup, repeat = args.M, args.N, args.dtype, args.threads, args.warmup, args.repeat
    PE_num = WORLD_SIZE
    assert M % PE_num == 0, "M must be divisible by PE_num"
    M_per_rank = M // PE_num
    torch_dtype = dtype_map[dtype]
    nelems = M * PE_num

    func = reducescatter(PE_num, M, N, dtype=dtype, threads=threads)
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True}, target='cuda')

    # Get CUDA Source
    if RANK == 0 and args.print_source:
        print(kernel.get_kernel_source())

    local_data = torch.randn([M, N], dtype=torch_dtype).cuda()
    
    ## Input: [M, N] per rank
    ## Output: [M_per_rank, N] per rank

    # Benchmark Torch
    def torch_rs():
        out = torch.empty((M_per_rank, N), dtype=torch_dtype).cuda()
        dist.reduce_scatter_tensor(out, local_data, group=TP_GROUP)
        return out
    
    dist.barrier(TP_GROUP)
    torch_out, torch_t = perf_fn(torch_rs, warmup, repeat)
    print(f"rank {RANK} torch reduce_scatter avg time: {torch_t} ms")

    # Benchmark Triton-dist
    def triton_rs():
        # Currently only support 'ce' implementation 
        input_buffer = pynvshmem.nvshmem_create_tensor([M, N], torch_dtype)
        input_buffer.copy_(local_data)
        input_flag = torch.ones((PE_num, ), device="cuda", dtype=torch.int32)
        symm_reduce_buffers = pynvshmem.nvshmem_create_tensor_list_intra_node([M, N], torch_dtype)
        symm_reduce_flags = pynvshmem.nvshmem_create_tensor_list_intra_node((PE_num, ), torch.int32)
        symm_reduce_flags[RANK].zero_()
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        output = reduce_scatter_ring_push_1d_intra_node_ce(
            RANK,
            WORLD_SIZE,
            input_buffer,
            input_flag,
            symm_reduce_buffers,
            symm_reduce_flags,
        )
        pynvshmem.nvshmemx_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)
        return output
    
    dist.barrier(TP_GROUP)
    tt_out, tt_t = perf_fn(triton_rs, warmup, repeat)
    print(f"rank {RANK} triton reduce_scatter avg time: {tt_t} ms")

    # Benchmark Tilelang-dist
    def tilelang_rs():
        rs_buffer = pynvshmem.nvshmem_create_tensor([M, N], torch_dtype)
        rs_buffer.copy_(local_data)
        out = pynvshmem.nvshmem_create_tensor([M_per_rank, N], torch_dtype)
        kernel(rs_buffer, out) # Since we use pull, we don't need sync after kernel
        return out

    dist.barrier(TP_GROUP)
    tl_out, tl_t = perf_fn(tilelang_rs, warmup, repeat)
    print(f"rank {RANK} tilelang reduce_scatter avg time: {tl_t} ms")

    # Check correctness
    assert torch.allclose(tl_out, torch_out, atol=1e-2, rtol=1e-2), f'max error: {(tt_out - torch_out).abs().max()}'
    print(f'max err: {(tt_out - torch_out).abs().max()}')
    print(f"rank {RANK} check passed.✅")

    dist.destroy_process_group()

