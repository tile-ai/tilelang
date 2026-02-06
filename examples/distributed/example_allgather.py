import argparse
import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.distributed import init_distributed, dtype_map, perf_fn


def allgather(PE_num, M, N, dtype="float16", threads=128):
    M_per_rank = M // PE_num
    block_M = 4

    @T.prim_func
    def a2a_split(
        A: T.Tensor((M_per_rank, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        # Each block is responsible for sending (block_M, N) to exact one rank.
        with T.Kernel(M_per_rank // block_M, PE_num - 1, threads=threads) as (bx, by):
            mype = T.get_pe()
            npes = T.get_pe_num()

            A_shared = T.alloc_shared((block_M, N), dtype)
            local_base = bx * block_M
            global_base = M_per_rank * mype + local_base
            T.copy(A[local_base : local_base + block_M, :], A_shared)
            peer = (mype + by + 1) % npes
            T.putmem_nbi_block(T.address_of(B[global_base, 0]), T.address_of(A_shared[0, 0]), block_M * N * dtype_map[dtype].itemsize, peer)

    return a2a_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=12288)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=128, help="number of threads in a block")
    parser.add_argument("--print_source", action="store_true", help="print kernel source code")
    parser.add_argument("--warmup", type=int, default=1, help="number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=5, help="number of repeat iterations")
    return parser.parse_args()


if __name__ == "__main__":
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

    # Torch
    def torch_ag():
        out = torch.empty((M, N), dtype=torch_dtype).cuda()
        dist.all_gather_into_tensor(out, local_data, group=TP_GROUP)
        return out

    dist.barrier(TP_GROUP)
    ref, t = perf_fn(torch_ag, warmup, repeat)
    print(f"rank {RANK} torch all_gather avg time: {t} ms")

    # Tilelang-dist
    def tilelang_ag():
        ag_buffer = pynvshmem.nvshmem_create_tensor([M_per_rank, N], torch_dtype)
        ag_buffer.copy_(local_data)
        out = pynvshmem.nvshmem_create_tensor([M, N], torch_dtype)
        out[RANK * M_per_rank : (RANK + 1) * M_per_rank, :].copy_(local_data)
        kernel(ag_buffer, out)
        pynvshmem.nvshmem_barrier_all()  # Ensure all ranks have completed
        return out

    dist.barrier(TP_GROUP)
    out, t = perf_fn(tilelang_ag, warmup, repeat)
    print(f"rank {RANK} tilelang all_gather avg time: {t} ms")

    # Check correctness
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    print(f"rank {RANK} check passed.✅")

    dist.destroy_process_group()
