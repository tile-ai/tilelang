import torch
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.profiler import TensorSupplyType
from tilelang.distributed import init_distributed
import argparse
import random

tilelang.disable_cache()


def all_to_all(PE_num, TOKEN_NUM, TOPK, HIDDEN, EXPERT_NUM, dtype="float16"):
    EXPERTS_PER_RANK = EXPERT_NUM // PE_num

    @T.prim_func
    def main(
        data_src: T.Tensor((TOKEN_NUM * TOPK, HIDDEN), "float16"),
        # split_src: T.Tensor((M, HIDDEN), "float16"),
        signal: T.Tensor((PE_num,), "uint64"),
        splits_cumsum: T.Tensor((EXPERT_NUM + 1,), "int32"),
        data_dst: T.Tensor((TOKEN_NUM * TOPK, HIDDEN), "float16"),
        # split_dst: T.Tensor((M * PE_num, HIDDEN), "float16"),
    ):
        with T.Kernel(PE_num, threads=128) as (bx):
            peer = bx
            tx = T.thread_binding(128, thread="threadIdx.x")

            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            m_start = T.alloc_local([1], "int32")
            m_end = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()
            m_start[0] = splits_cumsum[peer * EXPERTS_PER_RANK]
            m_end[0] = splits_cumsum[(peer + 1) * EXPERTS_PER_RANK]

            T.putmem_nbi_block(
                T.address_of(data_dst[0, 0]), T.address_of(data_src[m_start[0], 0]), (m_end[0] - m_start[0]) * HIDDEN * 2, peer
            )

            T.fence()

            if tx == 0:
                T.signal_op(
                    T.address_of(signal[mype[0]]),
                    99,
                    9,
                    peer,
                )
                T.signal_wait_until(
                    T.address_of(signal[peer]),
                    0,
                    99,
                )

    return main


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


args = parse_args()

# M, N, block_M, block_N = 32, 32, 32, 32
# dtype = torch.float16
# nelems = M * PE_num * N

WORLD_SIZE, RANK, LOCAL_RANK = init_distributed()
PE_num = WORLD_SIZE

func = all_to_all(PE_num, args.M, args.topk, args.N, args.G)
kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True})

# Get CUDA Source
if RANK == 0:
    print(kernel.get_kernel_source())

profiler = kernel.get_profiler(tensor_supply_type=TensorSupplyType.Randn)

ref_tensor = torch.randn(args.M * args.topk * PE_num, args.N, dtype=torch.float16).cuda()

token_num = args.M
exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

# print("exp_indices:", exp_indices)
# print("splits_gpu_cur_rank:", splits_gpu_cur_rank)
# print("split_cumsum:", split_cumsum)

data_src = pynvshmem.nvshmem_create_tensor([args.M * args.topk, args.N], torch.float16)
data_src[:].copy_(ref_tensor[args.M * args.topk * RANK : args.M * args.topk * (RANK + 1), :])

splits_cumsum = pynvshmem.nvshmem_create_tensor([args.G + 1], torch.int32)
splits_cumsum[:].copy_(split_cumsum)

signal = pynvshmem.nvshmem_create_tensor([PE_num], torch.uint64)
signal[:].fill_(0)
data_dst = pynvshmem.nvshmem_create_tensor([args.M * args.topk, args.N], torch.float16)

print("data_src:", data_src)
print("splits_cumsum:", splits_cumsum)
out = kernel(data_src, signal, splits_cumsum, data_dst)
print("out:", data_dst)


def bench(func, *args):
    bench_iters = 1
    torch.cuda._sleep(1000000000)
    # warmup
    # for _ in range(20):
    #     _ = func(*args)

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    # bench
    st.record()
    for _ in range(bench_iters):
        _ = func(*args)
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / bench_iters

    return avg_time


avg_time = bench(kernel, data_src, signal, splits_cumsum, data_dst)
print(f"avg time of RANK {RANK}: {avg_time} ms")
