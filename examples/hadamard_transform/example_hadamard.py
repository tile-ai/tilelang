import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

import math
import argparse
import torch
from torch.nn import functional as F
import scipy


def is_pow_of_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


@tilelang.jit(out_idx=[1])
def hadamard(b, n, dtype):
    assert is_pow_of_2(n), "n must be a power of 2"
    assert 2 <= n <= 32768, "n must be in [2, 32768]"
    elem_size = {T.float32: 4, T.float16: 2, T.bfloat16: 2}[dtype]

    logN = int(math.log2(n))
    threads = [0, 1, 1, 1, 2, 4, 8, 16, 32, 32, 128, 256, 256, 256, 256, 256][logN]
    thread_elem = n // threads  # Each thread is responsible for a chunk of elements
    thread_round = int(math.log2(thread_elem))

    warps = 1 if threads <= 32 else threads // 32
    warp_round = int(math.log2(threads / warps))
    warp_size = threads // warps

    block_round = int(math.log2(warps))

    exchange_round = n * elem_size // 32768 if n * elem_size > 32768 else 1  # Suppose we use 32KB shared memory at most
    thread_elem_in_smem = thread_elem // exchange_round if exchange_round > 1 else thread_elem

    # debug log
    # print(f'{threads=}, {thread_round=}')
    # print(f'{warps=}, {warp_round=}, {warp_size=}')
    # print(f'{block_round=}')
    # print(f'{exchange_round=}')

    @T.macro
    def warp_shfl(local: T.Tensor((thread_elem,), dtype), buf: T.Tensor((thread_elem,), dtype), round: int):
        tx = T.get_thread_binding(0)
        for i in T.serial(round):
            tx_stride = 1 << i
            another_tx = tx ^ tx_stride
            sign = (tx >> i) & 1  # get i-th lowest bit of tx, which determines the operation type for shared[tx, :]

            for j in T.Pipelined(thread_elem, num_stages=1):
                buf[j] = T.tvm_warp_shuffle(
                    0xFFFFFFFF,  # mask of all threads
                    local[j],
                    another_tx % warp_size,
                    warp_size,
                    warp_size,
                )
                local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])

    @T.prim_func
    def main(A: T.Tensor((b, n), dtype), B: T.Tensor((b, n), dtype)):
        with T.Kernel(b, threads=threads) as bx:
            local = T.alloc_local((thread_elem,), dtype)
            shared = T.alloc_shared((threads, thread_elem_in_smem), dtype)
            T.annotate_layout({shared: make_mma_swizzle_layout(shared)})
            tx = T.get_thread_binding(0)

            # 1. Load from HBM to register
            for i in T.vectorized(thread_elem):
                local[i] = A[bx, tx * thread_elem + i]

            # 2. Hadamard inside thread, n<=8
            for i in T.serial(thread_round):
                chunksize = 1 << (i + 1)
                chunknum = thread_elem // chunksize
                for j in T.serial(chunknum):
                    chunkbase = j * chunksize
                    for k in T.serial(chunksize // 2):
                        local[chunkbase + k] = local[chunkbase + k] + local[chunkbase + k + chunksize // 2]
                        local[chunkbase + k + chunksize // 2] = local[chunkbase + k] - 2 * local[chunkbase + k + chunksize // 2]

            # 3. Hadamard inside warp, n<=512
            # In warp level, we rely on warp shuffle to exchange data inside each warp, without using shared memory
            another_val = T.alloc_local((thread_elem,), dtype)

            warp_shfl(local, another_val, warp_round)

            # 4. Hadamard inside block, n<=32768
            # Only exchange once for n<=8192, since shared mem can hold all elems
            if block_round > 0:
                warp_id = tx // warp_size
                lane_id = tx % warp_size
                src_tx = warp_id * warp_size + lane_id
                tgt_warp_id = tx % warps
                tgt_lane_id = tx // warps
                tgt_tx = tgt_warp_id * warp_size + tgt_lane_id

                # 4.1 Write to smem, swap, read from smem
                for cur_round in T.serial(exchange_round):
                    exchange_base = thread_elem_in_smem * cur_round
                    for j in T.vectorized(thread_elem_in_smem):
                        shared[src_tx, j] = local[exchange_base + j]

                    for j in T.vectorized(thread_elem_in_smem):
                        local[exchange_base + j] = shared[tgt_tx, j]

                # 4.2 Warp shuffle
                warp_shfl(local, another_val, block_round)

                # 4.3 Write to smem, swap, read from smem
                for cur_round in T.serial(exchange_round):
                    exchange_base = thread_elem_in_smem * cur_round
                    for j in T.vectorized(thread_elem_in_smem):
                        shared[tgt_tx, j] = local[exchange_base + j]

                    for j in T.vectorized(thread_elem_in_smem):
                        local[exchange_base + j] = shared[src_tx, j]

            # 5. Write back to HBM
            for i in T.vectorized(thread_elem):
                B[bx, tx * thread_elem + i] = local[i]

    return main


def ref_program(x: torch.Tensor):
    assert x.ndim == 2
    dim = x.shape[-1]
    assert is_pow_of_2(dim)
    return F.linear(x, torch.tensor(scipy.linalg.hadamard(dim, dtype=float), dtype=x.dtype, device=x.device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--dim", type=int, default=32768, help="Dimension")
    args = parser.parse_args()

    B, D = args.batch, args.dim
    x = torch.randn((B, D), device="cuda")
    kernel = hadamard(B, D, T.float32)
    y = kernel(x)
    y_ref = ref_program(x)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("All tests passed.")

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    latency = profiler.do_bench(warmup=100)
    print("Tile-lang: {:.2f} ms".format(latency))


if __name__ == "__main__":
    main()
