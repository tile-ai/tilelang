import tilelang
import tilelang.language as T
from tilelang.backend.target import determine_target
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

import math
import argparse
import torch


def is_pow_of_2(n):
    """Return whether ``n`` is a positive power of two."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


@T.macro
def warp_shfl(local, buf, thread_elem, warp_size, round):
    """Apply the Hadamard butterfly rounds handled by one hardware warp."""
    tx = T.get_thread_binding(0)
    for i in T.serial(round):
        tx_stride = 1 << i
        another_tx = tx ^ tx_stride
        sign = (tx >> i) & 1  # get i-th lowest bit of tx, which determines the operation type for shared[tx, :]
        for j in T.Pipelined(thread_elem, num_stages=1):
            buf[j] = T.tvm_warp_shuffle(
                # CUDA uses this 32-bit active mask. HIP lowering discards the
                # mask and emits __shfl(value, src_lane, width).
                0xFFFFFFFF,
                local[j],
                another_tx % warp_size,
                warp_size,
                warp_size,
            )
            local[j] = T.if_then_else(sign == 0, local[j] + buf[j], buf[j] - local[j])


@tilelang.jit
def hadamard(A, n, dtype):
    """Build a Hadamard transform specialized for the active accelerator."""
    b = T.const("b")

    A: T.Tensor((b, n), dtype)
    B = T.empty((b, n), dtype)

    assert is_pow_of_2(n), "n must be a power of 2"
    assert 2 <= n <= 32768, "n must be in [2, 32768]"

    elem_size = {T.float32: 4, T.float16: 2, T.bfloat16: 2}[dtype]

    logN = int(math.log2(n))
    threads = [0, 1, 1, 1, 2, 4, 8, 16, 32, 32, 128, 256, 256, 256, 256, 256][logN]
    target = determine_target("auto", return_object=True)
    is_hip = target.kind.name == "hip"
    if is_hip:
        target_warp_size = target.attrs.get("thread_warp_size")
        if target_warp_size is None:
            raise RuntimeError(f"Cannot determine the HIP wavefront size for target {target}")
        hardware_warp_size = int(target_warp_size)
    else:
        hardware_warp_size = 32
    if is_hip:
        threads = max(threads, hardware_warp_size)
    thread_elem = max(1, n // threads)  # Each active thread is responsible for a chunk of elements
    thread_round = int(math.log2(thread_elem))

    warps = 1 if threads <= hardware_warp_size else threads // hardware_warp_size
    warp_size = n if is_hip and n < hardware_warp_size else threads // warps
    warp_round = int(math.log2(warp_size))

    block_round = int(math.log2(warps))

    exchange_round = n * elem_size // 32768 if n * elem_size > 32768 else 1  # Suppose we use 32KB shared memory at most
    thread_elem_in_smem = thread_elem // exchange_round if exchange_round > 1 else thread_elem

    with T.Kernel(b, threads=threads) as bx:
        local = T.alloc_local((thread_elem,), dtype)
        shared = T.alloc_shared((threads, thread_elem_in_smem), dtype)
        if is_hip:
            T.annotate_layout({shared: tilelang.layout.make_swizzled_layout(shared)})
        else:
            T.annotate_layout({shared: make_mma_swizzle_layout(shared)})
        tx = T.get_thread_binding(0)

        # 1. Load from HBM to register
        if is_hip and n < hardware_warp_size:
            T.fill(local, 0)
            if tx < n:
                local[0] = A[bx, tx]
        else:
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

        warp_shfl(local, another_val, thread_elem, warp_size, warp_round)

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
            warp_shfl(local, another_val, thread_elem, warp_size, block_round)

            # 4.3 Write to smem, swap, read from smem
            for cur_round in T.serial(exchange_round):
                exchange_base = thread_elem_in_smem * cur_round
                for j in T.vectorized(thread_elem_in_smem):
                    shared[tgt_tx, j] = local[exchange_base + j]

                for j in T.vectorized(thread_elem_in_smem):
                    local[exchange_base + j] = shared[src_tx, j]

        # 5. Write back to HBM
        if is_hip and n < hardware_warp_size:
            if tx < n:
                B[bx, tx] = local[0]
        else:
            for i in T.vectorized(thread_elem):
                B[bx, tx * thread_elem + i] = local[i]

    return B


def ref_program(x: torch.Tensor):
    """Compute a matrix-free float64 Hadamard reference."""
    assert x.ndim == 2
    dim = x.shape[-1]
    assert is_pow_of_2(dim)
    # Compute a matrix-free float64 reference. Materializing the 32768-square
    # Hadamard matrix would require 8 GiB before accounting for the matmul.
    result = x.double()
    half = 1
    while half < dim:
        pairs = result.reshape(*x.shape[:-1], -1, 2, half)
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1)
        half *= 2
    return result.reshape_as(x).to(x.dtype)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--dim", type=int, default=32768, help="Dimension")
    args = parser.parse_args(argv)

    B, D = args.batch, args.dim
    x = torch.randn((B, D), device="cuda")
    y = hadamard(x, D, T.float32)
    y_ref = ref_program(x)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
    print("All tests passed.")

    from tilelang.profiler import do_bench

    latency = do_bench(lambda: hadamard(x, D, T.float32))
    print("Tile-lang: {:.2f} ms".format(latency))


if __name__ == "__main__":
    main()
