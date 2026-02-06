import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T
from tilelang.distributed import init_distributed, dtype_map
import math
import argparse

tilelang.disable_cache()


def cannon(MESH, M, N, K, block_M, block_N, block_K, dtype="float16", specialize=False):
    M_local = T.ceildiv(M, MESH)
    N_local = T.ceildiv(N, MESH)
    K_local = T.ceildiv(K, MESH)
    accum_dtype = "float32"

    sm_num = 132  # 132 SMs for H100
    total_tiles = T.ceildiv(M_local, block_M) * T.ceildiv(N_local, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((2, M_local, K_local), dtype),
        B: T.Tensor((2, N_local, K_local), dtype),
        A_signal_to: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
        A_signal_from: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
        B_signal_to: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
        B_signal_from: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
        C: T.Tensor((M_local, N_local), dtype),
    ):
        grid_size = T.min(sm_num, total_tiles)
        A_rows_per_block = T.ceildiv(M_local, grid_size)
        B_cols_per_block = T.ceildiv(N_local, grid_size)
        waves = T.ceildiv(total_tiles, sm_num)
        with T.Kernel(grid_size, threads=256) as (block_id):
            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            a_peer_from = T.alloc_local([1], "int32")
            a_peer_to = T.alloc_local([1], "int32")
            b_peer_from = T.alloc_local([1], "int32")
            b_peer_to = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.get_thread_binding(0)
            a_peer_from[0] = (mype[0] + 1) % MESH + MESH * (mype[0] // MESH)
            a_peer_to[0] = (mype[0] - 1 + MESH) % MESH + MESH * (mype[0] // MESH)
            b_peer_from[0] = (mype[0] + MESH) % npes[0]
            b_peer_to[0] = (mype[0] - MESH + npes[0]) % npes[0]
            T.clear(C_local)
            for ko in T.serial(MESH):
                if tx == 0:
                    T.signal_wait_until(
                        T.address_of(A_signal_from[0]),
                        T.CmpType.GE,
                        total_tiles * ko,
                    )
                    T.signal_wait_until(
                        T.address_of(B_signal_from[0]),
                        T.CmpType.GE,
                        total_tiles * ko,
                    )

                if block_id < T.ceildiv(M_local, A_rows_per_block):
                    T.putmem_signal_nbi_block(
                        T.address_of(A[(ko + 1) % 2, A_rows_per_block * block_id, 0]),
                        T.address_of(A[ko % 2, A_rows_per_block * block_id, 0]),
                        A_rows_per_block * K_local * dtype_map[dtype].itemsize,
                        T.address_of(A_signal_to[0]),
                        1,
                        T.Amo.SIGNAL_ADD,
                        a_peer_to[0],
                    )
                if block_id < T.ceildiv(N_local, B_cols_per_block):
                    T.putmem_signal_nbi_block(
                        T.address_of(B[(ko + 1) % 2, B_cols_per_block * block_id, 0]),
                        T.address_of(B[ko % 2, B_cols_per_block * block_id, 0]),
                        B_cols_per_block * K_local * dtype_map[dtype].itemsize,
                        T.address_of(B_signal_to[0]),
                        1,
                        T.Amo.SIGNAL_ADD,
                        b_peer_to[0],
                    )

                for w in T.serial(waves):
                    bx = (grid_size * w + block_id) // T.ceildiv(N_local, block_N)
                    by = (grid_size * w + block_id) % T.ceildiv(N_local, block_N)

                    if bx < T.ceildiv(M_local, block_M) and by < T.ceildiv(N_local, block_N):
                        T.copy(C[bx * block_M, by * block_N], C_local)
                        for ki in T.Pipelined(T.ceildiv(K_local, block_K), num_stages=4):
                            T.copy(A[ko % 2, bx * block_M, ki * block_K], A_shared)
                            T.copy(B[ko % 2, by * block_N, ki * block_K], B_shared)
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                        T.copy(C_local, C[bx * block_M, by * block_N])
                        if tx == 0:
                            T.signal_op(
                                T.address_of(A_signal_from[0]),
                                1,
                                T.Amo.SIGNAL_ADD,
                                a_peer_from[0],
                            )
                            T.signal_op(
                                T.address_of(B_signal_from[0]),
                                1,
                                T.Amo.SIGNAL_ADD,
                                b_peer_from[0],
                            )

                # TODO: check if __syncthreads() is needed
                T.signal_wait_until(
                    T.address_of(A_signal_to[0]),
                    T.CmpType.GE,
                    (ko + 1) * T.ceildiv(M_local, A_rows_per_block),
                )
                T.signal_wait_until(
                    T.address_of(B_signal_to[0]),
                    T.CmpType.GE,
                    (ko + 1) * T.ceildiv(N_local, B_cols_per_block),
                )

    # TODO: fix correctness
    @T.prim_func
    def main_specialize(
        A: T.Tensor((2, M_local, K_local), dtype),
        B: T.Tensor((2, N_local, K_local), dtype),
        A_signal_to: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
        A_signal_from: T.Tensor((T.ceildiv(M, block_M),), "uint64"),
        B_signal_to: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
        B_signal_from: T.Tensor((T.ceildiv(N, block_N),), "uint64"),
        C: T.Tensor((M_local, N_local), dtype),
    ):
        # 0-compute blocks: compute
        # compute_blocks-grid_size: copy
        copy_blocks = 20
        compute_blocks = T.min(sm_num - copy_blocks, total_tiles)
        grid_size = copy_blocks + compute_blocks
        A_rows_per_block = T.ceildiv(M_local, copy_blocks)
        B_cols_per_block = T.ceildiv(N_local, copy_blocks)
        waves = T.ceildiv(total_tiles, compute_blocks)
        with T.Kernel(grid_size, threads=256) as (block_id):
            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            a_peer_from = T.alloc_local([1], "int32")
            a_peer_to = T.alloc_local([1], "int32")
            b_peer_from = T.alloc_local([1], "int32")
            b_peer_to = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.get_thread_binding(0)
            a_peer_from[0] = (mype[0] + 1) % MESH + MESH * (mype[0] // MESH)
            a_peer_to[0] = (mype[0] - 1 + MESH) % MESH + MESH * (mype[0] // MESH)
            b_peer_from[0] = (mype[0] + MESH) % npes[0]
            b_peer_to[0] = (mype[0] - MESH + npes[0]) % npes[0]
            T.clear(C_local)
            for ko in T.serial(MESH):
                if block_id >= compute_blocks:
                    if tx == 0:
                        T.signal_wait_until(
                            T.address_of(A_signal_from[0]),
                            T.CmpType.GE,
                            total_tiles * ko,
                        )
                        T.signal_wait_until(
                            T.address_of(B_signal_from[0]),
                            T.CmpType.GE,
                            total_tiles * ko,
                        )
                    T.putmem_signal_nbi_block(
                        T.address_of(A[(ko + 1) % 2, A_rows_per_block * (block_id - compute_blocks), 0]),
                        T.address_of(A[ko % 2, A_rows_per_block * (block_id - compute_blocks), 0]),
                        A_rows_per_block * K_local * dtype_map[dtype].itemsize,
                        T.address_of(A_signal_to[0]),
                        1,
                        T.Amo.SIGNAL_ADD,
                        a_peer_to[0],
                    )
                    T.putmem_signal_nbi_block(
                        T.address_of(B[(ko + 1) % 2, B_cols_per_block * (block_id - compute_blocks), 0]),
                        T.address_of(B[ko % 2, B_cols_per_block * (block_id - compute_blocks), 0]),
                        B_cols_per_block * K_local * dtype_map[dtype].itemsize,
                        T.address_of(B_signal_to[0]),
                        1,
                        T.Amo.SIGNAL_ADD,
                        b_peer_to[0],
                    )

                if block_id < compute_blocks:
                    for w in T.serial(waves):
                        bx = (compute_blocks * w + block_id) // T.ceildiv(N_local, block_N)
                        by = (compute_blocks * w + block_id) % T.ceildiv(N_local, block_N)

                        if bx < T.ceildiv(M_local, block_M) and by < T.ceildiv(N_local, block_N):
                            T.copy(C[bx * block_M, by * block_N], C_local)
                            for ki in T.Pipelined(T.ceildiv(K_local, block_K), num_stages=4):
                                T.copy(A[ko % 2, bx * block_M, ki * block_K], A_shared)
                                T.copy(B[ko % 2, by * block_N, ki * block_K], B_shared)
                                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                            T.copy(C_local, C[bx * block_M, by * block_N])
                            if tx == 0:
                                T.signal_op(
                                    T.address_of(A_signal_from[0]),
                                    1,
                                    T.Amo.SIGNAL_ADD,
                                    a_peer_from[0],
                                )
                                T.signal_op(
                                    T.address_of(B_signal_from[0]),
                                    1,
                                    T.Amo.SIGNAL_ADD,
                                    b_peer_from[0],
                                )

                    T.signal_wait_until(
                        T.address_of(A_signal_to[0]),
                        T.CmpType.GE,
                        (ko + 1) * copy_blocks,
                    )
                    T.signal_wait_until(
                        T.address_of(B_signal_to[0]),
                        T.CmpType.GE,
                        (ko + 1) * copy_blocks,
                    )

    return main_specialize if specialize else main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=16384, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--K", default=16384, type=int)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")
    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()

    WORLD_SIZE, RANK, LOCAL_RANK = init_distributed()

    MESH = math.ceil(math.sqrt(WORLD_SIZE))
    assert MESH * MESH == WORLD_SIZE, "Mesh size must match world size"

    M, N, K = args.M, args.N, args.K
    specialize = False
    block_M, block_N, block_K = 128, 256, 64
    dtype = dtype_map[args.dtype]

    M_local = math.ceil(M / MESH)
    N_local = math.ceil(N / MESH)
    K_local = math.ceil(K / MESH)

    func = cannon(MESH, M, N, K, block_M, block_N, block_K, args.dtype, specialize)
    kernel = tilelang.compile(func, pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})

    # Get CUDA Source
    if RANK == 0:
        print(kernel.get_kernel_source())

    device = torch.device(f"cuda:{RANK}")
    ref = torch.empty((M_local, N_local), dtype=dtype, device=device)
    A_ref = torch.empty((M_local, K_local), dtype=dtype, device=device)
    B_ref = torch.empty((N_local, K_local), dtype=dtype, device=device)

    if RANK == 0:
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device)
        C = A @ B.T

        c_scatter_list = []
        a_scatter_list = []
        b_scatter_list = []
        for r in range(WORLD_SIZE):
            rr, cc = divmod(r, MESH)
            c_tile = C[M_local * rr : M_local * (rr + 1), N_local * cc : N_local * (cc + 1)]
            a_tile = A[M_local * rr : M_local * (rr + 1), K_local * ((cc + rr) % MESH) : K_local * ((cc + rr) % MESH + 1)]
            b_tile = B[N_local * cc : N_local * (cc + 1), K_local * ((cc + rr) % MESH) : K_local * ((cc + rr) % MESH + 1)]

            c_scatter_list.append(c_tile.contiguous())
            a_scatter_list.append(a_tile.contiguous())
            b_scatter_list.append(b_tile.contiguous())
    else:
        c_scatter_list = None
        a_scatter_list = None
        b_scatter_list = None

    dist.scatter(tensor=ref, scatter_list=c_scatter_list, src=0)
    dist.scatter(tensor=A_ref, scatter_list=a_scatter_list, src=0)
    dist.scatter(tensor=B_ref, scatter_list=b_scatter_list, src=0)
    dist.barrier()

    A = pynvshmem.nvshmem_create_tensor([2, M_local, K_local], dtype)
    B = pynvshmem.nvshmem_create_tensor([2, N_local, K_local], dtype)
    A[0, :, :].copy_(A_ref)
    B[0, :, :].copy_(B_ref)
    A_signal_to = pynvshmem.nvshmem_create_tensor([math.ceil(M / block_M)], torch.uint64)
    A_signal_from = pynvshmem.nvshmem_create_tensor([math.ceil(M / block_M)], torch.uint64)
    B_signal_to = pynvshmem.nvshmem_create_tensor([math.ceil(N / block_N)], torch.uint64)
    B_signal_from = pynvshmem.nvshmem_create_tensor([math.ceil(N / block_N)], torch.uint64)
    A_signal_to.fill_(0)
    A_signal_from.fill_(0)
    B_signal_to.fill_(0)
    B_signal_from.fill_(0)
    C_tilelang = pynvshmem.nvshmem_create_tensor([M_local, N_local], dtype)

    kernel(A, B, A_signal_to, A_signal_from, B_signal_to, B_signal_from, C_tilelang)

    for r in range(WORLD_SIZE):
        dist.barrier()
        if r == RANK:
            if torch.allclose(C_tilelang, ref, rtol=1e-2, atol=1e-2):
                print("-" * 100)
                print(f"[Rank {RANK}] ✅ Tilelang and Torch match")
            else:
                abs_error = torch.abs(C_tilelang - ref)
                rel_error = abs_error / (torch.abs(ref) + 1e-8)

                max_abs_error = abs_error.max().item()
                max_rel_error = rel_error.max().item()
                mismatch_ratio = (abs_error > (1e-2 + 1e-2 * torch.abs(ref))).float().mean().item()

                print("-" * 100)
                print(f"[Rank {RANK}] ❌ Tilelang and Torch mismatch")
                print(f"[Rank {RANK}] ref:\n{ref}")
                print(f"[Rank {RANK}] tilelang:\n{C_tilelang}")
                print(f"[Rank {RANK}] Mismatch ratio: {mismatch_ratio:.4f}")
                print(f"[Rank {RANK}] Max absolute error: {max_abs_error:.6f}")
                print(f"[Rank {RANK}] Max relative error: {max_rel_error:.6f}")
        dist.barrier()


def bench(func, *args):
    bench_iters = 10
    torch.cuda._sleep(1000000000)

    def preprocess():
        # clear signals
        args[2].fill_(0)
        args[3].fill_(0)
        args[4].fill_(0)
        args[5].fill_(0)

    # warmup
    for _ in range(20):
        preprocess()
        _ = func(*args)

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    # bench
    st.record()
    for _ in range(bench_iters):
        preprocess()
        _ = func(*args)
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / bench_iters

    return avg_time


def reduce_local_time(local_time):
    tensor = torch.tensor([local_time], dtype=torch.float32).to("cuda")
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        mean_time = (tensor / world_size).item()
        return mean_time
    return None


total_flops = 2 * M * N * K
avg_time = reduce_local_time(bench(kernel, A, B, A_signal_to, A_signal_from, B_signal_to, B_signal_from, C_tilelang))

if RANK == 0:
    print(f"avg time of RANK {RANK}: {avg_time} ms")
    print(f"TFlops: {total_flops / avg_time * 1e-9} TFlops")
