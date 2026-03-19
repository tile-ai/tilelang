import torch
import torch.backends
import tilelang
from tilelang import language as T
import math


def cdiv(a, b):
    return math.ceil(a / b)


# disable tf32
torch.backends.cuda.matmul.allow_tf32 = False

m = 256
n = 1024
k = 512

total_sm = 108

torch.random.manual_seed(0)
# uniform distribution from -1 to 1
A = torch.rand(m, k, device="cuda", dtype=torch.float16) * 2 - 1
B = torch.rand(n, k, device="cuda", dtype=torch.float16) * 2 - 1

streamk_programs = total_sm
BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
two_tiles = False
M, K = A.shape
N, K = B.shape
# accumulator types
# compute grid (work to do per SM on the first wave)
num_block_m = tilelang.cdiv(M, BLOCK_SIZE_M)
num_block_n = tilelang.cdiv(N, BLOCK_SIZE_N)
iters_per_tile = tilelang.cdiv(K, BLOCK_SIZE_K)
total_tiles = num_block_m * num_block_n

# Two-tile SK + DP
streamk_tiles = total_tiles % streamk_programs
if total_tiles - streamk_tiles > streamk_programs:  # (total_tiles // total_programs > 1)
    streamk_tiles += streamk_programs

blocking_tiles = total_tiles - streamk_tiles
streamk_iters = streamk_tiles * iters_per_tile

streamk_full_tiles = streamk_iters // streamk_programs
streamk_partial_tiles = streamk_iters % streamk_programs

print(f"{total_tiles=} ")
print(f"{iters_per_tile=} ")

sm_patition_factor = max(blocking_tiles // total_sm, 1)


@tilelang.jit
def tl_matmul_streamk(
    A,
    B,
    C,
    block_M: int = BLOCK_SIZE_M,
    block_N: int = BLOCK_SIZE_N,
    block_K: int = BLOCK_SIZE_K,
    trans_A: bool = False,
    trans_B: bool = True,
    dtypeAB: T.dtype = T.float16,
    dtypeC: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    num_stages: int = 2,
    threads: int = 64,
    p_streamk_tiles: int = streamk_tiles,
    p_streamk_programs: int = streamk_programs,
    p_streamk_full_tiles: int = streamk_full_tiles,
    p_streamk_partial_tiles: int = streamk_partial_tiles,
    p_iters_per_tile: int = iters_per_tile,
    p_sm_patition_factor: int = sm_patition_factor,
    p_total_sm: int = total_sm,
):
    assert not trans_A
    _M, _N, _K = T.const("M N K")
    A: T.Tensor[[_M, _K], dtypeAB]
    B: T.Tensor[[_N, _K], dtypeAB]
    C: T.Tensor[[_M, _N], dtypeC]

    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    with T.Kernel(p_streamk_programs, threads=threads) as pid:
        A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
        B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
        A_shared_full_tiles = T.alloc_shared(A_shared_shape, dtypeAB)
        B_shared_full_tiles = T.alloc_shared(B_shared_shape, dtypeAB)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # compute first wave
        start_iter = T.alloc_fragment((1,), T.int32, "local")
        end_iter = T.alloc_fragment((1,), T.int32, "local")

        start_iter[0] = pid * p_streamk_full_tiles + T.min(pid, p_streamk_partial_tiles)
        last_iter = (pid + 1) * p_streamk_full_tiles + T.min(pid + 1, p_streamk_partial_tiles)

        while start_iter[0] < last_iter:
            end_iter[0] = T.min(
                start_iter[0] + (p_iters_per_tile - (start_iter[0] % p_iters_per_tile)),
                last_iter,
            )

            tile_id = start_iter[0] // p_iters_per_tile
            remain_iters = start_iter[0] % p_iters_per_tile
            pid_m = tile_id // T.ceildiv(_N, block_N)
            pid_n = tile_id % T.ceildiv(_N, block_N)

            T.clear(C_local)
            for kk in T.Pipelined(end_iter[0] - start_iter[0], num_stages=num_stages):
                T.copy(
                    A[pid_m * block_M, (kk + (start_iter[0] % p_iters_per_tile)) * block_K],
                    A_shared,
                )
                T.copy(
                    B[pid_n * block_N, (kk + (start_iter[0] % p_iters_per_tile)) * block_K],
                    B_shared,
                )
                T.gemm(A_shared, B_shared, C_local, transpose_B=trans_B)

            # last iteration of the tile always happens before its start on another SM
            if remain_iters == 0 and (end_iter[0] % p_iters_per_tile == 0):
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

            start_iter[0] = end_iter[0]

        # compute full tiles
        if p_sm_patition_factor > 0:
            for p in T.serial(p_sm_patition_factor):
                tile_id = pid + p_streamk_tiles + p * p_total_sm
                pid_m = tile_id // T.ceildiv(_N, block_N)
                pid_n = tile_id % T.ceildiv(_N, block_N)
                T.clear(C_local)

                for kk in T.Pipelined(T.ceildiv(_K, block_K), num_stages=1):
                    T.copy(A[pid_m * block_M, kk * block_K], A_shared_full_tiles)
                    T.copy(B[pid_n * block_N, kk * block_K], B_shared_full_tiles)
                    T.gemm(A_shared_full_tiles, B_shared_full_tiles, C_local, transpose_B=trans_B)
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])


def main():
    b_c = torch.zeros((m, n), device="cuda", dtype=torch.float16)

    tl_matmul_streamk(A, B, b_c)

    kernel = tl_matmul_streamk.compile(A, B, b_c)
    print(kernel.get_kernel_source())

    C = torch.matmul(A, B.T)

    print(b_c)
    print(C)
    torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)


def run_regression_perf():
    b_c = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    kernel = tl_matmul_streamk.compile(A, B, b_c)

    from tilelang.profiler import do_bench

    return do_bench(lambda: kernel(A, B, b_c), backend="cupti")


if __name__ == "__main__":
    main()
