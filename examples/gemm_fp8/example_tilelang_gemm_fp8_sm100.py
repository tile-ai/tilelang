import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm_v2(
                    A_shared,
                    B_shared,
                    C_tmem,
                    trans_A,
                    trans_B,
                    mbar=mbar,
                    wg_wait=-1,
                    clear_accum=(k == 0),
                )
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


M, N, K = 4096, 4096, 8192
block_M, block_N, block_K = 64, 256, 32
trans_A, trans_B = False, True
num_stages = 2
threads = 256
for tvm_fp8_dtype in [T.float8_e4m3fn, T.float8_e5m2]:
    for tvm_acc_dtype in [T.float16, T.float32]:  # , torch.float16]:
        torch_fp8_dtype = map_torch_type(tvm_fp8_dtype)
        torch_acc_dtype = map_torch_type(tvm_acc_dtype)
        print(f"running {tvm_fp8_dtype} -> {tvm_acc_dtype}")
        in_dtype, out_dtype, accum_dtype = tvm_fp8_dtype, tvm_acc_dtype, tvm_acc_dtype

        func = matmul(
            M,
            N,
            K,
            block_M,
            block_N,
            block_K,
            trans_A,
            trans_B,
            in_dtype,
            out_dtype,
            accum_dtype,
            num_stages,
            threads,
        )
        jit_kernel = tilelang.compile(
            func,
            out_idx=[2],
            target="cuda",
            pass_configs={
                tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
                tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                tilelang.PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT: True,
            },
        )
        # jit_kernel.export_ptx("./dump.ptx")
        # jit_kernel.export_sources("./dump.cu")

        a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch_fp8_dtype)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch_fp8_dtype)

        c = jit_kernel(a, b)
        ref_c = (a.to(torch.half) @ b.T.to(torch.half)).float()
        c = c.float()
        diff = calc_diff(c, ref_c)
        # assert diff < 1e-3, f"{diff}"
        print(f"[{tvm_fp8_dtype} -> {tvm_acc_dtype}] diff = {diff}")

        profiler = jit_kernel.get_profiler()
        latency = profiler.do_bench()
        print(f"[{tvm_fp8_dtype} -> {tvm_acc_dtype}] Latency: {latency} ms")
        print(f"[{tvm_fp8_dtype} -> {tvm_acc_dtype}] Flops: {2 * M * N * K / (latency / 1e3) / 1e12} TFLOPS")
