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
    """
    Constructs a TileLang primitive function that performs a pipelined, block-tiled matrix multiplication (GEMM) with shared-memory tiling and software-managed accumulation.
    
    Parameters:
        M (int): Number of rows of the output matrix C.
        N (int): Number of columns of the output matrix C.
        K (int): Reduction dimension size.
        block_M (int): Tile size for M dimension.
        block_N (int): Tile size for N dimension.
        block_K (int): Tile size for K (reduction) dimension.
        trans_A (bool): If True, interpret input A as transposed.
        trans_B (bool): If True, interpret input B as transposed.
        in_dtype (str): Input tensor element type (TileLang dtype string).
        out_dtype (str): Output tensor element type (TileLang dtype string).
        accum_dtype (str): Accumulator element type used for intermediate accumulation.
        num_stages (int): Number of pipeline stages for K-tiling (software pipelining depth).
        threads (int): Number of threads per work-group used when emitting the kernel.
    
    Returns:
        T.PrimFunc: A TileLang prim_func named `main` that takes tensors A, B, and C and implements the specified tiled, pipelined GEMM.
    """
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
        """
            TileLang kernel implementing a blocked, pipelined GEMM that accumulates into thread-local memory and writes the result to C.
            
            Performs a tiled matrix multiplication over K in a pipelined fashion: each pipeline stage copies A and B tiles into shared memory, runs a device GEMM (gemm_v2) into a temporary accumulator (tmem) with barrier synchronization between stages, then transfers the accumulated tile into shared and finally to the global output C.
            
            Parameters:
                A: Input matrix A with shape derived from A_shape and input dtype; tile segments of A are loaded into shared memory.
                B: Input matrix B with shape derived from B_shape and input dtype; tile segments of B are loaded into shared memory.
                C: Output matrix with shape (M, N) and output dtype; receives the final accumulated results written by the kernel.
            """
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
    """
    Compute a normalized difference between two tensors based on their elementwise inner product and magnitudes.
    
    Parameters:
        x (torch.Tensor): First tensor; values will be converted to double precision.
        y (torch.Tensor): Second tensor; values will be converted to double precision.
    
    Returns:
        diff (float): Value equal to 1 - (2 * sum(x * y) / sum(x*x + y*y)), representing the dissimilarity between x and y (0 indicates perfect similarity).
    """
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


M, N, K = 4096, 4096, 8192
block_M, block_N, block_K = 64, 256, 32
trans_A, trans_B = False, True
num_stages = 2
threads = 256
for tvm_fp8_dtype in ["float8_e4m3", "float8_e5m2"]:
    for tvm_acc_dtype in ["float16", "float32"]:  # , torch.float16]:
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
        print(
            f"[{tvm_fp8_dtype} -> {tvm_acc_dtype}] Flops: {2 * M * N * K / (latency / 1e3) / 1e12} TFLOPS"
        )