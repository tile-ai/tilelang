import argparse
import itertools
import tilelang
import tilelang.language as T
from tilelang.engine.param import KernelParam
from tilelang.utils.tensor import get_tensor_supply, TensorSupplyType
import torch
from typing import List
from tilelang.profiler import do_bench

DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_K = 32
DEFAULT_NUM_STAGES = 2
DEFAULT_THREAD_NUM = 128
DEFAULT_ENABLE_RASTERIZATION = True

parser = argparse.ArgumentParser(description="Autotuned BlockSparse MatMul Benchmark")
parser.add_argument("--m", type=int, default=1024, help="Matrix dimension M")
parser.add_argument("--n", type=int, default=1024, help="Matrix dimension N")
parser.add_argument("--k", type=int, default=1024, help="Matrix dimension K")
parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity ratio (0-1)")
parser.add_argument("--use_autotune", action="store_true", default=False, help="Whether to use autotune")

args, _ = parser.parse_known_args()
M, N, K = args.m, args.n, args.k
sparsity = args.sparsity
use_autotune = args.use_autotune
default_tensor_supply = get_tensor_supply(TensorSupplyType.Auto)

print(f"Running BlockSparse MatMul Benchmark for M={M}, N={N}, K={K}")
print(f"Target Block Sparsity: {sparsity}")
print(f"Using Autotuner: {use_autotune}\n")


def get_configs():
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]

    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num, enable_rasterization))

    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "num_stages": c[3],
            "thread_num": c[4],
            "enable_rasteration": c[5],
        }
        for c in _configs
    ]


def ref_program(A, B, BlockMask, block_M, block_N, block_K):
    ref_c = torch.zeros((M, N), dtype=torch.float16, device=A.device)
    for i in range(M // block_M):
        for j in range(N // block_N):
            accu = torch.zeros((block_M, block_N), dtype=torch.float32, device=A.device)
            for k in range(K // block_K):
                if BlockMask[i, j, k]:
                    accu += A[i * block_M : (i + 1) * block_M, k * block_K : (k + 1) * block_K].to(torch.float32) @ B[
                        k * block_K : (k + 1) * block_K, j * block_N : (j + 1) * block_N
                    ].to(torch.float32)
            ref_c[i * block_M : (i + 1) * block_M, j * block_N : (j + 1) * block_N] = accu.to(torch.float16)
    return ref_c


def supply_program(params: List[KernelParam]):
    input_tensors = []

    for p in params:
        # Check if the kernel parameter is BlockMask tensor.
        # Here, BlockMask is uniquely identified by having 3 dimensions.
        if len(p.shape) != 3:
            # For non-BlockMask tensors, use the default tensor generation logic.
            input_tensors.append(default_tensor_supply(p))
        else:
            # For BlockMask tensor, randomly set elements to True based on desired
            # sparsity level.
            block_mask = torch.zeros(p.shape, dtype=torch.bool, device=torch.cuda.current_device())
            block_mask[:, :, :] = torch.rand(p.shape) > sparsity
            input_tensors.append(block_mask)

    return input_tensors


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(out_idx=[-1])
def blocksparse_matmul(
    M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32
):
    block_mask_shape = (M // block_M, N // block_N, K // block_K)

    @T.prim_func
    def block_sparse_matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, "bool"),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if BlockMask[by, bx, k]:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return block_sparse_matmul


def main():
    # Initialize input matrices A and B on the GPU with half precision
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    if args.use_autotune:
        # Run the autotuner to find the best kernel configuration and performance
        # get_best_config is expected to return an object containing the compiled kernel,
        # the best configuration found, latency, and reference latency.
        kernel = blocksparse_matmul(M, N, K)

        best_config = kernel.config
        best_latency = kernel.latency
        block_M, block_N, block_K = best_config["block_M"], best_config["block_N"], best_config["block_K"]

        print(f"Best Config: {best_config}")
        print(f"Sparsity Ratio: {sparsity}")
        print(f"Best Kernel Latency: {best_latency:.6f} ms")
    else:
        kernel = blocksparse_matmul(
            M,
            N,
            K,
            block_M=DEFAULT_BLOCK_M,
            block_N=DEFAULT_BLOCK_N,
            block_K=DEFAULT_BLOCK_K,
            num_stages=DEFAULT_NUM_STAGES,
            thread_num=DEFAULT_THREAD_NUM,
            enable_rasteration=DEFAULT_ENABLE_RASTERIZATION,
        )
        block_M, block_N, block_K = DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K
        print(f"Using default kernel with block size ({block_M}, {block_N}, {block_K})")
    # Create block mask with desired sparsity
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity

    # Run the compiled kernel (either tuned or default) with the inputs
    c = kernel(a, b, block_mask)

    # Compute the reference result using the naive PyTorch implementation
    ref_c = ref_program(a, b, block_mask, block_M, block_N, block_K)

    try:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        print("✅ Results are close! Verification successful.")
    except AssertionError as e:
        print("❌ Verification FAILED: Results differ significantly.")
        print(e)


def run_regression_perf():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(K, N).cuda().half()

    kernel = blocksparse_matmul(
        M,
        N,
        K,
        block_M=DEFAULT_BLOCK_M,
        block_N=DEFAULT_BLOCK_N,
        block_K=DEFAULT_BLOCK_K,
        num_stages=DEFAULT_NUM_STAGES,
        thread_num=DEFAULT_THREAD_NUM,
        enable_rasteration=DEFAULT_ENABLE_RASTERIZATION,
    )
    block_M, block_N, block_K = DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K
    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).cuda() > sparsity

    def run_kernel_only():
        kernel(a, b, block_mask)

    return do_bench(run_kernel_only, warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    main()
