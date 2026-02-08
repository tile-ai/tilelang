"""
A complex TileLang program with lots of redundant code and a bug that triggers an error.
AutoDD will simplify it to the minimal code needed to reproduce the error.

This example demonstrates how AutoDD can help developers quickly isolate bugs
in complex TileLang programs by automatically removing irrelevant code.

To run AutoDD on this file:
    python -m tilelang.autodd tilelang_buggy.py --err-msg "Dimension mismatch" -o minimized.py -j 4

The bug in this file: B_shared has shape (block_M, block_N) instead of (block_K, block_N),
causing a GEMM dimension mismatch error.
"""

import tilelang
import tilelang.language as T
import torch


# Useless helper function - will be removed by AutoDD
def calculate_optimal_block_size(M, N, K):
    """Calculate optimal block size - this function is completely useless"""
    options = [32, 64, 128, 256]
    best = 128
    for opt in options:
        if M % opt == 0 and N % opt == 0:
            best = opt
            break
    return best, best, 32


def get_memory_requirements(M, N, K, block_M, block_N, block_K, dtype_size=2):
    """Calculate memory requirements - completely useless"""
    shared_mem_a = block_M * block_K * dtype_size
    shared_mem_b = block_K * block_N * dtype_size
    total_shared = shared_mem_a + shared_mem_b
    return total_shared


def validate_parameters(M, N, K, block_M, block_N, block_K):
    """Validate parameters - redundant check"""
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError("Matrix dimensions must be positive")
    if block_M <= 0 or block_N <= 0 or block_K <= 0:
        raise ValueError("Block sizes must be positive")
    if M % block_M != 0:
        print(f"Warning: M ({M}) not divisible by block_M ({block_M})")
    if N % block_N != 0:
        print(f"Warning: N ({N}) not divisible by block_N ({block_N})")
    if K % block_K != 0:
        print(f"Warning: K ({K}) not divisible by block_K ({block_K})")
    return True


class MatmulConfig:
    """Configuration class - increases code complexity but is actually useless"""

    def __init__(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K
        self.block_M = 128
        self.block_N = 128
        self.block_K = 32
        self.num_stages = 3
        self.threads = 128
        self.dtype = "float16"
        self.accum_dtype = "float32"

    def get_grid_size(self):
        grid_x = (self.N + self.block_N - 1) // self.block_N
        grid_y = (self.M + self.block_M - 1) // self.block_M
        return grid_x, grid_y

    def get_shared_memory_size(self):
        return get_memory_requirements(self.M, self.N, self.K, self.block_M, self.block_N, self.block_K)

    def validate(self):
        return validate_parameters(self.M, self.N, self.K, self.block_M, self.block_N, self.block_K)


def create_reference_output(a, b, activation="relu"):
    """Create reference output - not actually used in verification"""
    result = a @ b
    if activation == "relu":
        result = torch.relu(result)
    elif activation == "gelu":
        result = torch.nn.functional.gelu(result)
    elif activation == "sigmoid":
        result = torch.sigmoid(result)
    return result


def benchmark_pytorch(M, N, K, num_iters=10, warmup=5):
    """PyTorch benchmark - not used"""
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    # Benchmark
    import time

    start = time.time()
    for _ in range(num_iters):
        _ = a @ b
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / num_iters * 1000  # ms


# Main TileLang kernel - contains a BUG: GEMM shape mismatch!
@tilelang.jit
def buggy_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            # BUG: the first dimension of B_shared should be block_K, but block_M is used here!
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Wrong shape!
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Allocate some useless temp variables
            temp_buffer = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Zero out
            T.clear(C_local)
            T.clear(temp_buffer)

            # Main loop
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy a tile of A
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy a tile of B - shape can mismatch here too
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # GEMM computation - shape mismatch will cause an error
                # A_shared: (block_M, block_K)
                # B_shared: (block_M, block_N) <- should be (block_K, block_N)
                T.gemm(A_shared, B_shared, C_local)

            # ReLU activation
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            # Some useless postprocessing
            for i, j in T.Parallel(block_M, block_N):
                if temp_buffer[i, j] > 0:
                    C_local[i, j] = C_local[i, j] + 0.0

            # Write back result
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


def run_kernel(config):
    """Run kernel - includes extra redundant logic"""
    # Validate parameters
    config.validate()

    # Get config
    M, N, K = config.M, config.N, config.K
    block_M, block_N, block_K = config.block_M, config.block_N, config.block_K

    # Calculate some useless statistics
    grid_size = config.get_grid_size()
    shared_mem = config.get_shared_memory_size()
    print(f"Grid size: {grid_size}")
    print(f"Shared memory: {shared_mem} bytes")

    # Create test data
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)

    # Compile and run kernel - will trigger the BUG here
    kernel = buggy_matmul(M, N, K, block_M, block_N, block_K)
    kernel(a, b, c)

    # Validate results (if it can get here)
    ref_c = torch.relu(a @ b)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    return c


def main():
    # Useless printing
    print("=" * 60)
    print("TileLang Matmul Kernel Test")
    print("=" * 60)

    # Create config
    M, N, K = 512, 512, 512
    config = MatmulConfig(M, N, K)

    # Calculate some useless values
    optimal_block = calculate_optimal_block_size(M, N, K)
    print(f"Optimal block size: {optimal_block}")

    # Run PyTorch benchmark - result is not used
    # pytorch_time = benchmark_pytorch(M, N, K)
    # print(f"PyTorch time: {pytorch_time:.3f} ms")

    # Run our kernel - will trigger the error here
    try:
        result = run_kernel(config)
        print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"Error: {e}")
        raise

    print("Done!")


if __name__ == "__main__":
    main()
