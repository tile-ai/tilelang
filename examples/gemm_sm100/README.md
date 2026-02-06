# TileLang SM100 Support (Preview)

This directory contains examples for TileLang's experimental SM100 architecture support. **This is a preview version** with limited functionality.

## Current Limitations (Manual Implementation Required)

### 1. Manual TCGEN5.MMA Management
Users must manually handle TCGEN5MMA operations using:
- `T.alloc_tmem()` - Allocate Tensor Memory
- `T.gemm()` with `wg_wait=-1` - Launch TCGEN5MMA without waiting
- Manual synchronization with mbarrier

### 2. Manual mbarrier Synchronization
TCGEN5MMA is asynchronous and requires explicit synchronization:
```python
mbar = T.alloc_barrier(1)  # expect-arrive-count = 1
T.gemm(A_shared, B_shared, C_tmem, trans_A, trans_B, mbar=mbar, wg_wait=-1, clear_accum=k==0)
T.mbarrier_wait_parity(mbar, k%2)  # Manual phase calculation required
```

## Examples

### TCGEN5MMA Example (`gemm_tcgen5mma.py`)
Demonstrates TCGEN5MMA operations with:
- Tensor Memory allocation
- Manual mbarrier synchronization
- TCGEN5MMA gemm operations

### Traditional MMA Example (`gemm_mma.py`)
Shows standard MMA operations that work across architectures for comparison.

## Code Example

The following code is based on `gemm_tcgen5mma.py`, demonstrating TCGEN5MMA matrix multiplication:

```python
import torch
import tilelang
import tilelang.language as T

@T.prim_func
def main(
    A: T.Tensor((M, K), T.bfloat16),
    B: T.Tensor((N, K), T.bfloat16),
    C: T.Tensor((M, N), T.bfloat16),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        # 1. Allocate memory buffers
        A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)  # A matrix shared memory
        B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)  # B matrix shared memory
        C_tmem = T.alloc_tmem([block_M, block_N], T.float)         # TCGEN5MMA output to Tensor Memory
        mbar = T.alloc_barrier(1)                                 # mbarrier synchronization primitive

        C_local = T.alloc_fragment((block_M, block_N), T.float)   # Register storage
        C_shared = T.alloc_shared((block_M, block_N), T.bfloat16) # Output shared memory

        # 2. Main computation loop
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
            # Data loading: global memory to shared memory
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)

            # TCGEN5MMA computation: asynchronous launch, output to Tensor Memory
            T.gemm(A_shared, B_shared, C_tmem, trans_A=False, trans_B=True,
                   mbar=mbar, wg_wait=-1, clear_accum=k==0)

            # Critical: wait for TCGEN5MMA completion
            T.mbarrier_wait_parity(mbar, k%2)

        # 3. Output processing (only subset of threads)
        T.copy(C_tmem, C_local)      # Tensor Memory → registers
        T.copy(C_local, C_shared)    # registers → shared memory

        # 4. Write back to global memory
        T.copy(C_shared, C[by * block_M, bx * block_N])
```

### Compilation and Usage

```python
# Parameter setup
M, N, K = 4096, 4096, 8192
block_M, block_N, block_K = 128, 256, 128

# Compile kernel
jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,        # Required
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, # Required
})

# Run test
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
c = jit_kernel(a, b)

# Verify correctness
ref_c = (a.to(torch.float) @ b.T.to(torch.float)).to(torch.bfloat16)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# Performance benchmark
profiler = jit_kernel.get_profiler()
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
print(f"Performance: {2 * M * N * K / (latency/1e3) / 1e12:.2f} TFLOPS")
```
