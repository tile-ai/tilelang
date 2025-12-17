# TileLang GEMM (Matrix Multiplication) Examples

TileLang is a domain-specific language designed to simplify the process of writing GPU kernels. It provides high-level abstractions for memory allocation, scheduling, and tiling, which are critical for achieving maximum performance on modern hardware architectures like NVIDIA GPUs. This README demonstrates how to write and optimize a matrix multiplication (GEMM) kernel using TileLang.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Simple GEMM Example](#simple-gemm-example)
  - [Code Walkthrough](#code-walkthrough)
  - [Compiling and Profiling](#compiling-and-profiling)
- [Advanced GEMM Features](#advanced-gemm-features)
  - [Custom Memory Layout / Swizzling](#custom-memory-layout--swizzling)
  - [Parallel Copy and Auto-Pipelining](#parallel-copy-and-auto-pipelining)
  - [Rasterization for L2 Cache Locality](#rasterization-for-l2-cache-locality)
- [Enhanced GEMM Example with Annotations](#enhanced-gemm-example-with-annotations)
- [Verifying Correctness](#verifying-correctness)
- [Fine-grained MMA Computations](#fine-grained-mma-computations)
  - [Example Workflow](#example-workflow)
  - [Summary](#summary)
- [References](#references)

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** with a recent CUDA toolkit installed
- **PyTorch** (optional, for easy correctness verification)
- **tilelang**
- **bitblas** (optional; used for swizzle layout utilities in the advanced examples)

### Installation

```bash
pip install tilelang bitblas
```

*(Adjust accordingly if you are installing from source or using a different environment.)*

---

## Simple GEMM Example

Below is a basic matrix multiplication (GEMM) example demonstrating how TileLang handles buffer allocation, tiling, and kernel dispatch. For simplicity, we'll multiply two 1024×1024 matrices using 128 threads/block.

```python
import tilelang
from tilelang import Profiler
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Define a grid with enough blocks to cover M×N
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):

            # Allocate shared memory for the current tile of A and B
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Allocate a local (register) fragment for partial accumulations
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize the local accumulation buffer to zero
            T.clear(C_local)

            # Loop over the K dimension in block_K chunks, using a 3-stage pipeline
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy from global memory to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Perform a matrix multiply-accumulate on the tile
                T.gemm(A_shared, B_shared, C_local)

            # Copy the accumulated result from local memory (C_local) to global memory (C)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Code Walkthrough

1. **Define the Kernel Launch Configuration:**
   ```python
   with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
   ```
   This creates a grid of blocks (ceildiv(N, block_N) in x-dimension, ceildiv(M, block_M) in y-dimension), each with 128 threads.

2. **Shared Memory Allocation:**
   ```python
   A_shared = T.alloc_shared((block_M, block_K), dtype)
   B_shared = T.alloc_shared((block_K, block_N), dtype)
   ```
   Tiles of \(A\) and \(B\) are loaded into these shared memory buffers for faster access.

3. **Local Fragment Accumulation:**
   ```python
   C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
   ```
   Partial results are stored in registers (or local memory) to reduce writes to global memory.

4. **Pipelined Loading and GEMM:**
   ```python
   for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
       T.copy(...)
       T.gemm(...)
   ```
   Loads blocks of \(A\) and \(B\) in a pipelined fashion (up to 3 stages). This exploits overlap of data transfer and computation.

5. **Copy Out the Results:**
   ```python
   T.copy(C_local, C[by * block_M, bx * block_N])
   ```
   Writes the final computed tile from registers/shared memory to global memory.

### Compiling and Profiling

```python
func = matmul(1024, 1024, 1024, 128, 128, 32)
print(func)  # Prints an IR-like representation of the TileLang kernel

artifact = tilelang.lower(func)

profiler = Profiler(artifact.rt_mod, artifact.params, result_idx=[2])

import torch
a = torch.randn(1024, 1024).cuda().half()
b = torch.randn(1024, 1024).cuda().half()

c = profiler(a, b)
ref_c = a @ b

# Validate results
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# Get CUDA Kernel Source
print(artifact.kernel_source)
```

---

## Advanced GEMM Features

### Custom Memory Layout / Swizzling

**Swizzling** rearranges data in shared memory or global memory to mitigate bank conflicts, improve cache utilization, and better match the GPU’s warp execution pattern. TileLang provides helper functions like `make_swizzle_layout` to annotate how buffers should be laid out in memory.

### Parallel Copy and Auto-Pipelining

- **Parallel Copy** allows you to distribute the copy of a block tile across all threads in a block, speeding up the transfer from global memory to shared memory.
- **Auto-Pipelining** uses multiple stages to overlap copying with computation, reducing idle cycles.

### Rasterization for L2 Cache Locality

Enabling **swizzle (rasterization)** at the kernel level can improve data reuse and reduce cache thrashing in L2. This is especially important when matrices are large.

---

## Enhanced GEMM Example with Annotations

Below is a more advanced snippet that showcases how to apply memory layouts, enable swizzling, and parallelize the copy operations to maximize performance:

```python
import tilelang.language as T
# `make_mma_swizzle_layout` is a python-defined layout function
# that helps align data for MMA (Matrix Multiply-Accumulate) operations.
from tilelang.intrinsics import make_mma_swizzle_layout as make_swizzle_layout

def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Allocate shared and local fragments
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Annotate memory layout
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Enable swizzle-based rasterization for better L2 locality
            T.use_swizzle(panel_size=10, enable=True)

            # Clear the local accumulation buffer
            T.clear(C_local)

            # Pipelined iteration over K dimension
            for idx in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, idx * block_K], A_shared)

                # Parallel copy tile of B
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[idx * block_K + ko, bx * block_N + j]

                # Perform local GEMM on the shared-memory tiles
                T.gemm(A_shared, B_shared, C_local)

            # Copy the result tile back
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

**Key Differences vs. Basic Example**
1. **`T.annotate_layout(...)`**: Annotates how data should be organized in shared memory (swizzling).
2. **`T.use_swizzle(...)`**: Enables swizzle-based rasterization.
3. **Parallel Copy Loop** with `T.Parallel(...)`: Distributes global-to-shared copy across all threads, potentially vectorizing load/store instructions.

---

## Verifying Correctness

Once you compile and load your kernel into a runtime module (`rt_mod`), you can use tools like **PyTorch** to easily create random matrices on the GPU, run your TileLang kernel, and compare the results to a reference implementation (e.g., `torch.matmul` or `@` operator).

```python
import torch

# Suppose your compiled kernel is in rt_mod
profiler = Profiler(rt_mod, params, result_idx=[2])

A = torch.randn(1024, 1024).cuda().half()
B = torch.randn(1024, 1024).cuda().half()

C_tilelang = profiler(A, B)
C_ref = A @ B

torch.testing.assert_close(C_tilelang, C_ref, rtol=1e-2, atol=1e-2)
print("Results match!")
```

---

## Fine-grained MMA Computations

For advanced users who require full control over warp-level matrix multiplication operations, TileLang allows you to specify fine-grained MMA (Matrix Multiply-Accumulate) computations in a manner similar to writing raw CUDA. While higher-level abstractions like `T.gemm(...)` or automatic MMA emitters are sufficient for many use cases, specialized workloads (for example, dequantize gemm may require fine-grained layout transformation on shared to register stage) may benefit from explicitly controlling each MMA instruction, the data layout, and the synchronization points.

### Example Workflow

```python
@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == T.int32:
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_local,
                C_shared,
            )

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]
```

1. **Set Up Tile Sizes and Thread Bindings**
   Just like in CUDA, you will typically start by defining how many warps or threads per block you want and how your matrix is subdivided. In TileLang, this is done via `T.Kernel(...)` and `T.thread_binding(...),` which ensure that the correct number of threads are active, and each thread is bound to a specific role (e.g., warp ID or lane ID).

2. **Allocate Warp-local Fragments**
   Instead of using a single shared buffer for partial sums, you allocate local buffers (register fragments) to hold sub-blocks of matrices \(A\) and \(B\). In TileLang, this is done with something like:
   ```python
   A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
   B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
   C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)
   ```
   Each of these `local` allocations represents a region of per-thread storage, which collectively forms the warp’s register tiles.

3. **Load Data via `ldmatrix`**
   Fine-grained loading instructions allow you to specify exactly how data moves from shared memory to the warp-level fragments. In the example below, `mma_emitter.ldmatrix_a()` and `.ldmatrix_b()` are higher-level wrappers around warp-synchronous intrinsics. You can write your own load logic as well:
   ```python
   for ki in T.serial(0, (block_K // micro_size_k)):
       # Warp-synchronous load for A
       mma_emitter.ldmatrix_a(A_local, A_shared, ki)

       # Warp-synchronous load for B
       mma_emitter.ldmatrix_b(B_local, B_shared, ki)
   ```
   Internally, these calls orchestrate how each thread in the warp issues the correct load instructions, performs address calculations, and stores the data into registers.

4. **Perform the MMA Instruction**
   After loading sub-tiles (fragments), the warp executes the `mma` instruction. This operation is essentially:
   \[
     C_{\text{local}} \;+=\; A_{\text{local}} \;\times\; B_{\text{local}}
   \]
   where each thread in the warp calculates a small portion of the final tile. For instance:
   ```python
   mma_emitter.mma(A_local, B_local, C_local)
   ```
   Under the hood, this translates into Tensor Core instructions (e.g., `wmma.mma.sync` in PTX), which process multiple data elements per warp in parallel.

5. **Store Results via `stmatrix`**
   Finally, you write the results from the warp-level fragments back to shared memory or global memory. This step might happen multiple times in a loop or just once at the end. The code snippet:
   ```python
   mma_emitter.stmatrix(C_local, C_shared)
   ```
   orchestrates the warp-synchronous stores, ensuring each thread places the correct fragment element into the correct location of the shared or global buffer.

### Summary

By combining warp-synchronous intrinsics (`ldmatrix`, `mma`, `stmatrix`) with manual thread bindings and memory allocations, you can replicate the control and performance of raw CUDA at the TileLang level. This approach is best suited for expert users who are comfortable with GPU warp-level programming, since it does require a deep understanding of hardware concurrency, memory hierarchies, and scheduling. However, the payoff can be significant for performance-critical paths, where every byte of bandwidth and every cycle of latency must be carefully orchestrated.

---

## References

- [NVIDIA CUTLASS Library](https://github.com/NVIDIA/cutlass): A collection of high-performance CUDA C++ template abstractions for GEMM.
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html): Official documentation for CUDA.
- [PyTorch Documentation](https://pytorch.org/docs): For verifying correctness via CPU or GPU-based matmul.
