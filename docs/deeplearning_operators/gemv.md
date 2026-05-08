# General Matrix-Vector Multiplication (GEMV)
===========================================

<div style="text-align: left;">
    <em>Contributor: </em> <a href="https://github.com/botbw">@botbw</a>
</div>

:::{warning}
   This document is still **experimental** and may be incomplete.
   Suggestions and improvements are highly encouraged—please submit a PR!
:::

:::{tip}
Example code can be found at `examples/gemv/example_gemv.py`.
:::

General matrix-vector multiplication (GEMV) can be viewed as a specialized case of general matrix-matrix multiplication (GEMM). It plays a critical role in deep learning, especially during the inference phase of large language models. In this tutorial, we will optimize GEMV from a thread-level perspective step by step using `TileLang`.

## Triton Implementation
When implementing a GEMV kernel, you might start with a high-level approach using a tool like `Triton`.

A simple Triton kernel for GEMV might look like this:
```python
@triton.jit
def _gemv_naive(
    x_ptr, A_ptr, y_ptr,
    N, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask = offs_k < K
    a_ptrs = A_ptr + n * K + offs_k
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offs_k, mask=mask, other=0.0)
    dot = tl.sum(a_vals * x_vals, axis=0)
    tl.store(y_ptr + n, dot)
```

`Triton` is straightforward to use, as it operates at the block level. However, this approach may not allow for fine-grained thread-level optimization. In this tutorial, we will demonstrate how to write an optimized GEMV kernel in `TileLang` that exposes more low-level control.

## Naive Implementation in TileLang
If you have a basic understanding of CUDA C, it is natural to start with a naive GEMV kernel by adapting a GEMM tiling strategy. You can think of GEMV as a `(1, k) * (k, n)` GEMM. Below is a simple example:

```python
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            tn = T.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn,
                                                                            tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main
```

And your kernel will be compiled into CUDA by `TileLang` (in `~/.tilelang/cache`):

```C++
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_reg[1];
  __shared__ uint64_t _mbarrier[2];
  if (((int)threadIdx.x) == 0) {
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int bk = 0; bk < 8; ++bk) {
      tl::mbarrier_wait(_mbarrier[1], ((bk & 1) ^ 1));
      for (int tk = 0; tk < 128; ++tk) {
        ((half_t*)buf_dyn_shmem)[tk] = A[((bk * 128) + tk)];
        ((half_t*)buf_dyn_shmem)[(((((int)threadIdx.x) * 128) + tk) - 16256)] = B[(((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (bk * 128)) + tk) - 131072)];
      }
      tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(_mbarrier[0]);
      tl::mbarrier_arrive(_mbarrier[0]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    C_reg[0] = 0.000000e+00f;
    for (int bk_1 = 0; bk_1 < 8; ++bk_1) {
      tl::mbarrier_wait(_mbarrier[0], (bk_1 & 1));
      for (int tk_1 = 0; tk_1 < 128; ++tk_1) {
        C_reg[0] = (C_reg[0] + (((float)((half_t*)buf_dyn_shmem)[tk_1]) * ((float)((half_t*)buf_dyn_shmem)[(((((int)threadIdx.x) * 128) + tk_1) + 128)])));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[1]);
    }
    C[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((half_t)C_reg[0]);
  }
}
```

In this design, the first 128 threads act as the data producer and the last 128 threads as the consumer within a block (assuming a 1D block).

At this level, we only gain very little computation power from our GPU with around **~0.17 ms** compared to torch/cuBLAS's **~0.008 ms**, which is around 20x slower.

## More Concurrency

To further increase the concurrency of our kernel, we can exploit finer thread-level parallelism. Instead of assigning each thread to compute a single output element in C, you can introduce parallelism along the K dimension. Each thread computes a partial accumulation, and you then combine these partial results. This approach requires primitives like `atomicAdd` in CUDA.

Here’s a simplified version:
```python
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((1,), dtype)
            B_local = T.alloc_local((1,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                A_local[0] = A[bk * BLOCK_K + tk]
                B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

By introducing parallelism along K dimension, our kernel now achieves **~0.024 ms**, an improvement, but still not on par with torch/cuBLAS.

### Customizing Parallelism in K Dimension
If your K dimension is large, you can further customize how many elements each thread processes by introducing a `reduce_threads` parameter. This way, each thread handles multiple elements per iteration:

```python
def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K = T.ceildiv(BLOCK_K, reduce_threads)

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

## Vectorized Reads

GEMV is less computation intensive than GEMM as the computation intensity and memory throughput will be the optimization bottleneck. One effective strategy is to use vectorized load/store operations (e.g., `float2`, `float4`). In `TileLang`, you can specify vectorized operations via `T.vectorized`:

```python
def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

With vectorized read, now the kernel finishes in **~0.0084 ms**, which is getting close to cuBLAS performance.

## `tvm_thread_allreduce` Instead of `atomicAdd`

[`tvm_thread_allreduce`](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.tvm_thread_allreduce) has implemented optimization when making an all-reduce across a number of threads, which should outperfrom out plain smem + `atomidAdd`:

```python
def splitk_gemv_vectorized_tvm(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main
```

With this optimization, the kernel latency now reduces from **~0.0084 ms** to **~0.0069 ms**, which is faster than torch/cuBLAS!

## Autotune

`BLOCK_N`, `BLOCK_K`, `reduce_threads` are hyperparameters in our kernel, which can be tuned to improve performance. We can use the `tilelang.autotune` feature to automatically search for optimal configurations:

```python
def get_best_config(N, K):

    def get_configs():
        BLOCK_N = [2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        _configs = list(itertools.product(
            BLOCK_N,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_N": c[0],
            "reduce_threads": c[1],
        } for c in _configs]
        return configs

    @autotune(
        configs=get_configs(),
        warmup=3,
        rep=20,
    )
    @jit(
        out_idx=[-1],
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        target="auto",
    )
    def kernel(
        BLOCK_N=None,
        reduce_threads=None,
    ):
        dtype = "float16"
        accum_dtype = "float"
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_K = reduce_threads * TILE_K

        @T.prim_func
        def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                A_local = T.alloc_local((TILE_K,), dtype)
                B_local = T.alloc_local((TILE_K,), dtype)
                C_accum = T.alloc_local((1,), accum_dtype)

                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                C_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_accum[0],
                            True,
                            C_reduced[0],
                            tk,
                            dtype="handle",
                        ))

                C[bn * BLOCK_N + tn] = C_reduced[0]

        return main

    return kernel()
```

After autotuning, now our kernel gets **~0.0067 ms**, the final generated CUDA kernel might like this:

```C++
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  float C_accum[1];
  half_t A_local[8];
  half_t B_local[8];
  __shared__ float red_buf0[64];
  C_accum[0] = 0.000000e+00f;
  for (int bk = 0; bk < 4; ++bk) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((bk * 256) + (((int)threadIdx.y) * 8)));
    *(uint4*)(B_local + 0) = *(uint4*)(B + ((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 1024)) + (bk * 256)) + (((int)threadIdx.y) * 8)));
    for (int k = 0; k < 8; ++k) {
      C_accum[0] = (C_accum[0] + (((float)A_local[k]) * ((float)B_local[k])));
    }
  }
  tl::fence_proxy_async();
  __syncthreads();
  red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = C_accum[0];
  __syncthreads();
  if (((int)threadIdx.y) < 16) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 16)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 8) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 8)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 4) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 4)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 2) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 2)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 1) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 1)]);
  }
  __syncthreads();
  C[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = ((half_t)red_buf0[(((int)threadIdx.x) * 32)]);
}
```

This corresponds closely to our `TileLang` program, with necessary synchronization and low-level optimizations inserted automatically.

## Conclusion

### Benchmark Table on Hopper GPU

| Kernel Name   | Latency   |
|------------|------------|
| torch/cuBLAS | 0.00784 ms |
| Triton | 0.00773 ms |
| naive_gemv | 0.16607 ms |
| splitk_gemv | 0.02419 ms |
| splitk_gemv_vectorized | 0.00809 ms |
| splitk_gemv_vectorized_tvm | 0.00675 ms |

Triton Time: 0.0077344514429569244
In this tutorial, we implemented a simple GEMV kernel and learn that `TileLang` exposes low level control to user such as thread-level programming and CUDA primitives.
