General Matrix-Vector Multiplication (GEMV)
===========================================

<div style="text-align: left;">
    <em>Contributor: </em> <a href="https://github.com/botbw">@botbw</a>
</div>

:::{warning}
   This document is still **experimental** and may be incomplete.  
   Suggestions and improvements are highly encouraged—please submit a PR!
:::

General matrix-vector multipliction (GEMV) as a specialized general matrix-matrix multiplication (GEMM) has its unique position in deep learning, especially for large language model inference decoding. In this tutorial, we are gonna optimize GEMV from thread prespective step by step in `TileLang`.

# Triton implementation
To implement an gemv kernel, you might use convenient tools like `Triton` and write code like this:

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

`Triton` is easy to develop as it operates at block level. However, this could lack thread-level control and miss futher optimization possibilities In this tutorial, we are going write an optimized gemv kernel with `TileLang` and exploit its fine-grained controlbility.

# Naive Implementation in TileLang
Suppose that you know some basic CUDA C programming. Immediately, we can implement a naive gemv kernel following GEMM's tiling strategy like this (seeing GEMV as a `(1, k) * (k, n)` GEMM):

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
        A: T.Buffer((K, ), dtype),
        B: T.Buffer((N, K), dtype),
        C: T.Buffer((N, ), dtype),
        
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            tn = T.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_K, ), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1, ), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(
                        accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]
    return main
```

And your kernel will be compiled into CUDA by tilelang like this (can be found at `~/.tilelang/cache`):

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

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(8, 1, 1), dim3(256, 1, 1), 33024, stream>>>(A, B, C);

return 0;
}
```

Where you see tilelang helps to generate a warp-specialized kernel with the first 128 threads as producer and last 128 threads as consumer in a block (when you are using 1D block). 

At this level, we only gain very little computation power from our GPU with around **~0.17 ms** compare with torch/cuBLAS's **~0.008 ms**, which is amost 20x slower.
# More concurrency

To further increase the concurrency of our kernel, we can utilize the thread level control of `TileLang`: Instead of making each thread calculate each output element of C, we now add parallelism at K-dimension and let each thread to calculate a partial (or pre-accumulated) result, and reduce the partial value at the end to calculate the correct one. To reduce the partial value, we will have to use some low-level primitives like `atomicAdd` in CUDA.

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

With the new implementation, our kernel now achieves **~0.024 ms**, which is till a bit far away from torch/cuBLAS.

## make k dimention custom
Sometimes, the K dimension could be quite large and we can make each thread to calculate multiple partial results by introducing `reduce_threads`, which allows us to control the concurrency along K dimension:

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
    """
    Naive GEMV following GEMM tiling strategy in SIMD manner.
    """
    @T.prim_func
    def main(
        A: T.Buffer((K, ), dtype),
        B: T.Buffer((N, K), dtype),
        C: T.Buffer((N, ), dtype),
        
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            TILE_K = T.ceildiv(BLOCK_K, reduce_threads)
            A_shared = T.alloc_shared((BLOCK_K, ), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_shared = T.alloc_shared((BLOCK_N, ), accum_dtype)
            
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_shared[tk * TILE_K + k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_shared[tn, tk * TILE_K + k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                C_reg = T.alloc_local((1,), accum_dtype)
                T.clear(C_reg)
                for k in T.serial(TILE_K):
                    C_reg[0] += A_shared[tk * TILE_K + k].astype(accum_dtype) * B_shared[tn, tk * TILE_K + k].astype(
                        accum_dtype)
                T.atomic_add(C_shared[tn], C_reg[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]
    return main
```


# Vectorilized Read

GEMV is less computation intensive than GEMM as the computation intensity and memory throuput will be our optimization bottleneck. One method to gain more throughput is to use vectorized read/write operation, instead of letting each k-reduce thread reading one element, we can make it read a "chunk" of elements, which can be achieved using `T.vectorize` primitive, which warps around `float2`, `float4`... data structures in CUDA C.

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
                    B_local[k] = B[bn * BLOCK_N + tn,
                                                    bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

With vectorized read, now the kernel finishs in **~0.084 ms**, we are almost there!


# Autotune

`BLOCK_N`, `BLOCK_K`, `reduce_threads` are hyperparameters in our kernel, which can be tuned to improve performance. We can utilize the `autotune` in `tilelang.autotune` to get the best perfomance under this given input:

```python
def get_best_config(N, K):
    def get_configs():
        BLOCK_N = [ 2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        _configs = list(
            itertools.product(
                BLOCK_N,
                reduce_threads,
            ))
        configs = [
            {
                "BLOCK_N": c[0],
                "reduce_threads": c[1],
            } for c in _configs
        ]
        return configs

    @autotune(
        configs=get_configs(),
        keys=[
            "BLOCK_N",
            "reduce_threads",
        ],
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
                A_shared = T.alloc_local((TILE_K,), dtype)
                B_shared = T.alloc_local((TILE_K,), dtype)
                C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
                C_reg = T.alloc_local((1,), accum_dtype)
                if tk == 0:
                    C_shared[tn] = 0
                T.clear(C_reg)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_shared[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                        B_shared[k] = B[bn * BLOCK_N + tn,
                                                        bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_reg[0] += A_shared[k].astype(accum_dtype) * B_shared[k].astype(accum_dtype)
                T.atomic_add(C_shared[tn], C_reg[0])
                C[bn * BLOCK_N + tn] = C_shared[tn]

        return main

    return kernel()
```

After autotuning, now our kernel gets **~0.008 ms**, which is comparable to torch/cuBLAS (although still a bit slower in my case).

Our final generated CUDA kernel will be like this:

```C++
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) float C_shared[];
  float C_accum[1];
  half_t A_local[8];
  half_t B_local[8];
  if (((int)threadIdx.y) == 0) {
    C_shared[((int)threadIdx.x)] = 0.000000e+00f;
  }
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
  AtomicAdd((&(C_shared[((int)threadIdx.x)])), C_accum[0]);
  C[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = ((half_t)C_shared[((int)threadIdx.x)]);
}
```

Which is exactly what we wrote in `TileLang` with necessary synchonization inserted!

# Conclusion

In this tutorial, we implemented a simple GEMV kernel and learn that `TileLang` exposes low level control to user such as thread-level programming and CUDA primitives.