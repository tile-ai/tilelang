# Language Basics

This page introduces the core TileLang (tile‑lang) DSL that you’ll use to write
high‑performance kernels. It focuses on how to define a kernel, express
iteration, move data across memory scopes, and run it with JIT.

The examples use the conventional aliases:

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## 1. Defining a Kernel with `@T.prim_func`

TileLang kernels are TIR (TVM IR) functions produced by the `@T.prim_func`
decorator. Arguments are annotated with shapes and dtypes via `T.Tensor` or
`T.Buffer`.

Note on dtypes
- You can pass dtypes as a string (e.g., 'float32'), a TileLang dtype (e.g., `T.float32`),
  or a framework dtype (e.g., `torch.float32`). TileLang normalizes all of these.
  See Type System for details.

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((N,), dtype),  # dtype could be 'float32' | T.float32 | torch.float32
    B: T.Tensor((N,), dtype),
    C: T.Tensor((N,), dtype),
): ...  # kernel body
```

- Shapes may be concrete integers or symbolic. For symbolic, you can pass
  Python ints through the outer `@jit` wrapper (shown below), or annotate with
  `T.dyn` when you want a named symbolic dimension.

```python
# Named symbolic dimension (optional)
K = T.dyn["K"]


@T.prim_func
def uses_dyn(A: T.Tensor((K,), "float32")): ...
```

### Dynamic symbolic dimensions: two ways

TileLang supports two complementary ways to introduce symbolic (dynamic) dims:

- Type-level annotations via `T.dyn[...]` (recommended for function signatures)
  - Use in `T.Tensor((T.dyn['K'], ...), dtype)` or bind once then reuse (as above).
  - Inside the kernel body, prefer reading from the buffer’s shape, e.g. `M = A.shape[0]`.

- Term-level variables via `T.dynamic(name, dtype)`
  - Creates a TIR `tir.Var` you can use directly in expressions/loops.
  - Handy when you need to reference the dimension symbol in the body.

```python
# 1) Annotation-only symbol; read the bound size via shape
K = T.dyn["K"]  # dtype defaults to int32


@T.prim_func
def foo(A: T.Tensor((K,), "float32")):
    N = A.shape[0]
    for i in T.serial(N):
        ...


# 2) Explicit Var symbol usable in the body
K = T.dynamic("K", "int32")  # or T.dynamic('K') defaults to int32


@T.prim_func
def bar(A: T.Tensor((K,), "float32")):
    for i in T.serial(K):
        ...
```

Notes
- Under `@jit`, concrete sizes come from the actual tensor arguments at the first call.
- Symbols in annotations do not need to be separate kernel arguments; TileLang binds them from argument shapes.

## 2. Launching Work with `T.Kernel`

`with T.Kernel(...)` declares a grid of tile programs. The positional
arguments give the grid extent along each axis and the returned variables are
the program indices along those axes. This is the part of a launch every
target shares: on CUDA a program is a thread block and `bx`/`by` are
`blockIdx.x`/`blockIdx.y`; on CPU the grid becomes the outer loop.

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...  # bx/by are the program indices (blockIdx.x/y on CUDA)
```

Keyword arguments are launch annotations that the backend interprets once the
target is known. Each language dialect's `Kernel` declares the annotations its
backend understands as explicit keyword parameters, so hovering or
autocompleting `T.Kernel` shows exactly those and anything else is rejected:
`tilelang.language` (the CUDA dialect) offers `threads`, `prelude` and
`cluster_dims`; `tilelang.rocm.language` / `tilelang.metal.language` offer
`threads` and `prelude`; `tilelang.cpu.language` offers only `prelude`.
`threads` is the SIMT one: how many threads run each tile program on
GPU-style backends. Those backends pick a default (128) when it is omitted;
a kernel written with the CUDA dialect still compiles for CPU, which ignores
the thread count. Code inside `T.Kernel` operates at the tile-program level,
so you rarely need raw thread indices; most kernels use structured loops
(`T.serial`, `T.unroll`, `T.Parallel`, `T.Pipelined`) that the compiler maps
onto threads. `T.get_thread_binding()` exposes the thread index for
thread-level code on SIMT targets; a kernel that uses it is rejected when
compiled for a target without SIMT threads.

`T.ClusterKernel(..., cluster_dims=...)` adds the CUDA thread-block-cluster
annotation (SM90+). A cluster is a `cluster_dims`-shaped tile of the grid, so
`T.get_cluster_id(axis)` is plain program-index arithmetic
(`bx // cluster_dims[axis]`) and works on every target, while
`T.block_rank_in_cluster()` reads the hardware rank and is CUDA-only. Targets
without clusters reject `cluster_dims` at compile time.

## 3. Loops and Control Flow

Core loop constructs map to familiar hardware patterns:

- `T.serial(start, stop[, step])`: plain for‑loop
- `T.unroll(start, stop[, step])`: unrolled loop
- `T.Parallel(ext0, ext1, ...)`: nested parallel loops (elementwise‑friendly)
- `T.Pipelined(iters, num_stages=N)`: software pipelining for producer/consumer

```python
for i in T.serial(N):
    ...

for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    # overlap copy/compute across stages
    ...
```

Conditionals use standard Python `if`/`else`. Guard edges with predicates when
tile sizes do not divide problem sizes evenly.

## 4. Memory Scopes and Allocation

TileLang exposes key software‑managed scopes:

- Global: device memory (default for `T.Tensor` arguments)
- Shared: on‑chip, block‑visible (`T.alloc_shared(shape, dtype)`)
- Fragment and scalars: per‑thread fragments and scalar vars but in Shared View
  (`T.alloc_fragment`, `T.alloc_var`)

```python
A_shared = T.alloc_shared((BM, BK), "float16")
B_shared = T.alloc_shared((BK, BN), "float16")
C_local = T.alloc_fragment((BM, BN), "float32")
T.clear(C_local)  # zero accumulators
```

## 5. Moving Data: `T.copy`

Use `T.copy(src, dst)` to move tiles between scopes. It accepts buffers,
buffer regions, or buffer loads; extents are inferred or can be broadcast.

```python
# Global -> Shared (tile copy), extents inferred from dst
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[ko * BK, bx * BN], B_shared)

# Fragment -> Global (store back)
T.copy(C_local, C[by * BM, bx * BN])
```

`T.copy` performs coalescing and scope‑specific lowering during compilation.

If you need explicitly asynchronous global->shared prefetch for manual pipelining,
use `T.async_copy(src, dst)`. Unlike `T.copy`, it does not auto-insert any wait:
you must explicitly insert `T.ptx_wait_group(...)` before consuming `dst`. A
shared-memory barrier is still required for cross-thread consumption, but in
most TileLang programs you do not need to write it manually because
`ThreadSync("shared")` will insert the necessary `T.tvm_storage_sync("shared")`
before the first read from `dst`.

## 6. A Minimal End‑to‑End Example (Vector Add)

```python
import tilelang
import tilelang.language as T
from tilelang import jit


@jit  # infers target from tensors at first call
def add(N: int, block: int = 256, dtype: str = "float32"):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                # Optional — LegalizeSafeMemoryAccess inserts a guard when an access may be OOB
                C[gi] = A[gi] + B[gi]

    return add_kernel


# Host side (PyTorch shown; NumPy/DLPack also supported)
import torch

N = 1 << 20
A = torch.randn(N, device="cuda", dtype=torch.float32)
B = torch.randn(N, device="cuda", dtype=torch.float32)
C = torch.empty(N, device="cuda", dtype=torch.float32)

kernel = add(N)
kernel(A, B, C)  # runs on GPU
torch.testing.assert_close(C, A + B)
```

Notes
- The `@jit` wrapper returns a callable kernel after the first compilation.
- You can pass compile‑time tunables (tile sizes, dtypes) through the outer
  Python function and bake them into the generated TIR.

## 7. Tiled GEMM Skeleton

Below is a minimal pattern for a tiled GEMM using shared memory staging and a
fragment accumulator. It mirrors the quickstart style found in the repository.

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), "float16")
        B_s = T.alloc_shared((BK, BN), "float16")
        C_f = T.alloc_fragment((BM, BN), "float32")
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)  # lowered to tensor‑core/ISA specific kernels

        T.copy(C_f, C[by * BM, bx * BN])
```

## 8. Debugging and Printing

Use `T.print` inside a kernel for quick introspection. TileLang emits printing
from a single thread for shared/fragment scopes to avoid floods.

```python
T.print(C_f, msg="accumulator:")
T.print(A_s, msg="A tile:")
T.print(C[0], msg="C[0] = ")
```

## 9. Where to Go Next

- Control flow details: see Programming Guides → Control Flow
- Memory topics: see Programming Guides → (removed cache/layout); basics are covered inline
- Autotuning tile sizes and mappings: Programming Guides → Autotuning
- Operator examples (GEMM, GEMV, attention): see Deep Learning Operators
