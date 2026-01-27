# Instructions

This page summarizes the core TileLang “instructions” available at the DSL
level, how they map to hardware concepts, and how to use them correctly.

## Quick Categories
- Data movement: `T.copy`, `T.c2d_im2col`, staging Global ↔ Shared ↔ Fragment
- Compute primitives: `T.gemm`/`T.gemm_sp`, elementwise math (`T.exp`, `T.max`),
  reductions (`T.reduce_sum`, `T.cumsum`, warp reducers)
- Control helpers: `T.clear`/`T.fill`, `T.reshape`/`T.view`
- Diagnostics: `T.print`, `T.device_assert`
- Advanced: atomics, memory barriers, warp‑group ops

## Data Movement

Use `T.copy(src, dst, *, coalesced_width=None, disable_tma=False, eviction_policy=None, loop_layout=None)`
to move tiles between memory scopes. It accepts `tir.Buffer`, `BufferLoad`, or
`BufferRegion`; extents are inferred or broadcast when possible.

```python
# Global → Shared tiles (extents inferred from dst)
T.copy(A[by * BM, ko * BK], A_s)
T.copy(B[ko * BK, bx * BN], B_s)

# Fragment/Register → Global (store result)
T.copy(C_f, C[by * BM, bx * BN])
```

Semantics
- Extents are deduced from arguments; missing sides broadcast to the other’s rank.
- Access patterns are legalized and coalesced during lowering. Explicit
  vectorization is not required in HL mode.
- Safety: the LegalizeSafeMemoryAccess pass inserts boundary guards when an
  access may be out‑of‑bounds and drops them when proven safe.

Other helpers
- `T.c2d_im2col(img, col, ...)`: convenience for conv‑style transforms.

## Compute Primitives

GEMM and sparse GEMM
- `T.gemm(A_shared, B_shared, C_fragment)`: computes a tile GEMM using shared
  inputs and a fragment accumulator; lowered to target‑specific tensor cores.
- `T.gemm_sp(...)`: 2:4 sparse tensor core variant (see examples and README).

Reductions and scans
- `T.reduce_sum`, `T.reduce_max`, `T.reduce_min`, `T.cumsum`, plus warp
  reducers (`T.warp_reduce_sum`, etc.).
- Allocate and initialize accumulators via `T.alloc_fragment` + `T.clear` or
  `T.fill`.

Elementwise math
- Most math ops mirror TVM TIR: `T.exp`, `T.log`, `T.max`, `T.min`, `T.rsqrt`,
  `T.sigmoid`, etc. Compose freely inside loops.

Reshape/view (no copy)
- `T.reshape(buf, new_shape)` and `T.view(buf, shape=None, dtype=None)` create
  new views that share storage, with shape/dtype checks enforced.

## Synchronization (HL usage)

In HL pipelines, you usually don’t need to write explicit barriers. Passes such
as PipelinePlanning/InjectSoftwarePipeline/InjectTmaBarrier orchestrate
producer/consumer ordering and thread synchronization behind the scenes.

If you need debugging or explicit checks:
- `T.device_assert(cond, msg='')` emits device‑side asserts on CUDA targets.
- `T.print(obj, msg='...')` prints scalars or buffers safely from one thread.

## Putting It Together: GEMM Tile

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), 'float16'),
    B: T.Tensor((K, N), 'float16'),
    C: T.Tensor((M, N), 'float16'),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), 'float16')
        B_s = T.alloc_shared((BK, BN), 'float16')
        C_f = T.alloc_fragment((BM, BN), 'float32')
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)  # Global → Shared
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)             # compute into fragment

        T.copy(C_f, C[by * BM, bx * BN])      # store back
```

## Instruction Reference (Concise)

Below is a concise list of TileLang instructions grouped by category. For full
signatures, behaviors, constraints, and examples, refer to API Reference
(`autoapi/tilelang/index`).

Data movement
- `T.copy(src, dst, ...)`: Move tiles between Global/Shared/Fragment.
- `T.c2d_im2col(img, col, ...)`: 2D im2col transform for conv.

Memory allocation and descriptors
- `T.alloc_shared(shape, dtype, scope='shared.dyn')`: Allocate shared buffer.
- `T.alloc_fragment(shape, dtype, scope='local.fragment')`: Allocate fragment.
- `T.alloc_var(dtype, [init], scope='local.var')`: Scalar var buffer (1 elem).
- `T.alloc_barrier(arrive_count)`: Allocate and initialize one or more mbarriers.
- `T.alloc_tmem(shape, dtype)`: Tensor memory (TMEM) buffer (Hopper+).
- `T.alloc_reducer(shape, dtype, op='sum', replication=None)`: Reducer buf.
- `T.alloc_descriptor(kind, dtype)`: Generic descriptor allocator.
  - `T.alloc_wgmma_desc(dtype='uint64')`
  - `T.alloc_tcgen05_smem_desc(dtype='uint64')`
  - `T.alloc_tcgen05_instr_desc(dtype='uint32')`
- `T.empty(shape, dtype='float32')`: Declare function output tensors.

Compute primitives
- `T.gemm(A_s, B_s, C_f)`: Tile GEMM into fragment accumulator.
- `T.gemm_sp(...)`: Sparse (2:4) tensor core GEMM.
- Reductions: `T.reduce_sum/max/min/abssum/absmax`, bitwise `and/or/xor`.
- Scans: `T.cumsum`, finalize: `T.finalize_reducer`.
- Warp reducers: `T.warp_reduce_sum/max/min/bitand/bitor`.
- Elementwise math: TIR ops (`T.exp`, `T.log`, `T.max`, `T.min`, `T.rsqrt`, ...).
- Fast math: `T.__log/__log2/__log10/__exp/__exp2/__exp10/__sin/__cos/__tan`.
- IEEE math: `T.ieee_add/sub/mul/fmaf` (configurable rounding).
- Helpers: `T.clear(buf)`, `T.fill(buf, value)`.
- Views: `T.reshape(buf, shape)`, `T.view(buf, shape=None, dtype=None)`.

Diagnostics
- `T.print(obj, msg='')`: Print scalar/buffer from one thread.
- `T.device_assert(cond, msg='')`: Device-side assert (CUDA).

Logical helpers
- `T.any_of(a, b, ...)`, `T.all_of(a, b, ...)`: Multi-term predicates.

Annotation helpers
- `T.use_swizzle(panel_size=..., enable=True)`: Rasterization hint.
- `T.annotate_layout({...})`: Attach explicit layouts to buffers.
- `T.annotate_safe_value(var, ...)`: Safety/const hints.
- `T.annotate_l2_hit_ratio(buf, ratio)`: Cache behavior hint.

Synchronization helpers
- `T.pdl_trigger()`: Signal programmatic launch completion for the current kernel.
- `T.pdl_sync()`: Wait until kernel dependencies are satisfied.

Atomics
- `T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False)`.
- `T.atomic_addx2(dst, value, return_prev=False)`; `T.atomic_addx4(...)`.
- `T.atomic_max(dst, value, memory_order=None, return_prev=False)`.
- `T.atomic_min(dst, value, memory_order=None, return_prev=False)`.
- `T.atomic_load(dst)`, `T.atomic_store(dst, value)`.

Custom intrinsics
- `T.dp4a(A, B, C)`: 4‑element dot‑product accumulate.
- `T.clamp(x, lo, hi)`: Clamp to [lo, hi].
- `T.loop_break()`: Break from current loop via intrinsic.

Barriers, TMA, warp‑group
- Barriers: `T.alloc_barrier(arrive_count)`.
- Parity ops: `T.mbarrier_wait_parity(barrier, parity)`, `T.mbarrier_arrive(barrier)`.
- Expect tx: `T.mbarrier_expect_tx(...)`; sugar: `T.barrier_wait(id, parity=None)`.
- TMA: `T.create_tma_descriptor(...)`, `T.tma_load(...)`,
  `T.tma_store_arrive(...)`, `T.tma_store_wait(...)`.
- Proxy/fences: `T.fence_proxy_async(...)`, `T.warpgroup_fence_operand(...)`.
- Warp‑group: `T.warpgroup_arrive()`, `T.warpgroup_commit_batch()`,
  `T.warpgroup_wait(num_mma)`, `T.wait_wgmma(id)`.

Lane/warp index
- `T.get_lane_idx(warp_size=None)`: Lane id in warp.
- `T.get_warp_idx_sync(warp_size=None)`: Canonical warp id (sync).
- `T.get_warp_idx(warp_size=None)`: Canonical warp id (no sync).
- `T.get_warp_group_idx(warp_size=None, warps_per_group=None)`: Group id.

Register control
- `T.set_max_nreg(reg_count, is_inc)`, `T.inc_max_nreg(n)`, `T.dec_max_nreg(n)`.
- `T.annotate_producer_reg_dealloc(n=24)`, `T.annotate_consumer_reg_alloc(n=240)`.
- `T.no_set_max_nreg()`, `T.disable_warp_group_reg_alloc()`.

## Notes on Dtypes

Dtypes accept three equivalent forms:
- String: `'float32'`
- TileLang dtype: `T.float32`
- Framework dtype: `torch.float32`
All are normalized internally. See Type System for details.
