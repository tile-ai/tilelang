# Instructions

This page summarizes the core TileLang “instructions” available at the DSL
level, how they map to hardware concepts, and how to use them correctly.

## Quick Categories
- Data movement: `T.copy`, `T.async_copy`, `T.tma_copy`, `T.c2d_im2col`, staging Global ↔ Shared ↔ Fragment
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

### Lowering `T.copy` to variants of copy mechanisms

TileLang supports both synchronous and explicitly-asynchronous copies.

`T.copy(src, dst, ...)` (synchronous semantics)
- Intended default for most TileLang programs.
- The compiler is free to lower it to different mechanisms (synchronous SIMT copy `ld.global`, warp-level copy     `ldmatrix`, async copy via TMA `cp.async.bulk`, old async copy `cp.async`, etc.) depending on target/hints, but the   observable semantics
  are *synchronous*: after the statement, it is safe to use `dst`.
- If `T.copy` lowers to `cp.async`, TileLang will still preserve synchronous
  semantics by emitting the required `commit`/`wait` (and any required
  synchronization) so that consuming `dst` is correct.

`T.async_copy(src, dst, ...)` (explicit async semantics)
- Intended for writing manual pipelines or warp-specialized code where you want
  to overlap global->shared copies with compute.
- Lowers through `cp.async` and emits:
  - `ptx_cp_async(...)`
  - `ptx_commit_group()`
  - No `ptx_wait_group(...)` is auto-inserted.
- You must explicitly insert `T.ptx_wait_group(...)` before consuming `dst`.
- A barrier is still required when `dst` is produced cooperatively and consumed
  across threads. In most TileLang programs you do not need to write it
  manually: `ThreadSync("shared")` will insert the required
  `T.tvm_storage_sync("shared")` before the first read from `dst`. If you want
  explicit control (or if you're writing very low-level code), you can insert
  `T.tvm_storage_sync("shared")` yourself (or `T.tvm_storage_sync("warp")` for
  warp-local consumption).
- This op is intentionally strict: if the copy cannot be lowered to `cp.async`
  (e.g., wrong scopes, unsupported vector width), compilation fails instead of
  silently falling back to a synchronous copy.

Example (manual async prefetch)
```python
# Prefetch into shared asynchronously (emits cp.async + commit).
T.async_copy(A[by * BM, ko * BK], A_s)

# ... independent work here ...

# Before consuming A_s, ensure the async copies are completed.
T.ptx_wait_group(0)
# The required shared-memory barrier will be inserted automatically before the
# first read from A_s by ThreadSync("shared") in the default lowering pipeline.
T.gemm(A_s, B_s, C_f)
```

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
- `T.async_copy(src, dst, ...)`: Explicit async global→shared copy via `cp.async`.
- `T.tma_copy(src, dst, ...)`: Explicit async global→shared copy via `cp.async.bulk`
- `T.transpose(src, dst)`: Transpose a 2D shared buffer: `dst[j, i] = src[i, j]`.
- `T.c2d_im2col(img, col, ...)`: 2D im2col transform for conv.

Memory allocation and descriptors
- `T.alloc_shared(shape, dtype, scope='shared.dyn')`: Allocate shared buffer.
- `T.alloc_fragment(shape, dtype, scope='local.fragment')`: Allocate fragment.
- `T.alloc_var(dtype, [init], scope='local.var')`: Scalar var buffer (1 elem).
- `T.alloc_barrier(arrive_count)`: Allocate and initialize one or more mbarriers.
- `T.alloc_tmem(shape, dtype)`: Tensor memory (TMEM) buffer (Blackwell+).
- `T.deallocate_tmem(buffer)`: Explicitly release a TMEM buffer at the current site.
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
- `T.sync_threads([barrier_id, arrive_count])`: Block-wide barrier (`__syncthreads()`).
- `T.sync_warp([mask])`: Warp-wide barrier (`__syncwarp([mask])`).
- `T.sync_grid()`: Cooperative grid barrier (requires cooperative launch).
- `T.pdl_trigger()`: Signal programmatic launch completion for the current kernel.
- `T.pdl_sync()`: Wait until kernel dependencies are satisfied.

Warp-vote / warp-ballot (CUDA ≥ 9 / HIP)
- `T.any_sync(predicate[, mask])` → `int32`: Non-zero if ANY lane in `mask` has non-zero predicate (`__any_sync`). `mask` defaults to `0xFFFFFFFF`.
- `T.all_sync(predicate[, mask])` → `int32`: Non-zero if ALL lanes in `mask` have non-zero predicate (`__all_sync`). `mask` defaults to `0xFFFFFFFF`.
- `T.ballot_sync(predicate[, mask])` → `uint64`: Bitmask of lanes in `mask` with non-zero predicate. CUDA: `__ballot_sync` zero-extended to 64 bits; HIP: `__ballot` returns natively as `uint64`, covering all 64 wavefront lanes. `mask` defaults to `0xFFFFFFFF`.
- `T.ballot(predicate)` → `uint64`: Full-warp/wavefront ballot (mask = `0xFFFFFFFF`). No truncation on HIP.
- `T.activemask()` → `uint64`: Bitmask of currently active lanes. CUDA: `__activemask` zero-extended to 64 bits; HIP: `__ballot(1)` as `uint64`.

Block-wide predicated sync
- `T.syncthreads_count(predicate)` → `int32`: Sync all threads; return count with non-zero predicate (`__syncthreads_count`).
- `T.syncthreads_and(predicate)` → `int32`: Sync; non-zero iff ALL threads have non-zero predicate (`__syncthreads_and`).
- `T.syncthreads_or(predicate)` → `int32`: Sync; non-zero iff ANY thread has non-zero predicate (`__syncthreads_or`).

Warp-shuffle (intra-warp data exchange). All accept a trailing `mask` kwarg that defaults to `0xFFFFFFFF`.
- `T.shfl_sync(value, src_lane[, width, mask])`: Broadcast value from `src_lane` to all lanes (`__shfl_sync`).
- `T.shfl_xor(value, delta[, width, mask])`: XOR-swap across lanes (`__shfl_xor_sync`).
- `T.shfl_down(value, delta[, width, mask])`: Shift down by `delta` lanes (`__shfl_down_sync`).
- `T.shfl_up(value, delta[, width, mask])`: Shift up by `delta` lanes (`__shfl_up_sync`).

Warp-match (CUDA sm_70+, not supported on HIP). `mask` defaults to `0xFFFFFFFF`.
- `T.match_any_sync(value[, mask])` → `uint32`: Bitmask of lanes in `mask` whose `value` matches the calling lane's (`__match_any_sync`).
- `T.match_all_sync(value[, mask])` → `uint32`: Returns `mask` if all lanes in `mask` agree on `value`, else 0 (`__match_all_sync`). The C-level `int*` predicate output is hidden; reconstruct it as `result != 0`.

> **Note on HIP:** `any_sync`/`all_sync` ignore the mask and call `__any`/`__all` directly. `ballot_sync`, `ballot`, and `activemask` call `__ballot` which returns `uint64` natively on 64-thread wavefronts — no truncation occurs. Shuffle intrinsics lower to `__shfl`/`__shfl_xor`/`__shfl_down`/`__shfl_up` (mask ignored). `syncthreads_count/and/or` have identical signatures on both platforms. `match_any_sync` and `match_all_sync` have no HIP equivalent and will fail to codegen on HIP.

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
