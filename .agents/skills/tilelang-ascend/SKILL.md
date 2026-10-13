---
name: tilelang-ascend
description: >
  Write, port, tune, and debug TileLang kernels for Huawei Ascend NPUs. Covers the
  Ascend dialect, AIC/AIV work partitioning, SIMD/SIMT operations, memory and
  data movement, T.Stage scheduling, synchronization, performance measurement,
  profiling, operation cycle measurements, and VF latency annotations.
  Explains how Ascend kernel programming differs from CUDA/GPU programming.
---

# TileLang Ascend NPU Programming

Use this guide to write, port, tune, and debug kernels for Ascend NPUs.
Import the Ascend dialect:

```python
import tilelang.ascend.language as T
```

The explicit dialect exposes Ascend-specific APIs such as `T.MixedKernel`
and `T.SimdVF`. For installation, follow the repository's current build
documentation and [tilelang-build](../tilelang-build/SKILL.md).

## Task-specific references

- [Scheduling and synchronization](references/synchronization.md): automatic,
  T.Stage performance tuning, buffer ownership and version modes, conditional
  execution, conflict hints, and fully manual protocols.

- [Profiling recipes](references/profiling.md): device timing, regression tools,
  generated-source experiments, and operation cycle measurements with npusim.

For a suspected compiler defect, preserve the smallest failing kernel together
with its generated source or lowering trace.

---

## 1. Mental Model: Ascend vs CUDA

| Concept | CUDA | Ascend NPU |
|---|---|---|
| Kernel launch | `T.Kernel(grid, threads=128)` | `T.Kernel(N)` -- **no threads arg** |
| Launch grid | 1-D, 2-D, or 3-D | 1-D grids only |
| Thread access | `T.get_thread_binding()` | No `threadIdx` directly; use `T.SimtVF(threads=N)` |
| Compute cores | SM (unified) | **Two distinct cores**: AIC (Cube) + AIV (Vector) |
| Matrix multiply | `T.gemm(...)` | Current L1-input path uses `transpose_B=True`; check the selected lowering |
| Shared memory | `T.alloc_shared(...)` ("shared") | `T.alloc_shared(...)` maps to UB (Unified Buffer) |
| Global memory | Global | GM (Global Memory / HBM) |
| L1 cache / data | N/A | `T.alloc_l1(...)` -- Cube Buffer (CBuf) |
| Register fragments | `T.alloc_fragment(...)` | `T.alloc_fragment(...)` |
| Cube engine buffers | N/A | `T.alloc_l0a/l0b/l0c(...)` -- L0A/L0B/L0C |
| Warp-level ops | `__shfl_sync` | `T.alloc_reducer` / `T.warp_reduce_sum` |
| Synchronization | `__syncthreads()`, `mbarrier`, `cp.async` | `asc_syncthreads()` (auto-inserted in SimtVF), Flag-based: `T.ascend_set_flag` / `T.ascend_wait_flag` |
| Auto-Schedule | `T.Pipelined(...)` | `T.Pipelined(...)` with `num_stages=...` |
| Compiler | nvcc | Bisheng (Huawei Ascend compiler) |

---

## 2. Thread Hierarchy

### 2.1 Kernel Launch

Ascend supports **only 1-D grids** and does **not** accept a `threads=`
parameter on `T.Kernel`. Thread domains are defined separately.

```python
# CORRECT - Ascend style:
with T.Kernel(NUM_CORES) as bx:
    ...

# WRONG - will raise TypeError:
with T.Kernel(NUM_CORES, threads=128) as bx:
    ...
```

### 2.2 SIMT VF: Thread-Parallel Compute

Use `T.SimtVF(threads=N)` to define a thread-parallel region. `threads` also
accepts a tuple/list of up to three dimensions, for example `(32, 4)`; omitted
dimensions default to one. Inside this block, `T.Parallel` loops distribute
work across the declared thread domain.

```python
with T.SimtVF(threads=128):
    for i in T.Parallel(N):
        out[i] = a[i] + b[i]
```

Auto-sync: The compiler inserts `asc_syncthreads()` when it detects
cross-thread data hazards; regression coverage is in
`testing/ascend/transform/test_ascend_thread_sync.py`. Thread binding is
accessed via `T.get_thread_binding()`. For random values inside a VF, use
`T.rng_init(seed, seq=..., off=...)` with `T.rng_rand()` or
`T.rng_rand_float()` (uniform) / `T.rng_rand_float(dist="normal")`. Choose
explicit sequences when reproducibility must be independent of launch geometry.

### 2.3 SIMD VF: Register-Level Vector (No Threads)

`T.SimdVF()` maps to CCE MicroAPI vector instructions (2048-bit vectors,
64 x float32 or 128 x float16). There is **no thread concept** -- the
code runs as SIMD vector instructions directly on the vector unit.

```python
with T.SimdVF():
    mask = T.simd.pset(32)  # All lanes active for float32 elements.
    for i in range(N // 64):
        r0 = T.simd.vld(buf[i * 64])
        r1 = T.simd.vadd(r0, r0, mask)
        T.simd.vsts(out[i * 64], r1, mask)
```

### 2.4 Task Boundaries

`T.Task()` groups serial statements into one scheduling task. All statements
must use one hardware PIPE and one AIC/AIV core affinity; the marker does not
select a core or insert synchronization inside its body. Split work requiring
an internal cross-PIPE handoff into separate tasks.

Optional `latency=` and `ii=` values override the cost of the complete task:
latency is the output-ready delay and II is the minimum issue interval on its
PIPE. The final pair must satisfy `latency >= ii > 0`; omitted values are
estimated. These are distinct from the enclosing loop's solved II.

```python
with T.Task():  # Two serial GM-to-L1 copies on MTE2.
    T.copy(A_tile, a_l1)
    T.copy(B_tile, b_l1)
```

### 2.5 Per-Core Task Contract

`T.PerCoreTask()` describes one logical operation whose concrete execution
site is selected by the kernel block index. It is not a general-purpose
conditional scope. Every dependency-bearing basic operation inside it is an
explicit `T.Task()` or a statement that the compiler can infer as one.

For every dynamic invocation and every participating `blockIdx.x` value, the
candidate guards for that logical operation must be mutually exclusive and
collectively exhaustive: **exactly one candidate `T.Task()` must execute on
each core**. Conditions may dispatch work by block/core ID, but must not make
the operation data-dependent, optional, or executable more than once on one
core. All candidates must use the same single hardware pipe and represent the
same logical read/write role.

Put the complete block-index dispatch inside one `T.PerCoreTask()` so the
scheduler can treat all branches as alternative sites of the same operation:

```python
# Correct: every block executes exactly one candidate of one logical task.
with T.PerCoreTask():
    if bx == 0:
        with T.Task():
            T.copy(A, tile)
    else:
        with T.Task():
            T.copy(B, tile)
```

Do not split the alternatives into independently guarded `T.PerCoreTask`
regions, and do not omit a block-index case:

```python
# Wrong: these are two unrelated logical tasks, not two candidates of one task.
if bx == 0:
    with T.PerCoreTask():
        T.copy(A, tile)
if bx != 0:
    with T.PerCoreTask():
        T.copy(B, tile)

# Wrong: blocks for which `runtime_condition` is false execute no candidate.
with T.PerCoreTask():
    if runtime_condition:
        with T.Task():
            T.copy(A, tile)
```

AutoSchedule inserts the logical task's synchronization beside every candidate
`T.Task()` while preserving its block-index branch. Correctness therefore
depends on the exact-once contract above; the compiler validates structure,
pipe compatibility, and guard relationships, but cannot generally prove that
arbitrary runtime predicates form an exact partition.

---

## 3. Core Architecture: AIC vs AIV

The Ascend NPU has two distinct compute cores on each die:

| Core | Engine | Purpose | TileLang Scope |
|---|---|---|---|
| AIC | Cube unit | Matrix multiply (GEMM, convolutions) | `T.Cube()` |
| AIV | Vector unit | Element-wise ops, reductions, data movement | `T.Vector()` |

### 3.1 Pure Kernel (Single-Scope)

Use `T.Kernel(N)` without explicit `T.Cube()`/`T.Vector()` scopes. The
codegen auto-detects the kernel type based on the operations inside:
Cube ops (`T.gemm`) produce a `__cube__` kernel on AIC, while Vector ops
produce a `__vector__` kernel on AIV.

```python
with T.Kernel(NUM_BLOCKS) as bx:
    x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
    w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
    res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
    # ... T.copy into L1, T.gemm, T.copy out ...
```

### 3.2 MixedKernel: Sub-Core Access and Selection

Use `T.MixedKernel(NUM_BLOCKS, sids=2)` when the program needs the AIV sub-core
ID returned with `bx`, or explicitly controls the active AIV count (`sids=1`
or `2`). If neither is needed, use `T.Kernel`; the compiler handles core
splitting and the appropriate `T.dual_copy` lowering.

```python
with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
    ...  # Partition each AIV's accesses according to sid and buffer ownership.
```

See the mixed-core epilogues in `examples/ascend/deepgemm/kernels/epilogue.py`.
Follow the actual buffer extents when partitioning output; a local half-sized
buffer is not a full-sized buffer to split again.

### 3.3 Fully Manual Mixed Kernels

Explicit `T.Cube()` and `T.Vector()` scopes with manually managed buffer slots
and intra-/cross-core flags belong to the fully manual path. Disable the
scheduling path with `PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False`. Read the
[synchronization reference](references/synchronization.md) for the complete
initialization, reuse and final-drain protocol. Small manual mixed-core
regressions are in `testing/ascend/language/test_tilelang_ascend_nd2nz_scatter.py`.

---

## 4. Memory Hierarchy and Allocation

Ascend memory hierarchy (top to bottom):

```
GM (HBM)                       -- global memory, large capacity
  ↕ DMA (MTE pipes)
UB (Unified Buffer, "shared")  -- T.alloc_shared(...)
  ↕ DMA
L1 (CBuf, Cube Buffer)         -- T.alloc_l1(...)
  ↕ load
L0A / L0B (Cube inputs)       -- T.alloc_l0a/l0b(...)
  → Cube MAD unit
L0C (accumulator)              -- T.alloc_l0c(...)
```

### 4.1 Allocation API

```python
# Unified Buffer (UB) -- general-purpose on-chip memory
buf = T.alloc_shared((STAGES, TILE_M, TILE_K), "bfloat16")

# Cube Buffer (L1) -- fast buffer for Cube engine input
buf = T.alloc_l1((TILE_M, TILE_K), "bfloat16")

# Cube L0 buffers -- direct Cube engine inputs/outputs
buf_a = T.alloc_l0a((STAGES, TILE_M, TILE_K_SUB), "bfloat16")
buf_b = T.alloc_l0b((STAGES, TILE_N, TILE_K_SUB), "bfloat16")
buf_c = T.alloc_l0c((TILE_M, TILE_N), "float32")

# Fragment (register-level, inside SimtVF/SIMD)
frag = T.alloc_fragment((TILE,), "float32")
```

For a deferred reduction inside `T.SimtVF`, use the reducer epoch API. Given
float32 UB buffers `x_ub[TILE]` and `sum_ub[1]`:

```python
with T.SimtVF(threads=128):
    acc = T.alloc_reducer((1,), "float32", op="sum")
    T.reducer_init(acc)
    for i in T.Parallel(TILE):
        T.reducer_update(acc[0], x_ub[i])
    result = T.alloc_fragment((1,), "float32")
    T.finalize_reducer(acc, result)
    T.copy(result, sum_ub)
```

Read the independent destination fragment after finalization. The reducer
handle itself cannot be read, assigned directly, cleared/filled with `T.clear`
or `T.fill`, or finalized in place. Keep init and finalize in the same enclosing
loop/branch scope. Ascend currently supports reducer `sum`, `max`, and `min`;
bitwise collectives are unsupported.

### 4.2 Buffer Versioning

Use `T.annotate_buffer_versions` to fix a version count, select `auto`,
`iteration`, or `counter` indexing, or specify both. For example,
`T.annotate_buffer_versions({buf: (2, "counter")})` fixes two counter-indexed
versions. See [version semantics](references/synchronization.md#buffer-versions-and-offsets)
for ownership, control-flow, and automatic buffer sizing. In particular,
`{buf: 1}` opts out of multibuffer eligibility, while `{buf: (1, "auto")}` keeps
eligibility with one physical version. Explicit `counter` mode cannot serve
accesses to the same buffer at different stages within one owner.

Prefer compiler-managed multibuffering. [Manual multibuffering](references/synchronization.md#manual-multibuffering-limited-support)
has limited support and is not recommended: complex access or control-flow
patterns may produce incorrect dependency analysis and synchronization.

### 4.3 Physical Capacity and Reuse

Allocation shape, a copy's access region, and its physical layout are different
quantities. Include layout padding, buffer versions, and compiler reservations
when checking capacity. Layout normalization does not initialize padded lanes.

Different Buffer views can alias, and separate logical buffers can share an
allocation when their asynchronous lifetimes permit reuse. Follow the
[allocation ownership rules](references/synchronization.md#allocation-ownership)
when changing manual kernels. For capacity or launch failures, record the
generated allocation sizes, buffer versions, VF mode, and compiler options.
Check SDK reservations as well as user buffers before changing tile sizes.

`T.annotate_unlimit_memory("shared", "shared.l1")` inside a kernel removes the
scheduler's capacity constraint for those pools. Other supported scopes are
`shared.l0a`, `shared.l0b`, and `shared.l0c`. This is useful for investigating a
capacity-constrained schedule; it does not enlarge hardware memory or make an
over-capacity allocation executable. Verify physical allocations before launch.

---

## 5. GEMM Constraints

### 5.1 Select the GEMM Lowering Path

The current L1-input path in `tilelang/ascend/op/gemm/gemm_mad.py` requires
`transpose_A=False` and `transpose_B=True`. L0-input GEMM has its own
layout/transpose handling; do not generalize the L1 restriction to every
Ascend GEMM form. When sharing GEMM macros across loop orders, recheck compact
L0 layouts, buffer lifetimes, and stage counts; equivalent mathematics alone
does not make the storage and scheduling contracts interchangeable.

```python
T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(kt == 0))
```

### 5.2 Weight Layout Convention

For this L1-input NT convention, weights are typically stored as
`[N, K]` in global memory (transposed relative to standard `[K, N]`):

```python
# Weight buffer shape: [N, K], not [K, N]
W: T.Buffer((N_DIM, K_DIM), dtype)
```

### 5.3 Clear Accumulator

SIMT supports `T.atomic_add`, `T.atomic_max` and `T.atomic_min` on GM values;
`return_prev=True` returns the value before the update. For a DMA/FixPipe store,
`T.set_atomic("add", dtype)` selects an accumulating store until
`T.set_atomic_none()` restores ordinary stores. Initialize the destination before
any accumulating write; the split-K example shows cross-core initialization.

For split-K and backward reductions, define which task owns each partial
output and where it is initialized and combined. Atomic and staged reductions
can differ in determinism; preserve the caller's requirement.

```python
T.gemm(a, b, c, transpose_B=True, clear_accum=True)  # for first k-step
T.gemm(a, b, c, transpose_B=True, clear_accum=False)  # for subsequent
```

### 5.4 HF32 Mode (fp32 throughput tradeoff)

For a supported fp32 GEMM path, HF32 trades precision for throughput; measure
the effect on the actual kernel and validate its numerical tolerance:

```python
T.set_hf32_mode("nearest_even")  # or "nearest_zero"
# ... T.gemm calls ...
T.set_hf32_mode(None)  # restore full fp32
```

---

## 6. Data Movement

### 6.1 T.copy

Basic DMA copy. Ascend-specific parameters:

```python
# GM → L1
T.copy(X[row_slice, col_slice], x_l1)

# L1 → L0A/L0B (used in L0-staged GEMM)
T.copy(x_l1[f, :, k_slice], x_l0[sf, :, :])

# L0C → UB (via dual_copy for M-split) or L0C → GM
T.copy(res, C[row, col])

# GM → L1 with an explicit L2 load policy
T.copy(x_gm, x_l1, l2_cache_ctrl="NOTALLOC_KEEP")
```

**Transpose is path-specific.** On GM→L1, `transpose=True` selects `dn2nz`
instead of `nd2nz` and reverses the logical source/destination axes. On
L1→L0A/L0B, it requests a transposed load; the effective transpose is the XOR
of this request and the source/destination fractal-major mismatch. Check the
resulting hardware grouping and destination capacity. It is not a general
transpose operation for arbitrary memory paths.

**L2 cache control:** names are case-insensitive and use the load/store policy
mapping in `tilelang/ascend/language/copy_op.py`. Defaults depend on the API and
path:

| API/path | Default policy |
|---|---|
| `T.copy`, GM→L1 or GM→UB | `NORMAL_FV` (0) |
| `T.copy`, UB→GM | `NOTALLOC_CI` (4) |
| `T.dual_copy`, UB→GM | `NORMAL_FV` (0), explicitly passed by its frontend |

**Padded GM→UB copy (`pad_value=` / `data_select=`).** Ordinary GM→UB DMA
supports byte-granular rows with independent source/destination pitches;
non-32B rows do not by themselves require padding. Opt in to fill the row tail
up to the next 32B boundary when the kernel needs those padded lanes.
Padding requires a statically known positive row byte length. The destination
UB allocation and row pitch must accommodate the padded width so rows do not
overlap. Two mutually exclusive modes:

```python
# Mode 1: pad_value=v — this copy sets the fill value AND pads.
# 30 fp32 cols = 120B (not 32B-aligned) -> padded to 128B; tail lanes = -1.0.
a_ub = T.alloc_shared((M, 32), T.float32)  # over-allocated to 128B rows
T.copy(A[:, :], a_ub[:, :30], pad_value=-1.0)

# Mode 2: data_select=True — pad, but reuse the pad register set beforehand.
# Reuse while the required value/dtype match and no intervening setter changes it.
T.ascend_set_copy_pad_value(-1.0, dtype="float32")
T.copy(A[:, :], a_ub[:, :30], data_select=True)
```

- `pad_value` emits a leading `T.ascend_set_copy_pad_value(v)` so AutoSchedule
  inserts the PIPE_S→PIPE_MTE2 sync between the scalar pad-register write and
  the MTE2 copy that reads it. The fill dtype is the destination element dtype.
- `data_select=True` uses the current pad-register value without setting or
  restoring it. Set a value with the matching dtype before its first use.
  Reuse across copies requires that no intervening operation changes the
  register; another `T.copy(..., pad_value=...)` also changes it. After such a
  change, set the desired value again before relying on `data_select=True`.
- Passing both `pad_value` and `data_select` raises `ValueError`.
- Ascend GM→UB only; supported dtypes are 8/16/32-bit (int8/uint8/int16/
  uint16/float16/bfloat16/int32/uint32/float32). Ignored on other paths/backends.
- Numerical coverage: `test_gm_to_ub_pad_value` in
  `testing/ascend/language/test_tilelang_ascend_dma_copy.py`.

### 6.2 T.dual_copy (Ascend-Only)

Distributes a logical tile across two AIV sub-cores in a mixed kernel. The
kernel must also contain Cube-side work; `dual_copy` alone cannot turn a pure
Vector kernel into a two-AIV launch.

| Path | Implementation and constraints |
|---|---|
| L0C→UB | Hardware dual-destination copy; matching source/destination dtypes, rank at least two; N-split requires full N divisible by 32 |
| GM→UB, UB→GM, UB→L1 | Per-AIV software copy indexed by sub-core ID; also supports one-dimensional regions with a 2:1 extent ratio |

For rank-two or higher regions, exactly one trailing dimension has a 2:1
relationship; the larger region is the full tile. The split is inferred from
region extents, with allocation shapes as a fallback for runtime tails.
For L0C→UB:

- **M-split** (`src[M,N] -> dst[M/2,N]`): AIV0 gets rows `[0,M/2)`,
  AIV1 gets rows `[M/2,M)`.
- **N-split** (`src[M,N] -> dst[M,N/2]`): splits across columns.

Statically known full split extents must be even. For L0C→UB dtype conversion,
use a separate supported copy or cast in a VF. With auto-scheduling disabled,
place software dual copies inside an explicit `T.Vector()` block within
`T.Kernel`, alongside the Cube work; `T.MixedKernel` is unsupported in this
manual path.

```python
# L0C → UB (M-split for 2 AIVs)
T.dual_copy(res_l0c, temp_ub)

# UB → L1 (ND-to-NZ conversion is automatic via InsertNd2Nz pass)
T.dual_copy(p_ub, p_l1)

# UB → GM (M-split write)
T.dual_copy(o_ub, O[row_start:row_end, 0:D])

# With L2 cache control
T.dual_copy(temp, C[row, col], l2_cache_ctrl="NOTALLOC_PW")
```

### 6.3 Strides and Tails

Preserve caller-provided strides and storage offsets. A sliced region need not
have a compact allocation pitch, and rounding an on-chip transfer does not
change the logical GM output extent. Padded lanes need explicit initialization
when subsequent operations consume them.

For high-rank DMA, compact L0 regions, MX scale storage, or L0C-to-UB alignment,
inspect the generated transfer operands against the selected SDK's instruction
contract. Check that rounded physical accesses fit the declared buffer views.
Tail changes can also affect [UnitFlag lifecycles](references/synchronization.md#unitflag-and-tail-lifecycles).

---

## 7. Synchronization

Automatic scheduling is enabled by default. `T.Pipelined` describes pipelined
loops and `T.Persistent` describes outer persistent work. The compiler manages
core assignment, flags, and buffer versions for supported programs.

Use one of three modes, according to the requested control:

- Automatic scheduling: ordinary tasks and loops; buffers need no extra manual
  stage dimension solely for compiler-managed multibuffering.
- `T.Stage` scheduling: constrain frontend stages and per-PIPE source order,
  while retaining automatic dependency analysis, sync, and multibuffering.
- Fully manual scheduling: explicit core scopes, slots, and flags with
  `TL_ENABLE_AUTO_SCHEDULE=False`.

Read [synchronization and multibuffering](references/synchronization.md) for
`T.Stage` nesting, `enable_offset`, version modes, and the full set/wait protocol.
Do not equate a stage number with a physical slot or rely on estimated latency
in place of a required synchronization edge.

For performance tuning, use `T.Stage` to test task-order and pipeline-overlap
hypotheses against an automatic-schedule baseline. Keep automatic dependency
analysis and synchronization enabled; follow the
[manual schedule tuning procedure](references/synchronization.md#tune-performance-with-tstage).

When replacing explicit flags with automatic scheduling or `T.Stage`, map each
operation to its task, buffer region, and PIPE. Remove explicit synchronization
only once the compiler manages the required buffer handoff and reuse. Preserve
guards, tails, and active sub-core partitioning.

Manual flag APIs remain available. Follow a complete current example instead
of combining fragments of manual and automatic protocols. For explicit pipe
barriers:

```python
T.ascend_pipe_barrier("PIPE_ALL")
T.ascend_pipe_barrier("PIPE_V")
T.ascend_pipe_barrier("PIPE_MTE1")
```

---

## 8. SIMD MicroAPI (T.simd.*)

Low-level CCE vector intrinsics for `T.SimdVF()` blocks. These map
directly to Ascend CCE MicroAPI instructions (2048-bit vectors).

For lane widths, rounding, saturation, dtypes, and predicates, check the
installed SDK headers and matching official Ascend documentation.

### 8.1 Core Operations

```python
with T.SimdVF():
    mask = T.simd.pset(32)  # all lanes active for 32-bit elements

    # Load/Store
    r = T.simd.vld(buf[offset])  # 2048-bit vector load
    T.simd.vsts(buf[offset], r, mask)  # 2048-bit vector store

    # Predicate Load/Store
    pred = T.simd.pld(pred_buf[runtime_offset], dist="US")
    T.simd.pst(pred_buf[runtime_offset], pred, dist="PK")

    # Arithmetic
    r = T.simd.vadd(a, b, mask)
    carry, r = T.simd.vaddc(a, b, mask)  # int32/uint32 only; carry is boolx256
    r = T.simd.vmul(a, b, mask)
    r = T.simd.vmax(a, b, mask)
    r = T.simd.vsub(a, b, mask)

    # Math
    r = T.simd.vexp(a, mask)
    r = T.simd.vln(a, mask)

    # Type conversion
    r = T.simd.vcvt(a, "float32")  # cast to float32
    r = T.simd.vcvt(a, "bfloat16")  # cast to bfloat16

    # Interleave/Deinterleave
    a0, a1 = T.simd.vintlv(x, y)  # interleave even/odd lanes
    x, y = T.simd.vdintlv(a0, a1)  # deinterleave back
```

`vld`, `vsts`, `pld`, and `pst` do not have a separate scalar `off`
parameter. Express the displacement in the buffer address, as shown above;
the Ascend code generator supplies the hardware-required zero offset. `vld2`
retains its `off` parameter because that operand has address-register semantics.

Confirm less common intrinsics in `tilelang/ascend/language/simd.py`;
the examples above are not an exhaustive API list.

### 8.2 Explicit Intrinsics and High-Level SIMD

Use explicit `T.simd.*` calls when the task needs a specific instruction, mask,
rounding mode, or evaluation order, as in the SIMD implementation of
`examples/ascend/example_per_token_cast_to_fp8.py`. High-level `T.Parallel`
inside `T.SimdVF()` is lowered by `AscendSimdVFLowerParallel`. Check the
current supported pattern and emitted code before replacing one form with the
other. If high-level lowering rejects the required pattern, express it with explicit `T.simd.*`
operations and validate the result.

### 8.3 VF Latency Annotation

`T.SimtVF` and `T.SimdVF` accept `latency=` in cycles as scheduling-model input.
Without an annotation, the compiler estimates VF cost from operations and
bandwidth; it does not automatically run the simulator during compilation.

```python
with T.SimdVF(latency=128):
    ...
with T.SimtVF(threads=256, latency=512):
    ...
```

Preserve the source and conditions of a measured estimate. If one macro scope
has several measured expansions, use their largest cost for its shared
annotation. Recompile and validate correctness after changing `latency=`.
Inaccurate estimates can produce poor schedules, but correctness still requires
sound dependencies and synchronization. Use
[operation cycle measurements](#124-measure-operation-cycles) for isolated cycle estimates
and [performance measurement](#12-performance-measurement-and-tuning) for
whole-kernel device timing; these measure different quantities.

---

## 9. Programming Patterns

The top-level `examples/ascend/` directory contains complete applications,
with one matching test per example. Use DeepGEMM for advanced matrix kernels:

| Task | Example under `examples/ascend/` |
|---|---|
| Persistent GEMM with swizzling, precision modes and mixed epilogues | `example_gemm.py` |
| Split-K with atomic or ordered reduction | `example_gemm_splitk.py` |
| Persistent traversal, L0 staging, mixed epilogues and quantized GEMM | `deepgemm/` |
| SIMT and explicit SIMD vector addition with pipelined copies | `example_vecadd.py` |
| SIMT reduction and normalization | `example_rmsnorm.py` |
| SIMT and SIMD group-wise FP8 quantization | `example_per_token_cast_to_fp8.py` |
| MHA / GQA | `flash_attention/example_mha.py` / `flash_attention/example_gqa.py` |
| Explicit pipeline stages in attention | `flash_attention/example_gqa_manual_schedule.py` |

Feature regressions live in `testing/ascend/`; do not import application kernels
into that suite. Use small `T.Tensor` fixtures at the owning compiler boundary;
see its `README.md` for test design and placement.
Copy padding is described in [Data movement](#61-tcopy). Version annotations,
cross-level ownership, explicit stages, manual buffer rings and while-loop
scheduling are described in the [synchronization reference](references/synchronization.md).

The RMSNorm example demonstrates `T.alloc_fragment` for vectorized loads and
register reuse. For reductions, follow the example in
[Allocation API](#41-allocation-api). Preserve intermediate casts and precision
when fusing normalization with other operations.

For attention examples, preserve head mapping, masks, scale placement,
normalization, and accumulation order; validate all gradient outputs when
adapting a backward kernel.

For regression coverage, start with `testing/ascend/language/` (including
`simtvf/` and `simdvf/`) for DSL and VF behavior. The sibling `layout/`,
`analysis/`, `auto_schedule/`, `transform/`, `target/`, and `runtime/` suites
cover their respective compiler and execution boundaries; shared framework
tests live in `testing/python/`. Integration tests also accompany the examples.
Run selected tests with `python -m pytest path/to/test.py`. Inspect fixtures
and setup for NPU requirements: source assertions alone do not establish
hardware independence. Inspect a file's main guard before running it directly.
Exercise applicable tails, short or zero dynamic lengths, non-contiguous inputs,
head groups, reduction boundaries, and reused outputs. Validate both the kernel
and its original caller, with repeated-run checks when determinism is required.

---

## 10. Common Pitfalls

1. **GEMM path mismatch**: The current L1-input lowering requires
   non-transposed A and transposed B. Validate the selected path and layouts
   before changing a transpose flag.

2. **Using `threads=` on `T.Kernel`**: Raises `TypeError` immediately.
   Thread domains go inside `T.SimtVF(threads=N)` scopes.

3. **Multi-dimensional grids**: Ascend only supports `T.Kernel(N)`.
   For 2-D decomposition, compute mapping yourself: e.g.,
   `row = tile_idx // N_TILES; col = tile_idx % N_TILES`.

4. **Weight layout**: The illustrated L1-input NT path uses `[N, K]`
   weights. Preserve the caller's layout or transform it explicitly.

5. **Buffer dimensions with auto-schedule**: The compiler adds required
   versions. Distinguish these from dimensions already belonging to the
   application's layout; avoid accidental double buffering of explicit slots.

6. **Accumulator dtype**: The illustrated fp16/bf16 GEMMs accumulate
   into float32. Check the chosen GEMM and SDK contract for other dtype
   combinations.

7. **Reducer epochs**: Use `T.reducer_init(acc)`,
   `T.reducer_update(acc[i], value)`, and `T.finalize_reducer(acc, dst)`.
   Read `dst` after finalization; `T.clear`/`T.fill`, direct reducer accesses,
   and in-place finalize are invalid for these reducer handles.

8. **SIMT VF thread sync**: The compiler auto-inserts `asc_syncthreads()`
   based on data-flow analysis. When writing to UB in one thread and
   reading in another, sync is automatic. No manual barriers needed.

9. **dual_copy shape mismatch**: For the illustrated L0C-to-UB M-split
   across two AIVs, the destination holds half the rows. Other copy directions
   have different ownership and shape relationships; inspect their lowering.

10. **AIV sid indexing in output**: In `T.MixedKernel`, `T.copy` from
    UB to GM uses `sid` to partition output rows. Forgetting the `sid`
    offset causes both AIV sub-cores to write the same region.

11. **Hardware warp-reduce dtypes**: `T.warp_reduce_sum`,
    `T.warp_reduce_max`, and `T.warp_reduce_min` lower directly to the
    `asc_reduce_*` hardware intrinsics, which support only `float16` (`half`),
    `float32`, `int32`, and `uint32`. Other `T.reduce_*` dtypes may still use a
    shuffle or UB fallback, but cannot call the hardware reduction directly.

---

## 11. Compilation and Execution

Given a PrimFunc `program` with bf16 inputs `X[M, K]`, `W[N, K]`, and one output:

```python
import tilelang
import torch

kernel = tilelang.compile(program, target="ascend", out_idx=-1)
print(kernel.get_kernel_source())
x = torch.randn((M, K), device="npu", dtype=torch.bfloat16)
w = torch.randn((N, K), device="npu", dtype=torch.bfloat16)
result = kernel(x, w)
torch.npu.synchronize()
```

For a fully manual flag kernel, additionally pass
`pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}`.
A staged kernel selects manual constraints through `T.Stage` without that
switch. For multiple output parameters, set their indices with `out_idx=[1, 3]`
as appropriate to that program's signature. Preserve the caller's mutation and
return conventions, dynamic dimensions, scalar parameters, and numerical
tolerances when adapting the kernel interface.

For resource-related compilation or launch failures, inspect existing
`compile_flags`, generated allocations, and the selected SDK's resource rules.
Keep compiler capacity rejection, vendor compilation, and device launch failures
distinct; compare one kernel configuration at a time.

For source-only compilation and pass traces, use the
[compile-only tool](../../../docs/tools/compile_only.md) and
[lowering trace](../../../docs/tools/lower_trace.md); keep device work out of
import-time code. To modify generated source before compilation,
see `tilelang.ascend.callback.register_ascend_postproc_callback` and
`testing/ascend/target/test_ascend_postproc.py`. Follow the
[controlled experiment guidance](#125-attribute-the-difference) before drawing
performance conclusions.

---

## 12. Performance Measurement and Tuning

### 12.1 Locate the performance mechanism

Start from the kernel and its measured callable. Distinguish tile/configuration
choice, data movement, and runtime launch cost before choosing an experiment.
Use generated source and a lowering trace to distinguish a changed schedule
from changed instruction lowering; see [lowering traces](../../../docs/tools/lower_trace.md).
The [profiling reference](references/profiling.md#measurement-and-tuning-components)
maps repository measurement, tuning, and regression tools.

### 12.2 Define a comparable experiment

Read the benchmark entry point, case generator, configuration, and result
schema. Enumerate expanded shapes, dtypes, layouts, scalar parameters, and
repetitions; record correctness and performance coverage separately.
Keep baseline results unchanged during measurement, and check whether harness
options overwrite them.

Record both revisions, Python/native origins, CANN/Bisheng and torch_npu
versions, device model and selected device, shape, dtype, strides, tile sizes,
launch geometry, dynamic/static parameters, deterministic mode, and pass config.
Select the device explicitly and check for competing workloads. Record how
process-visible device indices map to physical devices.

Check API behavior before timing. For GEMM, include grouped metadata,
accumulation/output dtypes, scaling, and output initialization. Record pipeline
depth and cache policy alongside tile sizes and core count; state whether
configurations are fixed or each implementation tunes independently. Freeze
both source revisions and loaded native extensions: clearing the TileLang
kernel cache does not rebuild a stale reference C++ extension.

Keep the same callable boundary and timing method across implementations. Decide
whether each callable includes allocations, output initialization, several
kernels, synchronization, or only the kernel launch. Keep required per-invocation
initialization inside the measured workload consistently. Separate first JIT,
module load, profiler initialization, and cache setup from steady-state timing.

### 12.3 Establish valid measurements

Use [profiling recipes](references/profiling.md) for current `do_bench`, msprof,
and regression-driver behavior. Confirm actual samples exist and correspond to
the intended kernel. A successful pytest run or process exit does not prove
profiling succeeded; read warnings, profiler output, and the measured case list.

Warm each variant, repeat complete measurements, and interleave A/B and reverse
order when drift matters. Keep input generation, tensor lifetime, cache policy,
and device activity comparable. Measure shared-device and idle performance
separately. For multi-device sweeps, keep each A/B pair on one device, avoid
worker contention, tag samples, and report results per device.

Choose aggregation and exclusion rules before examining results. Apply the
same validity criteria to both implementations; retain raw samples and reasons
for exclusion, preserving A/B pairing. Report units and variability rather than
an isolated minimum. An early-stop estimate is a different measurement path
and must not be mixed with full profiler samples.

Bound resampling by a repeat count or time budget, including the warmup and
launches within each profiler call. For short kernels, estimate an absolute
noise floor from repeated same-kernel measurements. If variability still
obscures the gap, report the case as inconclusive and investigate contention
or the timing method.

Start with a representative failing case, then cover the requested performance
set. Report missing, skipped, or failed cases explicitly. Keep correctness and
performance status separate; do not optimize an incorrect kernel into a new
baseline.

### 12.4 Measure operation cycles

Use npusim to investigate the cycles spent in a VF, data transfer, Cube
operation, synchronization handoff, or a sequence of these in your kernel.
Choose the measured region and preserve its shapes, dtypes, layouts, buffer
lifetimes, and dependencies. For `T.SimtVF` and `T.SimdVF`, also preserve the
actual call site, thread dimensions, masks, and loop trip counts. Match macro
expansions to the source scope rather than generated declaration order.

Follow the [cycle measurement recipe](references/profiling.md#operation-cycle-measurements)
to isolate the operation, compile a minimal launcher, record its execution,
and identify the relevant instruction interval. Measure each relevant
input-dependent path and retain the source, binary, build command, launcher,
input conditions, and trace with the result.

Verify the captured kernel and instruction count, check outputs against a
reference, and identify dispatch versus completion events before quoting
cycles. For repeated work, distinguish dependent latency from independent
throughput; see [repeated operations](references/profiling.md#repeated-operations-and-throughput).

To apply measured task costs, use
[`T.Task(latency=..., ii=...)`](references/profiling.md#override-automatic-task-estimates).
For VF-specific annotations, follow the
[VF latency annotation guidance](#83-vf-latency-annotation).
Keep simulated cycles, modeled II, compiler solve time, and measured device
latency separate. Whole-kernel overlap, transfers, and memory effects require
benchmarking the actual kernel after rescheduling.

### 12.5 Attribute the difference

Compare generated source and, where useful, compiled code before changing the
compiler. Then investigate the smallest relevant variable: task order, sync,
buffer count, actual copy volume, memory layout, core partition, or tiling.
Separate changes to reference configuration heuristics and numerical formulation
from changes to code generation when attributing a performance difference.
Keep model latency and solved II separate from measured device time. Use
[lowering traces](../../../docs/tools/lower_trace.md) to locate the first changed
representation when generated schedules or synchronization differ.

Use a postprocessing callback or a faithful manual replay for a narrowly scoped
code-generation hypothesis. Record the exact perturbation, preserve its guards
and memory protocol, verify correctness, and rerun the same harness. Validate
the resulting optimization across the supported cases before adopting it.
Quantify how much of the latency gap the perturbation explains; a generated-code
difference alone does not establish the cause of a runtime regression.

Identical generated code or ELF does not fix runtime addresses, cache state,
input values, device state, or timing boundaries. If a discrepancy persists,
hold those conditions fixed one at a time. Conversely, failure to reproduce in
one controlled environment supports only that tested scope.

### 12.6 Interpret bottlenecks and deliver

Use profiler pipe statistics to form hypotheses, then test them. Cube and
Vector utilization have different denominators; overlapping pipe ratios need
not add to 100%. Reconcile elapsed time, instruction activity, bytes transferred,
and the measured callable before declaring a bandwidth or compute bottleneck.
Simulated VF cycles and solved II are model inputs, not whole-kernel device time.

Deliver a compact case table with baseline and candidate latency, absolute gap,
ratio, correctness, and measurement validity. Define the ratio: for
`r = reference_us / candidate_us`, candidate slowdown is `1/r - 1`. Group larger
sweeps by suite, dtype, mode, and direction of change. Include the exact harness
command and environment, completed versus expected case counts, reference
failures, artifact paths, and the experiment supporting the cause. Distinguish
measured improvement from a proposed next experiment.
