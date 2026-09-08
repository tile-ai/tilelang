---
name: tilelang-ascend
description: >
  Guide for writing TileLang programs targeting the Huawei Ascend NPU.
  Covers Ascend-specific concepts (AIC/AIV dual-core, Cube/Vector engines,
  memory hierarchy, thread model, synchronization) and how they differ
  from CUDA/GPU programming. Use this skill whenever the user asks about
  Ascend, NPU, Huawei, or writes TileLang code for the Ascend backend.
---

# TileLang Ascend NPU Programming

You are helping the user write TileLang programs for the Huawei Ascend NPU.
The Ascend architecture differs substantially from CUDA GPUs. This skill
provides the concepts, API reference, programming patterns, and common
pitfalls you need to produce correct, high-performance Ascend kernels.

---

## 1. Mental Model: Ascend vs CUDA

| Concept | CUDA | Ascend NPU |
|---|---|---|
| Kernel launch | `T.Kernel(grid, threads=128)` | `T.Kernel(N)` -- **no threads arg** |
| Thread dimensions | 1-D, 2-D, or 3-D | 1-D grids only |
| Thread access | `T.get_thread_binding()` | No `threadIdx` directly; use `T.SimtVF(threads=N)` |
| Compute cores | SM (unified) | **Two distinct cores**: AIC (Cube) + AIV (Vector) |
| Matrix multiply | `T.gemm(...)` | `T.gemm(..., transpose_B=True)` -- **mandatory** |
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

# WRONG - will raise ValueError:
with T.Kernel(NUM_CORES, threads=128) as bx:
    ...
```

### 2.2 SIMT VF: Thread-Parallel Compute

Use `T.SimtVF(threads=N)` to define a thread-parallel region. Inside
this block, `T.Parallel` loops distribute work across threads. The
codegen automatically inserts `threadIdx.x` mappings.

```python
with T.SimtVF(threads=128):
    for i in T.Parallel(N):
        out[i] = a[i] + b[i]
```

Auto-sync: The compiler inserts `asc_syncthreads()` when it detects
cross-thread data hazards (see `example_simtvf_auto_sync.py`). Thread
binding is accessed via `T.get_thread_binding()`.

### 2.3 SIMD VF: Register-Level Vector (No Threads)

`T.SimdVF()` maps to CCE MicroAPI vector instructions (2048-bit vectors,
64 x float32 or 128 x float16). There is **no thread concept** -- the
code runs as SIMD vector instructions directly on the vector unit.

```python
with T.SimdVF():
    for i in range(N // 64):
        r0 = T.simd.vld(buf[i * 64])
        r1 = T.simd.vadd(r0, r0, mask)
        T.simd.vsts(out[i * 64], r1, mask)
```

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

### 3.2 MixedKernel: Auto-Scheduled Cube + Vector

**`T.MixedKernel` exists solely to provide the `sid` variable** (AIV
sub-core ID: 0 or 1) for manual output partitioning. If you don't need
`sid`, just use `T.Kernel` — the compiler handles `T.dual_copy` and AIC/AIV
splitting automatically.

```python
# MixedKernel: only needed when you want sid for manual partitioning
with T.MixedKernel(NUM_BLOCKS) as (bx, sid):
    ...
    T.copy(temp[sid * (TILE_M // 2):...], C[...])

# Equivalent without sid: just use T.Kernel + T.dual_copy
with T.Kernel(NUM_BLOCKS) as bx:
    ...
    T.dual_copy(res, temp)
    T.dual_copy(temp, C[...])
```

Reference: `examples/ascend/example_gemm_mixedkernel.py`

### 3.3 Manual Mixed: T.Cube() and T.Vector() inside T.Kernel

The explicit pattern for full manual control. Use `T.Kernel(N)` with
nested `T.Cube()` and `T.Vector()` scopes, and manage cross-core
synchronization with manual flags:

```python
with T.Kernel(NUM_BLOCKS) as bx:
    with T.Cube():
        T.ascend_cross_core_set_flag(4, "PIPE_FIX", my_aiv_flag)
        ...
        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", my_aic_flag)

    with T.Vector() as sid:
        my_aic_flag = 6 + sid * 16
        my_aiv_flag = 4 + sid * 16
        T.ascend_cross_core_wait_flag(4, "PIPE_FIX", my_aic_flag)
        ...
```

Reference: `examples/ascend/example_gemm_mix_manual.py`

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

# Reducer (warp-level reduction accumulator)
reducer = T.alloc_reducer((1,), "float32", op="sum", replication="all")
```

### 4.2 Buffer Versioning

The auto-schedule pass estimates operation latency to determine buffer
lifetimes and pipelining strategy. Use `T.annotate_buffer_versions` to fix a
version count, select the indexing mode, or do both:

```python
buf1 = T.alloc_shared((TILE,), "float32")
buf2 = T.alloc_shared((TILE,), "float32")
buf3 = T.alloc_shared((TILE,), "float32")
T.annotate_buffer_versions({
    buf1: 2,                # fixed count, automatic mode
    buf2: (2, "counter"),  # fixed count and mode
    buf3: "iteration",     # mode only; infer the count
})
```

The supported modes are `"auto"`, `"iteration"`, and `"counter"`.

---

## 5. GEMM Constraints

### 5.1 Mandatory transpose_B=True

The Ascend Cube MAD unit requires `transpose_B=True`. This is a hardware
constraint -- the B matrix must be in transposed layout in L1.

```python
# ALWAYS use transpose_B=True on Ascend
T.gemm(a_l1, b_l1, c_l0c, transpose_B=True, clear_accum=(kt == 0))
```

### 5.2 Weight Layout Convention

Because of `transpose_B=True`, weights are typically stored as
`[N, K]` in global memory (transposed relative to standard `[K, N]`):

```python
# Weight buffer shape: [N, K], not [K, N]
W: T.Buffer((N_DIM, K_DIM), dtype)
```

### 5.3 Clear Accumulator

```python
T.gemm(a, b, c, transpose_B=True, clear_accum=True)   # for first k-step
T.gemm(a, b, c, transpose_B=True, clear_accum=False)  # for subsequent
```

### 5.4 HF32 Mode (fp32 throughput tradeoff)

For fp32 GEMM, enable HF32 to trade precision for ~2x throughput:

```python
T.set_hf32_mode("nearest_even")   # or "nearest_zero"
# ... T.gemm calls ...
T.set_hf32_mode(None)              # restore full fp32
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

# Ascend-specific params:
T.copy(src, dst, l2_cache_ctrl="NOTALLOC_KEEP")   # L2 cache bypass
T.copy(src, dst, transpose=True)                   # transposed DMA
```

**L2 cache control values:** `"NORMAL_FV"` (default), `"NOTALLOC_KEEP"`,
`"NOTALLOC_PW"`, `"NOTALLOC_CLEAN"`, etc.

**Padded GM→UB copy (`pad_value=` / `data_select=`).** An Ascend MTE GM→UB
copy requires each row to be a multiple of 32B. When a copied row is not
32B-aligned, a multi-row copy normally errors. Opt in to right-pad the row
tail up to the next 32B boundary and fill the pad lanes with a chosen value.
The destination UB buffer must be over-allocated to (at least) the 32B-aligned
row width so padded rows don't overlap. Two mutually-exclusive modes:

```python
# Mode 1: pad_value=v — this copy sets the fill value AND pads.
# 30 fp32 cols = 120B (not 32B-aligned) -> padded to 128B; tail lanes = -1.0.
a_ub = T.alloc_shared((M, 32), T.float32)      # over-allocated to 128B rows
T.copy(A[:, :], a_ub[:, :30], pad_value=-1.0)

# Mode 2: data_select=True — pad, but reuse the pad register set beforehand.
# Set once, reuse across several copies without re-setting.
T.ascend_set_copy_pad_value(-1.0, dtype="float32")
T.copy(A[:, :], a_ub[:, :30], data_select=True)
```

- `pad_value` emits a leading `T.ascend_set_copy_pad_value(v)` so AutoSchedule
  inserts the PIPE_S→PIPE_MTE2 sync between the scalar pad-register write and
  the MTE2 copy that reads it. The fill dtype is the destination element dtype.
- `data_select=True` pads but does NOT set the fill value — it uses whatever
  the hardware pad register currently holds, which you must set once via
  `T.ascend_set_copy_pad_value(v, dtype=...)` beforehand.
- Passing both `pad_value` and `data_select` raises `ValueError`.
- Ascend GM→UB only; supported dtypes are 8/16/32-bit (int8/uint8/int16/
  uint16/float16/bfloat16/int32/uint32/float32). Ignored on other paths/backends.
- Example: `examples/ascend/example_copy_pad_value.py` (+ `test_copy_pad_value.py`).

### 6.2 T.dual_copy (Ascend-Only)

Splits L0C data across 2 AIV sub-cores. The split direction is inferred
from the shape mismatch:

- **M-split** (`src[M,N] -> dst[M/2,N]`): AIV0 gets rows `[0,M/2)`,
  AIV1 gets rows `[M/2,M)`. Used in standard GEMM + MixedKernel.
- **N-split** (`src[M,N] -> dst[M,N/2]`): splits across columns.

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

---

## 7. Synchronization

### 7.1 Modern: Auto-Schedule (Recommended)

Use `T.Pipelined` for loop-level double buffering. Use `T.Persistent`
for outer persistent loops. The compiler auto-manages all flags. This
is enabled by default (`TL_ENABLE_AUTO_SCHEDULE: True`). Set
`TL_ENABLE_AUTO_SCHEDULE: False` to skip the complete scheduling path.

```python
for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
    m_tile, n_tile = compute_tile(tile_idx)
    for kt in T.Pipelined(K_TILES, num_stages=2):
        T.copy(X[m_slice, k_slice], x_l1)
        T.copy(W[n_slice, k_slice], w_l1)
        T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
    T.dual_copy(res, temp)
    T.dual_copy(temp, C[m_slice, n_slice])
```

Key points:
- Buffers do **not** need a leading `STAGES` dimension; the compiler
  auto-multi-buffers.
- No manual SetFlag/WaitFlag calls needed.
- Reference: `examples/ascend/example_gemm.py`

**Buffer version annotation**: The auto-schedule pass estimates
operation latency to determine buffer lifetimes and pipelining strategy.
When the automatic latency estimation is inaccurate (e.g. for
non-standard operations or complex control flow), use
`T.annotate_buffer_versions` to specify counts and/or indexing modes:

```python
buf1 = T.alloc_shared((TILE,), "float32")
buf2 = T.alloc_shared((TILE,), "float32")
T.annotate_buffer_versions({buf1: (2, "counter"), buf2: "auto"})
```

An integer fixes the count, `(count, mode)` fixes both, and a mode string leaves
the count to the scheduler. Supported modes are `"auto"`, `"iteration"`, and
`"counter"`.

**Pipeline offset control (`enable_offset`)**: Ascend hardware already
overlaps work on **different** resource pipes (MTE1/MTE2/MTE3/Cube/Vector/
Fixpipe/Scalar) asynchronously — that cross-pipe pipelining happens
automatically and is *not* affected by this flag. `enable_offset` only
changes how the scheduler treats tasks that share the **same** pipe, for
example the two `T.gemm` calls in flash attention (both on the Cube pipe).

By default (`enable_offset=False`) the Z3 loop scheduler keeps all tasks
on a given pipe within a single II window (`max(start) - min(start) < II`)
and at the same pipeline stage, so same-pipe ops are **not** offset across
loop iterations. To let the scheduler offset same-pipe tasks across
iterations/stages, opt in with the `enable_offset` annotation on the loop:

```python
for k in T.Pipelined(NUM_KV_BLOCKS, num_stages=2,
                     annotations={"enable_offset": True}):
    ...
```

- `enable_offset=False` (default): same-pipe tasks stay in one II window /
  one stage. Simpler, more predictable; use when the offset schedule is
  incorrect or unnecessary.
- `enable_offset=True`: the scheduler may stagger same-pipe tasks across
  iterations (e.g. issue iteration `i+1`'s GEMM before iteration `i`'s
  consumer finishes), at the cost of a larger II / more buffering.

Buffer multi-versioning (`T.annotate_buffer_versions`) is independent of
this flag and still applies in both modes.

**Manual schedule mode** keeps dependency analysis, synchronization, and
multi-buffering, but constrains Z3 with per-pipe source order and frontend
stages. It is selected independently for each scheduled child list: if any
direct child task or loop carries `T.Stage`, that list uses manual scheduling.
Nested lists without a staged child remain automatic.

```python
for k in T.Pipelined(NUM_KV_BLOCKS, num_stages=2,
                     annotations={"enable_offset": True}):
    with T.Stage(0):
        T.copy(A[k], a_l1)
        T.copy(B[k], b_l1)
    with T.Stage(1):
        T.gemm(a_l1, b_l1, accum, transpose_B=True)
```

Every scheduler task materialized inside one `T.Stage(s)` receives stage `s`;
unwrapped siblings in the same child list default to stage 0. Within each
hardware pipe, issue order is the materialized source order; different pipes
remain freely reorderable. Do not pass `order=` or `stage=` to `T.Pipelined`
when its body contains `T.Stage`. Non-zero stages require `enable_offset=True`.
`T.Pipelined(..., num_stages=...)` bounds automatic buffer-version selection;
it does not bound frontend `T.Stage` values.

`T.Stage` must be the outer scope around complete scheduler tasks. Never place
it inside `T.Task`, `T.SimtVF`, `T.SimdVF`, an SBlock, or a parallel loop. With
multiple Python context managers, write `with T.Stage(1), T.SimtVF(...):`; the
reverse order nests `T.Stage` inside the SimtVF task and is invalid.

Reference: `examples/ascend/example_manual_schedule.py`

### 7.2 Legacy: Manual Flag Pipelining (Deprecated)

Manual `T.ascend_set_flag`/`T.ascend_wait_flag` pipelining and
`T.ascend_cross_core_set_flag`/`T.ascend_cross_core_wait_flag` cross-core
sync are **deprecated**. Use auto-schedule (Section 7.1) instead. The
low-level API functions still exist in the DSL for advanced users who
need full manual control, but example files demonstrating these patterns
have been removed from the tree.

### 7.3 Pipe Barrier

```python
T.ascend_pipe_barrier("PIPE_ALL")   # all pipes
T.ascend_pipe_barrier("PIPE_V")     # vector pipe
T.ascend_pipe_barrier("PIPE_MTE1")  # MTE1 pipe
```

---

## 8. SIMD MicroAPI (T.simd.*)

Low-level CCE vector intrinsics for `T.SimdVF()` blocks. These map
directly to Ascend CCE MicroAPI instructions (2048-bit vectors).

For detailed instruction semantics (lane widths, rounding modes, saturation
behavior, mask predicates), refer to the PTO Micro-Instruction Spec:
https://github.com/PTO-ISA/PTO-Gym/blob/main/docs/PTO-micro-Instruction-SPEC.md

### 8.1 Core Operations

```python
with T.SimdVF():
    mask = T.simd.pset(32)            # all lanes active for 32-bit elements

    # Load/Store
    r = T.simd.vld(buf[offset])       # 2048-bit vector load
    T.simd.vsts(buf[offset], r, mask) # 2048-bit vector store

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
    r = T.simd.vcvt(a, "float32")     # cast to float32
    r = T.simd.vcvt(a, "bfloat16")    # cast to bfloat16

    # Interleave/Deinterleave
    a0, a1 = T.simd.vintlv(x, y)      # interleave even/odd lanes
    x, y = T.simd.vdintlv(a0, a1)     # deinterleave back
```

`vld`, `vsts`, `pld`, and `pst` do not have a separate scalar `off`
parameter. Express the displacement in the buffer address, as shown above;
the Ascend code generator supplies the hardware-required zero offset. `vld2`
retains its `off` parameter because that operand has address-register semantics.

Full list of available SIMD intrinsics: `pld`, `pst`, `vld`,
`vsts`, `vsstb`, `vadd`, `vaddc`, `vsub`, `vmul`, `vdiv`,
`vmax`, `vmin`, `vand`, `vexp`, `vln`, `vsqrt`, `vrsqrt`, `vabs`, `vneg`,
`vrelu`, `vdup`, `vdupv`, `vcadd`, `vcmax`, `vcgadd`, `vintlv`, `vdintlv`,
`vpack`, `vgatherb`, `vcvt`, `vsel`, `vmuls`, `vadds`, `pset`, `mem_bar`.

### 8.2 SIMD Lower vs High-Level SIMD

Two levels of SIMD programming:

- **SIMD Lower (Recommended)**: Use raw `T.simd.*` intrinsics with
  manual loop unrolling (`for i in range(TILE // 64)`). This is the
  reliable approach -- you have full control over which instructions
  are emitted. See `examples/ascend/example_simdvf_vecadd_lower.py`.

- **High-level SIMD**: Use `T.Parallel` inside `T.SimdVF()`, letting the
  compiler auto-lower to vector operations. See
  `examples/ascend/example_simdvf_vecadd.py`.

  **Warning: High-level SIMD lowering is not fully reliable.** The
  `AscendSimdVFLowerParallel` pass may fail on non-trivial patterns
  beyond simple element-wise operations. When in doubt, use SIMD Lower
  with explicit `T.simd.*` calls.

### 8.3 VF Latency Annotation

Both `T.SimtVF` and `T.SimdVF` accept a `latency=` parameter (in cycles)
that feeds into the auto-schedule pass. The Z3 pipeline scheduler uses
this value to determine buffer lifetimes and orchestrate DMA/compute
overlap. Inaccurate latency estimates can lead to suboptimal pipelining
or correctness issues.

**Automatic measurement** with `measure_vf_latency.py`:

```bash
# Measure all VF blocks in a kernel and write latency= back to source
python measure_vf_latency.py examples/ascend/example_simdvf_vecadd_lower.py

# Preview without modifying the file
python measure_vf_latency.py examples/ascend/example_simdvf_vecadd_lower.py --dry-run
```

This tool compiles the kernel, extracts generated VF functions from
the `.asc` output, runs each in isolation under `cannsim` (Ascend
cycle-accurate simulator), and injects the measured cycle count as
`latency=<N>` back into the Python source file.

**Manual annotation**: You can also directly set `latency=` on the
scope if you already know the cycle count:

```python
with T.SimdVF(latency=128):
    ...

with T.SimtVF(threads=256, latency=512):
    ...
```

The auto-schedule pass will use the manually provided value instead of
estimating, bypassing the potentially inaccurate built-in estimator.

---

## 9. Programming Patterns (with Example References)

### Pattern 1: Simple GEMM with Auto-Schedule

The recommended modern pattern for basic GEMM.

```python
@T.prim_func
def main(X: T.Buffer((M, K), dtype),
         W: T.Buffer((N, K), dtype),
         C: T.Buffer((M, N), accum_dtype)):
    with T.Kernel(NUM_BLOCKS) as bx:
        x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
        w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
        res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
        temp = T.alloc_shared((TILE_M // 2, TILE_N), accum_dtype)

        for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
            m_tile, n_tile = compute_tile(tile_idx)
            for kt in T.Pipelined(K_TILES, num_stages=2):
                T.copy(X[m_slice, k_slice], x_l1)
                T.copy(W[n_slice, k_slice], w_l1)
                T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
            T.dual_copy(res, temp)
            T.dual_copy(temp, C[m_slice, n_slice])
```

Reference: `examples/ascend/example_gemm.py`

### Pattern 2: MixedKernel GEMM (Auto-Scheduled, sid Access)

Same as Pattern 1 but uses `T.MixedKernel` to get `sid` for AIV output
partitioning. No `T.Cube()`/`T.Vector()` scopes needed — the compiler
auto-splits AIC/AIV.

```python
with T.MixedKernel(NUM_BLOCKS) as (bx, sid):
    x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
    w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
    res = T.alloc_l0c((TILE_M, TILE_N), accum_dtype)
    temp = T.alloc_shared((TILE_M // 2, TILE_N), accum_dtype)

    for tile_idx in T.Persistent([NUM_TILES], NUM_BLOCKS, bx):
        for kt in T.Pipelined(K_TILES, num_stages=2):
            T.copy(X[...], x_l1)
            T.copy(W[...], w_l1)
            T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=(kt == 0))
        T.dual_copy(res, temp)
        T.copy(temp[sid * (TILE_M // 2):...], C[...])
```

Reference: `examples/ascend/example_gemm_mixedkernel.py`

### Pattern 3: SIMD VF (Register-Level Vector)

Element-wise operations using `T.SimdVF()`. No threads -- uses CCE
MicroAPI vector instructions directly.

```python
with T.Kernel(NUM_BLOCKS) as bx:
    for iter in T.Pipelined(NUM_TILES, num_stages=2):
        T.copy(A[begin:end], temp1)
        T.copy(B[begin:end], temp2)
        with T.SimdVF():
            for i in T.Parallel(TILE_ELEMS):
                temp3[i] = temp1[i] * (temp1[i] + temp2[i])
        T.copy(temp3, C[begin:end])
```

Reference: `examples/ascend/example_simdvf_vecadd_lower.py`

### Pattern 4: SIMT VF (Thread-Parallel Vector)

Element-wise operations distributed across threads.

```python
with T.SimtVF(threads=NUM_THREADS):
    for i in T.Parallel(TILE_ELEMS):
        temp3[i] = temp1[i] + temp2[i]
```

Reference: `examples/ascend/example_simtvf_vecadd.py`

### Pattern 5: RMSNorm with Fragment Trick

Uses `T.alloc_fragment` for register-level vectorized loads (float4
chunks) and `T.alloc_reducer` + `T.finalize_reducer` for cross-warp
reduction of `sum(x^2)`.

Reference: `examples/ascend/example_rmsnorm.py`

### Pattern 6: L0-Staged GEMM

Double-buffers L0A and L0B within each K-tile for maximum Cube
utilization. Adds an inner `SUB_K` tiling loop.

Reference: `examples/ascend/example_gemm_l0.py`

---

## 10. Common Pitfalls

1. **Forgetting `transpose_B=True`**: The single most common Ascend
   error. The Cube MAD unit requires it. GEMM without it fails at
   compile time or produces garbage.

2. **Using `threads=` on `T.Kernel`**: Raises `ValueError` immediately.
   Thread domains go inside `T.SimtVF(threads=N)` scopes.

3. **Multi-dimensional grids**: Ascend only supports `T.Kernel(N)`.
   For 2-D decomposition, compute mapping yourself: e.g.,
   `row = tile_idx // N_TILES; col = tile_idx % N_TILES`.

4. **Weight layout**: Because `transpose_B=True` is mandatory, store
   weight matrices as `[N, K]` (not `[K, N]` in global memory).

5. **Buffer dimension with auto-schedule**: With `T.Pipelined` or
   `T.Persistent`, buffers should **not** have a leading `STAGES`
   dimension. The compiler multi-buffers automatically.

6. **L0C is float32-only**: The cube accumulator only supports
   `"float32"` dtype. L1 and L0A/L0B can be `"bfloat16"` or `"float16"`,
   but L0C must be `"float32"`.

7. **Reducer usage**: `T.alloc_reducer` requires both `T.clear(reducer)`
   before accumulation and `T.finalize_reducer(reducer)` before reading
   results.

8. **SIMT VF thread sync**: The compiler auto-inserts `asc_syncthreads()`
   based on data-flow analysis. When writing to UB in one thread and
   reading in another, sync is automatic. No manual barriers needed.

9. **dual_copy shape mismatch**: `T.dual_copy` splits data across AIV
   sub-cores. The destination must be exactly half the source dimension
   (for M-split) or the compiler raises `ValueError`.

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

```python
import tilelang

# Standard compilation (auto-schedule enabled by default)
kernel = tilelang.compile(program, out_idx=-1)

# Skip the complete AutoSchedule path for manual-flag kernels
kernel = tilelang.compile(program, out_idx=-1,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})

# T.Stage automatically selects manual scheduling for its containing child list
kernel = tilelang.compile(staged_program, out_idx=-1)

# Multiple outputs
kernel = tilelang.compile(program, out_idx=[1, 3])

# Inspect generated Ascend C source
print(kernel.get_kernel_source())

# Run on NPU
x, w = torch.randn(...), torch.randn(...)
result = kernel(x, w)
torch.npu.synchronize()
```

Run scripts with: `npu-env python your_script.py`

### Post-Processing Callback

Register a callback to modify generated Ascend C source before
compilation:

```python
from tilelang.engine.callback import register_ascend_postproc

@register_ascend_postproc
def customize_source(code, target):
    # Modify 'code' (str), return modified source
    return code
```

Reference: `examples/ascend/example_ascend_postproc_callback.py`

---

## 12. Example Files Reference

All examples are in `examples/ascend/`. Key files organized by category:

### GEMM variants
| File | Description |
|---|---|
| `example_gemm.py` | **Recommended**: auto-scheduled GEMM with T.Persistent + T.Pipelined |
| `example_gemm_mixedkernel.py` | Auto-scheduled MixedKernel GEMM |
| `example_gemm_mix_manual.py` | Manual Cube+Vector mix with flags and cross-core sync (teaching reference) |
| `example_gemm_l0.py` | Auto-scheduled L0-staged GEMM with nested pipelining |
| `example_gemm_bypass_l2.py` | GEMM with L2 cache bypass |
| `example_gemm_various_shapes.py` | DeepGEMM-style auto-config with multiple tile shapes |
| `example_blockscaled_gemm.py` | Block-scaled GEMM (MXFP8) with per-group scale factors |

### SIMD VF (register-level vector operations)
| File | Description |
|---|---|
| `example_simdvf_vecadd.py` | High-level SIMD vector add using T.Parallel in SimdVF |
| `example_simdvf_vecadd_lower.py` | SIMD vector add using raw T.simd.* intrinsics |
| `example_simdvf_vintlv.py` | SIMD vintlv/vdintlv interleave operations |
| `example_simdvf_per_token_cast_to_fp8.py` | Per-token fp8 type conversion with SimdVF |
| `example_simdvf_topk_gate.py` | TopK-gate for MoE routing with SimdVF selection loop |

### SIMT VF (thread-parallel operations)
| File | Description |
|---|---|
| `example_simtvf_vecadd.py` | Auto-scheduled SIMT vector add |
| `example_simtvf_vecadd_mutex.py` | Mutex-based get_buf/rls_buf sync (teaching reference) |
| `example_simtvf_auto_sync.py` | Auto thread sync demonstration |
| `example_simtvf_ubuf_multi.py` | Multi-buffer UB with pipe barriers in SIMT |
| `example_simtvf_per_token_cast_to_fp8.py` | Per-token fp8 cast with SimtVF + reduce |
| `example_simtvf_vector_add.py` | Minimal SIMT vector add |

### Normalization
| File | Description |
|---|---|
| `example_rmsnorm.py` | Auto-scheduled RMSNorm with fragment trick + reducer |

### Attention
| File | Description |
|---|---|
| `example_flash_attn.py` | FlashAttention variant |

### Misc
| File | Description |
|---|---|
| `example_manual_schedule.py` | Frontend-staged vector add using manual AutoSchedule mode |
| `example_copy_pad_value.py` | Padded GM→UB copy (`T.copy(pad_value=)` / `data_select=`) for non-32B-aligned rows |
| `example_buffer_version_annotation.py` | Buffer version annotation for pipelining |
| `example_ascend_postproc_callback.py` | Post-processing codegen callback |

### Tests

Ascend unit tests live under `testing/ascend/`, not in `examples/`:
- `testing/ascend/analysis/` — compile-time pass/analysis checks (`test_ascend_*`),
  e.g. VF checker, dcache-bypass detection. No NPU execution.
- `testing/ascend/language/` — DSL primitive / codegen correctness (`test_tilelang_ascend_*`),
  e.g. copy, reduce, cast, print. Some execute on NPU.
- `testing/ascend/language/simtvf/` — SimtVF-specific semantics and checkers.

Each test file ends with `if __name__ == "__main__": tilelang.testing.main()` so it can
run standalone via `python <file>` or under `pytest`. Heavier kernel/integration tests that
reuse an `example_*.py` still live alongside them in `examples/ascend/`.
