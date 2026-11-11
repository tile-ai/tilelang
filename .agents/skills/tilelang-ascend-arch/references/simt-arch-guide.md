# SIMT Architecture Concepts Reference

> Core SIMT vs SIMD differences, mixed programming, and selection guidance are covered in `npu-arch-guide.md` and not repeated here. SIMT is available only on DAV_3510 (Ascend950PR/950DT).

---

## SIMT Thread Hierarchy

### Overview

The AscendC SIMT programming model uses a three-level thread hierarchy:

| Level | Concept | CUDA equivalent | Configuration |
|-------|---------|-----------------|----------------|
| Launched core count | Number of blocks | Block count in a grid | Tiling-side `SetBlockDim` |
| Launched thread count | Threads per core | Threads per block | Kernel-side `constexpr` constant |
| Warp | Groups of 32 threads | Warp | Grouped automatically by hardware |

### Core Count (Block Dim)

- Set through the tiling-side `SetBlockDim` interface
- Corresponds to the CUDA block concept
- Partitioning formula: `total cores = ceil(total output elements / minimum elements per core)`

### Thread Count

- **Must be determined at compile time** (via `constexpr` or literals)
- Must not be derived dynamically from tiling data
- `LAUNCH_BOUND(N)` and `Simt::Dim3(N)` must use the same compile-time constant
- Default: 1024; tune based on data volume and compute complexity

### Warp Scheduling

- Threads are grouped into warps of **32 threads**
- A warp is the smallest hardware scheduling unit
- Each AIV has 4 warp schedulers; the scheduler index is `warp_id % 4`
- An AIV executes only one thread-block task at a time
- Warps within a thread block are scheduled in turn

---

## Warp Scheduling Mechanics

### Basics

- Every **32 threads** form a warp
- A warp is the smallest hardware scheduling unit
- Each thread in a warp is called a lane, numbered 0~31

### Hardware Scheduling

- Each AIV has **4 warp schedulers**; the scheduler index is `warp_id % 4`
- An AIV executes only one thread-block task at a time
- Warps within a thread block are scheduled onto the AI Core in turn

### SIMT Execution Semantics

- All threads in a warp execute the **same instruction** (SIMT semantics)
- Under branch divergence, the branch paths execute serially:
  - Some threads in a warp take the if path while others take the else path
  - Hardware runs the if path first (else threads wait), then the else path (if threads wait)
  - This creates pipeline bubbles and reduces effective utilization

### Branch Divergence Optimization

- Minimize divergence within a warp
- Hoist runtime branches to compile time (`if constexpr` / template parameters)
- Keep threads in a warp on the same execution path where possible

---

## SIMT Memory Spaces

### Overview

The AscendC SIMT programming model has three memory spaces:

| Memory type | Description | Access scope | Address-space qualifier |
|-------------|-------------|--------------|-------------------------|
| Registers and stack | Per-thread | Thread-private | None (default) |
| Unified Buffer (UB) | Local memory | Shared within the core | `__ubuf__` |
| Global Memory (GM) | Global memory | Accessible to all threads | `__gm__` |

### Registers and Stack

- Each thread owns a private set of registers
- The register count depends on the thread count (see §LAUNCH_BOUND and Register Count)
- Variables beyond register capacity spill to the stack

### Unified Buffer (UB)

- Local memory shared within the core
- 256 KB total, used in partitions (see §UB Memory Partitioning)
- SIMT operators cannot use the whole UB: reserve ≥32 KB for the DCache
- Usable UB = 256 KB − 8 KB (reserved) − 32 KB (DCache) = **216 KB**
- Accessed in SIMT VF through `__ubuf__` pointers

### Global Memory (GM)

- Accessible to all threads on all cores
- SIMT VF supports reading/writing GM directly, without explicit Load/Store
- Accessed through `__gm__` pointers
- Passed in at the kernel entry via `GM_ADDR` parameters

### Differences from SIMD

| Dimension | SIMD | SIMT |
|-----------|------|------|
| Data movement | Explicit Load/Store required | Direct GM and UB reads/writes supported |
| GM access | No direct access to registers | Direct access supported |

---

## UB Memory Partitioning

### Total Size

The Unified Buffer totals **256 KB**, partitioned from low to high addresses:

| Region | Size | Notes |
|--------|------|-------|
| Static memory | Fixed at compile time | `__ubuf__` array declarations; used for SIMD/SIMT shared data in mixed programming |
| Dynamic memory | Set on the tiling side | TBuf/LocalTensor allocation; specified via the dynUBufSize of `<<<>>>` in mixed programming |
| Reserved | Fixed 8 KB | Reserved by the compiler; unusable |
| Data Cache | = 256 KB − static − dynamic − 8 KB | SIMT-only DCache |

### Key Constraints

- **SIMT operators cannot use the whole UB**; reserve ≥32 KB for the DCache
- A DCache below 32 KB fails compile-time validation
- Usable UB = 256 KB − 8 KB − 32 KB = **216 KB**

### Tiling-Side Configuration

Reserve space for the DCache: set the dynamic UB upper bound to `ubsize - DCACHE_SIZE` (e.g., a 128 KB DCache reservation) through the tiling context's dynamic-UB interface (`SetLocalMemorySize`, renamed `SetDynUBufSize` in newer versions; see the signature in the CANN package's `tiling_context.h`).

### Static vs Dynamic Memory

| Type | Declaration | Use case |
|------|-------------|----------|
| Static memory | `__ubuf__ T buffer[SIZE];` | Shared buffers whose size is known at compile time |
| Dynamic memory | `TBuf<QuePosition::VECCALC>` + `pipe_->InitBuffer()` | Allocated on demand at runtime |

---

## Data Types

### SIMT VF Function Parameter Types

#### Scalar Types

bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, half, bfloat16_t

#### Pointer Types

- `__gm__ T*` — Global Memory pointer
- `__ubuf__ T*` — Unified Buffer pointer

#### Return Value

Must be `void`

### Special-Value Macros

| Macro | Description | Header |
|-------|-------------|--------|
| ASCRT_INF_F32 | float positive infinity | asc_simt.h |
| ASCRT_MAX_NORMAL_F32 | float maximum normal value | asc_simt.h |

> Note: ASCRT_*_F32 macros were not located in `asc_simt.h` of the current CANN 9.3.0 package; verify against the actual header before use.

---

## Function Call Hierarchy

### Hierarchy

```
Kernel function (__global__ __aicore__)
  ├── __aicore__ functions
  ├── SIMD VF (__simd_vf__) ← invoked via asc_vf_call
  │     └── __simd_callee__ sub-functions
  └── SIMT VF (__simt_vf__) ← invoked via VF_CALL
        └── __simt_callee__ sub-functions
```

### Level Descriptions

#### Kernel Function

```cpp
__global__ __aicore__ void {op_name}(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

- Operator entry point, invoked by the framework
- Handles tiling-data parsing and scenario dispatch
- Obtains tiling via `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`

#### __aicore__ Functions

```cpp
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TilingData* tilingData)
```

- Core main-logic function
- Handles UB buffer initialization and GM address conversion
- Launches the SIMT VF via `Simt::VF_CALL`

#### SIMT VF Functions

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...)
```

- Thread-level parallel compute function
- Each thread executes independently, traversing data at a stride
- May call `__simt_callee__` sub-functions and `constexpr` functions

#### __simt_callee__ Sub-Functions

```cpp
__simt_callee__ inline void HelperFunc(...)
```

- Helper functions inside a SIMT VF
- Must carry the `__simt_callee__` qualifier
- Callable from `__simt_vf__` functions

### Call Constraints

- Inside `__simt_vf__`, only `__simt_callee__` and `constexpr` functions may be called
- Inside `__simd_vf__`, only `__simd_callee__` and `constexpr` functions may be called
- No cross-mode calls (a SIMT VF cannot call SIMD callees, and vice versa)

---

## LAUNCH_BOUND and Register Count

### Register Count Mapping

`__launch_bounds__(N)` caps the thread count of a VF block; the thread count directly determines the per-thread register count:

| Thread range | Registers per thread |
|--------------|----------------------|
| 1025~2048 | 16 |
| 513~1024 | 32 |
| 257~512 | 64 |
| 1~256 | 127 |

### Usage Principles

- Registers hold thread-local variables
- Exceeding register capacity spills to the stack and hurts performance
- Keep the N in `__launch_bounds__(N)` equal to the actual launched thread count
- `LAUNCH_BOUND(N)` and `Simt::Dim3(N)` must use the same compile-time constant

### Declaration Syntax

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void YourKernel(...);
```

- `LAUNCH_BOUND(thread_num)` is optional; default 1024
- The argument must be a `constexpr` constant or a literal

### Thread Count Selection Guide

| Operator type | Suggested threads | Reason |
|---------------|-------------------|--------|
| Memory-movement operators | 2048 / 1024 | Memory-bandwidth bound; more threads hide latency |
| Compute operators | 512 / 1024 | Higher register pressure; balance parallelism against registers |

### Register Pressure and Thread Count Tuning

Register pressure grows with VF complexity:

| VF complexity | Threads (uint32_t indexing) | Threads (uint64_t indexing) |
|---------------|------------------------------|-----------------------------|
| Very low (NONE/1D) | 1024 | 512 |
| Low (2D) | 1024 | 512 |
| Medium (3D) | 1024 | 512 |
| High (4D) | 1024 | 512 |
| Highest (ND runtime loops) | 256 | 128 |
