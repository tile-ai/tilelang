# SIMT Architecture Concept Reference

> The core differences, hybrid programming and selection guide of SIMT vs SIMD have been covered in `npu-arch-guide.md` and will not be repeated here.

---

## SIMT thread hierarchy

### Overview

The AscendC SIMT programming model uses a three-level thread hierarchy:

| Hierarchy | Concept | Corresponding to CUDA | Setting method |
|------|------|----------|---------|
| Start core number | Block number | Block number in Grid | tiling side `SetBlockDim` |
| Number of startup threads | Number of single-core threads | Number of single-Block threads | Kernel side `constexpr` constant |
| Warp | 32 thread grouping | Warp | Hardware automatic grouping |

### Block Dim

- Set through the `SetBlockDim` interface on the tiling side
- Corresponds to the Block concept in CUDA
- Segmentation formula: `Total number of cores = ceil (total number of output elements / minimum number of elements processed by a single core)`

### Number of threads

- Single core supports maximum **2048** threads
- **Must be determined at compile time** (using `constexpr` or literal)
- Disable dynamic acquisition from tiling data
- `LAUNCH_BOUND(N)` and `Simt::Dim3(N)` must use the same compile-time constant
-Default value: 1024, the recommended range is adjusted according to the data volume and computational complexity

### Warp Scheduling

- Internally grouped into warps by **32 threads**
- Warp is the smallest unit of hardware scheduling
- Each AIV core contains 4 Warp Schedulers, and the scheduler number is `warp_id % 4`
- An AIV core only executes one thread block task at the same time
- Multiple Warps within the thread block are scheduled and executed sequentially

---

## Warp scheduling mechanism

### Basic concepts

- Each **32 threads** form a Warp
- Warp is the smallest unit of hardware scheduling
- Each thread in Warp is called Lane, numbered 0~31

### Hardware Scheduling

- Each AIV core contains **4 Warp Schedulers**, the scheduler number is `warp_id % 4`
- An AIV core only executes one thread block task at the same time
- Multiple Warps within the thread block are sequentially scheduled to the AI Core for execution

### SIMT execution semantics

- All threads in a Warp execute **the same instructions** (SIMT semantics)
- Execute each branch path serially during Branch Divergence
  - If some threads in the warp take the if branch, some of them take the else branch.
  - The hardware first executes the if path (the else thread waits), and then executes the else path (the if thread waits)
  - Causes pipeline bubbles and reduces effective utilization

### Branch divergence optimization

- Reduce branch differentiation within the same warp
- Promote runtime branches to compile time (`if constexpr` / template parameters)
- Threads in the same warp should try to take the same execution path

---

## SIMT memory space

### Overview

The AscendC SIMT programming model contains three types of memory spaces:

| Memory type | Description | Access scope | Address space modifier |
|---------|------|---------|--------------|
| Registers and stack | Independent per thread | Thread private | None (default) |
| Unified Buffer (UB) | Local memory | In-core sharing | `__ubuf__` |
| Global Memory (GM) | Global Memory | Accessible by all threads | `__gm__` |

### Registers and stack

- Each thread has its own set of registers
- The number of registers is affected by the number of threads (see `launch_bounds_registers.md`)
- Variables that exceed the register capacity will cause stack overflow

### Unified Buffer (UB)

- Local memory shared within the core
- Total size 256KB, used by partition (see `ub_partition.md` for details)
- SIMT operator cannot use the entire UB space and needs to reserve ≥32KB for DCache
- Available UB = 256KB - 8KB (reserved) - 32KB (DCache) = **216KB**
- Access via `__ubuf__` pointer in SIMT VF

### Global Memory (GM)

- Accessible by all cores and all threads
- SIMT VF supports direct reading and writing of GM without explicit Load/Store
- accessed via `__gm__` pointer
- Pass in the `GM_ADDR` parameter at the kernel entry

### Differences from SIMD

| Dimensions | SIMD | SIMT |
|------|------|------|
| Data transfer | Explicit Load/Store required | Supports direct reading and writing of GM and UB |
| GM access | Direct to register is not supported | Direct access is supported |

---

## UB memory partition

### Total amount

The total amount of Unified Buffer is **256KB**, partitioned sequentially from low address to high address:

| Area | Size | Description |
|------|------|------|
| Static memory | Determined at compile time | `__ubuf__` array declaration, used for SIMD/SIMT shared data in hybrid programming |
| Dynamic memory | tiling side `SetLocalMemory` setting | TBuf/LocalTensor application, specified by dynUBufSize of `<<<>>>` in hybrid programming |
| Reserved space | Fixed 8KB | Reserved by the compiler and cannot be used |
| Data Cache | = 256KB - Static - Dynamic - 8KB | SIMT Proprietary DCache |

### Key constraints

- **SIMT operator cannot use all UB space**, ≥32KB needs to be reserved for DCache
- If DCache < 32KB, compilation and verification will report an error
- Available UB = 256KB - 8KB - 32KB = **216KB**

### Tiling side settings

Define `DCACHE_SIZE` as 128KB, use `SetLocalMemorySize` to set the parameter to `ubsize - DCACHE_SIZE`:

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

### Static memory vs dynamic memory

| Type | Declaration method | Applicable scenarios |
|------|---------|---------|
| Static memory | `__ubuf__ T buffer[SIZE];` | Shared buffer of known size at compile time |
| Dynamic memory | `TBuf<QuePosition::VECCALC>` + `pipe_->InitBuffer()` | Allocate on demand at runtime |

---

## Data type quick check

### Classified by bit width

| Bit width | Data type |
|------|---------|
| b8 | bool, int8_t, uint8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t |
| b16 | int16_t, uint16_t, **half**, **bfloat16_t** |
| b32 | int32_t, uint32_t, **float**, complex32 |
| b64 | int64_t, uint64_t, double, complex64 |

### SIMT VF function parameter support types

#### Scalar type

bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, float, half, bfloat16_t

#### Pointer type

- `__gm__ T*` — Global Memory pointer
- `__ubuf__ T*` — Unified Buffer pointer

#### Return value

Must be `void`

### Special value macro

| Macro name | Description | Header file |
|------|------|--------|
| ASCRT_INF_BF16 | bfloat16 positive infinity | asc_bf16.h |
| ASCRT_MAX_NORMAL_BF16 | bfloat16 maximum value | asc_bf16.h |
| ASCRT_NAN_BF16 | bfloat16 NaN | asc_bf16.h |
| ASCRT_INF_F16 | half positive infinity | asc_fp16.h |
| ASCRT_MAX_NORMAL_F16 | half maximum value | asc_fp16.h |
| ASCRT_NAN_F16 | half NaN | asc_fp16.h |
| ASCRT_INF_F32 | float positive infinity | asc_simt.h |
| ASCRT_MAX_NORMAL_F32 | float maximum value | asc_simt.h |

---

## Function call level

### Hierarchy

```
Kernel function (__global__ __aicore__)
  ├── __aicore__ function
  ├── SIMD VF (__simd_vf__) ← Called by asc_vf_call
  │ └── __simd_callee__ sub-function
  └── SIMT VF (__simt_vf__) ← Called by VF_CALL
        └── __simt_callee__ sub-function
```

### Description of each level

#### Kernel function

```cpp
__global__ __aicore__ void {op_name}(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

- Operator entry, called by the framework
- Responsible for tiling data analysis and scene distribution
- Get tiling via `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA_WITH_STRUCT`

#### __aicore__ function

```cpp
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TilingData* tilingData)
```

- Main logic function in the core
- Responsible for UB buffer initialization and GM address conversion
- Call `Simt::VF_CALL` to start SIMT VF

#### SIMT VF function

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...)
```

- Thread-level parallel computing functions
- Each thread executes independently, traversing data at stride intervals
- Callable `__simt_callee__` sub-function and `constexpr` function

#### __simt_callee__ sub-function

```cpp
__simt_callee__ inline void HelperFunc(...)
```

- Auxiliary functions inside SIMT VF
- Must have `__simt_callee__` modifier
- Can be called by `__simt_vf__` function

### Call constraints

- Only the `__simt_callee__` function and the `constexpr` function can be called within `__simt_vf__`
- Only the `__simd_callee__` function and the `constexpr` function can be called within `__simd_vf__`
- Not callable across modes (SIMT VF cannot call SIMD callee and vice versa)

---

## LAUNCH_BOUND and the number of registers

### Register number mapping

`__launch_bounds__(N)` limits the maximum number of threads used by each VF Block. The number of threads directly affects the number of available registers per thread:

| Thread number range | Number of registers available per thread |
|-----------|-----------------|
| 1025~2048 | 16 |
| 513~1024 | 32 |
| 257~512 | 64 |
| 1~256 | 127 |

### Usage principles

- Registers are used to store thread local variables
- If the register capacity is exceeded, the stack will overflow, affecting performance.
- It is recommended that the N of `__launch_bounds__(N)` be consistent with the actual number of startup threads
- `LAUNCH_BOUND(N)` and `Simt::Dim3(N)` must use the same compile-time constant

### Declaration syntax

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void YourKernel(...);
```

- `LAUNCH_BOUND(thread_num)` optional, default 1024
- Parameter must be a `constexpr` constant or literal

### Thread number selection reference

| Operator type | Recommended number of threads | Reason |
|---------|-----------|------|
| Moving operators | 2048 / 1024 | Memory bandwidth is limited, more threads hide delays |
| Computational operators | 512 / 1024 | The register pressure is high, and parallelism and registers need to be balanced |

### Register pressure and thread number tuning

Register pressure increases with VF complexity:

| VF complexity | uint32_t number of index threads | uint64_t number of index threads |
|-----------|-------------------|-------------------|
| Extremely low (NONE/1D) | 1024 | 512 |
| Medium Low (2D) | 1024 | 512 |
| Medium (3D) | 1024 | 512 |
| Higher (4D) | 1024 | 512 |
| Highest (ND runtime loop) | 256 | 128 |
