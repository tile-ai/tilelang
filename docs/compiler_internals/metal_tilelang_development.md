# TileLang Metal Backend Internals

This document explains how the TileLang Metal backend lowers a TileLang kernel
to Metal shader source, how the current GEMM fast path is implemented, which
backend work does not map one-to-one to CUDA, and where the next optimization
work should land.

This is not an Apple GPU architecture overview or a TileLang frontend tutorial.
Metal and Apple GPU background is introduced only where it affects lowering,
codegen, runtime behavior, or performance policy. Reference material is grouped
in Section 5.

How to read this document:

- To understand the current fast path, read Section 0 and Section 1.
- To work on the Metal backend or codegen, read Sections 1, 2, and 3, then use
  Section 5 as reference.
- To write TileLang kernels for Metal, read Section 0 and Section 4.
- To find performance data, implementation files, or test commands, go to
  Section 5.

## 0. TL;DR

- The current default fast path is **direct-global cooperative tensor**: A/B are
  loaded from `device` memory without `threadgroup` staging, C accumulates in a
  cooperative tensor destination, and the result is stored back to global C.
- Default benchmark configuration: `ct_global`, `block_M=64`, `block_N=128`,
  `threads=128` (4 simdgroups), fp16 input, fp16 output, fp32 accumulation.
- On the current full-tile fp16 GEMM snapshot, TileLang is at or slightly above
  MLX and about 5-6% behind PyTorch MPS. See Section 5.6.
- The most important mental model: **Metal is not CUDA**. On Apple unified
  memory, `gmem -> smem` is not a performance hint. Shared staging should be
  preserved when it carries real semantics such as layout, padding, or reuse.
- The other paths, shared cooperative tensor and simdgroup, still exist for
  compatibility and semantic coverage. They are not the current performance
  mainline. See Section 3.

## 1. The Fast Path, End to End

This section traces one concrete direct-global cooperative tensor kernel from
TileLang source to generated MSL and then to the performance snapshot. Later
sections explain the mechanisms behind this path.

### 1.1 The Kernel

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    T.gemm(
        A[by * block_M:(by + 1) * block_M, 0:K],
        B[0:K, bx * block_N:(bx + 1) * block_N],
        C[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N],
        clear_accum=True,
    )
```

A, B, and C are all tile views over `global` tensors. There is no
`alloc_shared`, no `T.copy` staging, and no handwritten K-loop. For full-tile
GEMM, the backend expands the K reduction during lowering. This is the intended
frontend shape: express operand regions and dtypes, and let the backend own lane
mapping, operand construction, and accumulation.

### 1.2 Path Selection

When the backend sees this `T.gemm`, the instruction selection and Metal GEMM
lowering logic pick the direct-global cooperative tensor path. The detailed
selection rules are in Section 2.1. For this kernel, A/B are `global`, C is not
shared, and the tile shape and thread count allow cooperative tensor lowering.

For `M=128, N=256, K=128, block_M=64, block_N=128, threads=128`:

- `threads=128` means 4 simdgroups.
- The 4 simdgroups cover a `64 x 128` C tile as a `2 x 2` partition.
- Each simdgroup owns a `32 x 64` C subtile.
- Each `32 x 64` subtile is covered by `(32 / 16) x (64 / 32) = 4` op-level
  fragments, corresponding to 4 destination cooperative tensors.

### 1.3 The MPP op: `matmul2d(M, N, K)`

The core operation is an MPP tensor op. The generated MSL contains:

```cpp
constexpr auto __pct_desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, /*trans_a=*/false, /*trans_b=*/false, /*accumulate=*/true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<__pct_desc, metal::execution_simdgroup> __pct_op;
```

In this document, `matmul2d(M, N, K)` refers to the first three shape parameters
of the descriptor, in M / N / K order. A single MPP op produces an `M x N`
destination fragment and reduces along K. These parameters are not TileLang
`block_M` / `block_N`, and they are not the whole threadgroup tile shape. The
current fast path specializes to `matmul2d(16, 32, 16)`. Larger simdgroup and
threadgroup tiles are covered by multiple op-level fragments, such as the 4
destination cooperative tensors above.

### 1.4 The Generated MSL

The following is a representative snippet for the shape above. Repeated
fragments are omitted for readability. It is generated with
`accum_dtype=T.float16`, so C is `device half*` and the final store contains a
`half4(...)` cast. Existing codegen tests use `accum_dtype=T.float32` by default,
which generates `device float* C` and no fp32-to-fp16 store cast. To reproduce
this fp16-output source exactly, pass `accum_dtype=T.float16`; see Section 5.8.

Kernel signature and execution attributes:

```cpp
[[kernel, max_total_threads_per_threadgroup(128)]] void main_kernel(
    const device half* __restrict A [[ buffer(0) ]],
    const device half* __restrict B [[ buffer(1) ]],
    device half* __restrict C [[ buffer(2) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 __gridDim [[threadgroups_per_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]],
    uint __simd_group_id [[simdgroup_index_in_threadgroup]]) {
  const ushort __lane = __metal_get_thread_index_in_simdgroup(ushort());
  ...
```

Destination cooperative tensors are created and zeroed before the K-loop. C
storage is elided: C does not round-trip through a `thread` array and instead
persists in cooperative tensor destination objects.

```cpp
auto __pct_c0 = __pct_op.get_destination_cooperative_tensor<...>();
// __pct_c1 / __pct_c2 / __pct_c3 are created the same way.
TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < 16; __i++) __pct_c0[__i] = 0.0f;
```

Inside the K-loop, A/B are loaded from `device` memory into thread-private
temporary storage and then moved into MPP cooperative tensor operands. There is
no `threadgroup` allocation and no barrier.

```cpp
for (int k_outer = 0; k_outer < 8; ++k_outer) {
  const device half* __src =
      (const device half*)(&(A[((blockIdx.y * 8192) +
                                ((__simd_group_id & 1) * 4096)) +
                               (k_outer * 16)]));
  // half4 vector loads into A_local ...
  // B loads follow the same pattern ...
  auto __ct_a = __pct_op.get_left_input_cooperative_tensor<half, half, float>();
  auto __ct_b = __pct_op.get_right_input_cooperative_tensor<half, half, float>();
  // A_local / B_local -> __ct_a / __ct_b
  __pct_op.run(__ct_a, __ct_b, __pct_c0);  // fp32 accumulation
}
```

After the K-loop, C is stored. For fp16 output, the fp32 destination values are
cast on store:

```cpp
device half* __dst = (device half*)(&(C[...]));
*(device half4*)(&__dst[__r0 * 256 + __c0]) =
    half4(*(thread float4*)(&__pct_c0[0]));
```

Important source-shape properties:

- A/B are `const device half* __restrict`; C is `device half* __restrict`. This
  preserves read-only and noalias information for the Metal compiler.
- simdgroup id and lane id come from Metal execution attributes
  (`__simd_group_id`, `__metal_get_thread_index_in_simdgroup`) instead of being
  repeatedly derived from `threadIdx`.
- Addresses have the shape `base + tile offset + simdgroup offset + K-loop
  induction`.
- `mpp::...matmul2d` destinations accumulate in fp32; fp16 output is handled at
  store.
- There is no `threadgroup half` declaration and no barrier.

### 1.5 Why This Is Not a CUDA Translation

This path is not a step-by-step translation of the common CUDA
`global -> shared -> ldmatrix -> mma` pipeline:

```text
CUDA:  device A/B -> shared -> ldmatrix -> mma -> shared/reg -> device C
Metal: device A/B ------------> MPP operand -> matmul2d -> destination CT -> device C
```

The Metal backend chooses a source shape that matches MPP directly. On Apple
unified memory, moving A/B through `threadgroup` before feeding the tensor op is
usually an extra copy and barrier when it carries no semantic value. This is why
the shared path in Section 3 is correct but slower for the current full-tile
GEMM snapshot.

## 2. How the Backend Gets There

This section explains how the backend gets from `T.gemm` to the MSL shown in
Section 1. It only covers mechanisms needed for that mainline path.

### 2.1 Path Selection

Path selection has two stages.

**Stage 1: instruction kind (`SelectInst`, `src/metal/op/gemm.cc`).** This stage
only chooses between `metal.cooperative_tensor` and `metal.simdgroup`. It looks
at C scope, tile shape, warp policy, and warp count. It does not look at whether
A/B are global or shared, and it does not check runtime or hardware capability.

| Condition | Instruction kind |
| --- | --- |
| C is in `local.fragment` or `metal.simdgroup` | `metal.simdgroup` |
| Otherwise `CanUseCooperativeTensor(policy, M, N, K, warps)` is true | `metal.cooperative_tensor` |
| Otherwise | `metal.simdgroup` as a safe lowering fallback |

`CanUseCooperativeTensor` is a pure shape and policy check: `M % 16 == 0`,
`N % 32 == 0`, `K % 16 == 0`, and the warp policy must be able to split
`num_warps` into a legal `m_warp x n_warp` partition. `threads` must be a
multiple of 32 because each simdgroup has 32 lanes.

**Stage 2: cooperative tensor dataflow (`GemmMetal.lower`,
`tilelang/metal/op/gemm/gemm_metal.py`).** After Stage 1 selects
`metal.cooperative_tensor`, the Python lowering chooses the concrete dataflow
from A/B scope:

| A/B scope | Dataflow | Notes |
| --- | --- | --- |
| A/B are `global` (`is_gemm_gg`) | direct-global cooperative tensor | Current fast path, Section 1 |
| A/B are `shared` (`is_gemm_ss`) | shared cooperative tensor | Preserves shared semantics, Section 3.1 |

C in cooperative tensor or fragment form uses destination storage elision. C in
shared scope uses writeback. Warp partitioning prefers a balanced grid of
16x32 op-level fragments per simdgroup.

One current capability boundary is important: all of the above choices happen
at **lowering time** based on shape, scope, and policy. They are not runtime or
hardware capability checks. The runtime guard in Section 5.1 is separate: it
only chooses which MSL language version to request when compiling the generated
source. It cannot rewrite already generated MPP source back to simdgroup. If
`SelectInst` chose cooperative tensor, the generated source is MPP source.
Instruction-level runtime capability fallback still needs a separate design.

### 2.2 Surrounding Lowering

The launch, allocation, loop, copy, and swizzle lowering around GEMM determine
what source shape the intrinsic sequence sits in. For the direct-global path:

- **Launch**: `T.Kernel(..., threads=N)` lowers to a Metal threadgroup launch,
  emitting `max_total_threads_per_threadgroup(N)`, `blockIdx` / `threadIdx`
  style indices, simdgroup id, and lane id. Matrix work is still partitioned by
  32-lane simdgroups; there is no CUDA WGMMA-style 128-thread warpgroup.
- **Allocation**: the direct-global path does not allocate `threadgroup`
  buffers. A/B use `thread` temporary arrays. C persists in
  `metal.cooperative_tensor` destination objects when storage elision applies.
  The full scope-to-object mapping is in Section 5.2.
- **Loop**: the K-loop generates affine addresses of the form
  `base + tile offset + simdgroup offset + K-loop induction`. It does not rely
  on CUDA-style software pipelining or make shared staging the inner-loop
  policy. Pointer induction and address hoisting remain optimization targets
  (Section 5.5).
- **Copy / sync**: the direct-global path is device load/store plus MPP operand
  construction. With no shared dataflow, it avoids threadgroup copy and barrier
  operations.
- **Swizzle**: block-level swizzle is supported. `T.use_swizzle(panel_size,
  order="mlx")` maps to `rasterization2DMLX`, canonicalizing physical block
  index to logical tile index and avoiding repeated physical-index arithmetic in
  generated MSL. This is block/grid-level swizzle, not CUDA-style thread-level
  or per-thread swizzle.

### 2.3 Metal-Specific Work (No CUDA Counterpart)

The direct-global path includes backend work that does not have a direct CUDA
counterpart:

- **Direct-global cooperative tensor lowering**: A/B become MPP operands from
  `device` memory without explicit shared staging.
- **Cooperative tensor destination management**: C accumulates in MPP
  destination objects; small C cooperative tensor storage can elide a
  thread-local array (Section 1.4).
- **fp16 output / fp32 accumulation split**: MPP destinations use fp32, and the
  store to fp16 C performs the cast.
- **MLX-style block rasterization**: physical block indices are canonicalized to
  logical tile indices.
- **simdgroup id / lane id lowering**: Metal execution attributes are used
  instead of re-deriving them from thread id.
- **Metal 4 runtime language-version guard**: MSL 4.0 is requested only when the
  SDK, runtime OS, and device family support it.

This work is closer to Metal backend canonicalization and source-shape
construction than to mechanical CUDA op lowering.

## 3. The Other Paths

Besides the fast path, the backend keeps two additional paths. They are correct
and tested, but they are not the current performance mainline.

### 3.1 Shared Cooperative Tensor

If the frontend IR already contains shared staging, the backend lowers it as a
shared cooperative tensor path:

```text
global A/B -> threadgroup A/B -> cooperative tensor operand -> MPP -> C
```

This preserves the shape of CUDA-style kernels and is appropriate when shared
memory carries real semantics: layout transforms, padding or edge zero-fill,
multi-consumer reuse, threadgroup communication, or address patterns that the
direct-global path cannot currently express.

If shared memory is only staging, it is an extra copy plus barrier on Metal. In
the current measurements, it is slower than the direct-global path.

### 3.2 Simdgroup Fallback

The simdgroup path lowers GEMM to simdgroup matrix intrinsics:

```text
threadgroup A/B -> simdgroup_load -> simdgroup_multiply_accumulate -> simdgroup_store
```

It is a compatibility path used when C is in `local.fragment` or
`metal.simdgroup`, or when lowering-time cooperative tensor conditions are not
met. Compared with cooperative tensor, it behaves more like an explicit
fragment backend: the backend manages fragment layout, load/store, and
accumulation/store policy. It is therefore not the M5 / Metal 4 optimization
mainline.

### 3.3 Why These Paths Are Kept

- Existing TileLang kernels may already use CUDA-style staging.
- Shared staging may carry layout transform, padding, edge zero-fill, or
  multi-consumer reuse semantics.
- Some non-contiguous or edge-heavy workloads cannot yet be expressed by the
  direct-global path.
- Older Apple GPUs and non-Metal-4 runtimes need a complete simdgroup fallback
  story, although instruction-level runtime fallback is not implemented yet.

The key requirement is conservatism: a shared buffer can be treated as pure
staging, or bypassed by a future pass, only when the compiler can prove it has
no extra semantics. If that proof fails, the shared path must be preserved.

## 4. Contracts and Guidance

### 4.1 Frontend Contract

For the Metal backend, the frontend should express stable program semantics:

- memory scope;
- GEMM operand region;
- block-level swizzle;
- dtype and output dtype;
- required boundary conditions.

These fields should describe the program, not prescribe the final MSL source
shape. Lane mapping, operand construction, register tile shape, and instruction
selection belong to the backend and codegen. The current Metal control surface
is summarized in the concept map in Section 5.1.

### 4.2 Bad CUDA Assumptions

Do not carry CUDA GEMM assumptions directly into Metal:

- Do not treat `gmem -> smem` as a performance hint. Its semantics and bypass
  conditions are described in Section 3.
- Do not add shared staging just to mimic a CUDA tensor-core pipeline.
- Do not treat `local` / rmem staging as free. It can compete with threadgroup
  memory for the same class of on-chip storage budget and can increase register
  pressure or spilling (Section 5.2).
- Do not expose CUDA warpgroup, thread-level layout, or per-thread swizzle as
  Metal frontend semantics.

### 4.3 Lowering and Codegen Rules

Metal lowering should keep path boundaries clear:

- maintain the direct-global cooperative tensor fast path;
- preserve shared path semantics;
- do not mix simdgroup and cooperative tensor policy into one path;
- report unsupported shapes clearly or use a safe fallback.

Metal codegen should focus on MSL source shape:

- emit simple affine addresses;
- use explicit simdgroup id and lane id;
- preserve `const` for read-only buffers and `__restrict` for noalias buffers;
- avoid repeated physical swizzle arithmetic;
- preserve the fp32 accumulation to fp16 output store-cast path.

## 5. Reference

### 5.1 CUDA / Metal Concept Map and Capability

| CUDA / NVIDIA concept | Metal concept | TileLang Metal mapping | Notes |
| --- | --- | --- | --- |
| Grid | Grid of threadgroups | `T.Kernel` grid dims | Mostly direct mapping |
| CTA / thread block | Threadgroup | `T.Kernel(..., threads=N)` | Matrix execution is still split by simdgroup |
| Thread | Thread | thread index lowering | Mostly direct mapping |
| Warp | Simdgroup | simdgroup id / lane id | Current backend assumes 32 lanes |
| Warpgroup (128-thread WGMMA) | No direct exposed equivalent | Not exposed | Cooperative tensor is organized around simdgroup execution context |
| Shared memory | Threadgroup memory | `shared` scope | Semantic scope; performance policy is discussed in Section 3 |
| Register / fragment | thread storage / simdgroup matrix / cooperative tensor destination | `local` / `metal.simdgroup` / `metal.cooperative_tensor` | Do not reuse CUDA fragment assumptions directly |
| Tensor Core MMA | simdgroup matrix or MPP tensor op | `metal.simdgroup` / `metal.cooperative_tensor` | M5+ prefers cooperative tensor |
| `ldmatrix` / shared-to-MMA | cooperative tensor load / MPP operand | codegen intrinsic emission | Direct-global can bypass explicit shared staging |
| Thread-level layout / per-thread swizzle | No exposed equivalent | Not user-controlled | No CUDA-style per-thread data layout or register swizzle control surface |
| Block swizzle | Rasterization | `T.use_swizzle(..., order="mlx")` | Block-level swizzle |

The key difference is that the Metal backend currently does not expose
CUDA-style thread-level layout or per-thread swizzle. It exposes block/grid-level
rasterization swizzle; lane mapping is decided by the backend and the Metal
compiler, not by stable frontend semantics.

The runtime guard in `metal_module.mm` has one job: avoid requesting MSL 4.0
unconditionally. It selects `MTLLanguageVersion4_0` only when the SDK, runtime
OS, and device family support Metal 4; otherwise it compiles with
`MTLLanguageVersion2_3`. This is not an instruction-level fallback. It cannot
rewrite generated MPP source into simdgroup source. Cooperative tensor vs.
simdgroup is a lowering-time shape/scope/policy decision (Section 2.1), separate
from runtime language-version selection. A complete runtime capability fallback
still needs a separate design.

### 5.2 Memory Scopes and Unified Memory

| TileLang scope | Metal address space / object | Purpose |
| --- | --- | --- |
| `global` | `device` | Input and output tensors |
| `shared` | `threadgroup` | Threadgroup scratchpad / staging; barrier required for cross-thread visibility |
| `local` | `thread` private array | Per-thread temporary array; may occupy registers or spill |
| `local.var` | `thread` scalar | Scalar index or loop state |
| `metal.simdgroup` | `simdgroup_*8x8` object | simdgroup matrix fragment |
| `metal.cooperative_tensor` | MPP operand / destination object | Cooperative tensor path; shape comes from the descriptor and is not fixed 8x8 |

Apple platforms use unified memory, but that does not erase MSL address spaces
or make CUDA memory policy carry over unchanged. One migration assumption to
avoid is physical rmem / smem separation. `thread` storage and `threadgroup`
memory are distinct logical address spaces, but physical placement and spilling
are managed by the Metal compiler and Apple GPU runtime. For TileLang Metal
lowering, treat `local` private arrays and `shared` threadgroup staging as
competing for the same class of on-chip storage budget, rather than assuming a
CUDA-like model where the register file is free and shared memory is budgeted
separately.

### 5.3 Execution Model and Simdgroup Builtins

Execution hierarchy: grid (threadgroup grid), threadgroup (closest to a CUDA
CTA), thread, and simdgroup (32-lane SIMD execution unit). Threadgroup size and
matrix execution group are distinct: TileLang Metal can launch 128 or 256
threads, while cooperative tensor and simdgroup matrix execution are organized
around 32-lane simdgroups. There is no exposed 128-thread warpgroup equivalent.

Supported simdgroup matrix builtins include `metal.simdgroup` scope,
`make_filled_simdgroup_matrix`, `simdgroup_load`,
`simdgroup_multiply_accumulate`, and `simdgroup_store`. `simdgroup_*8x8` is the
simdgroup matrix object shape; it is distinct from the cooperative tensor MPP
descriptor shape described in Section 1.3.

### 5.4 Implemented Features

| Feature | Status | Notes |
| --- | --- | --- |
| Metal 4 runtime guard | Implemented | Requests MSL 4.0 only when SDK, OS, and device family support it |
| Cooperative tensor GEMM lowering | Implemented | M5+ mainline |
| Direct-global A/B cooperative tensor | Implemented / default | Current recommended performance path, Section 1 |
| Shared A/B cooperative tensor | Implemented | Compatible with CUDA-style staging, Section 3.1 |
| fp16 output handling | Implemented | fp32 accumulation plus store cast, Section 1.4 |
| C cooperative tensor storage elision | Implemented | Reduces thread-local C round-trip |
| MLX-style block swizzle | Implemented | Block-level rasterization |
| simdgroup GEMM | Implemented | Compatibility path, Section 3.2 |
| simdgroup id / lane id lowering | Implemented | Also used by cooperative tensor kernels |
| `const` / `__restrict` parameter emission | Implemented | Improves MSL alias information |

### 5.5 Known Limitations and Roadmap

Current limitations:

- The direct-global path mainly covers full-tile GEMM.
- Edge masking and partial tile support are not complete.
- The `gmem -> smem` bypass pass is not implemented.
- NAX / MLX-style 8-simdgroup lowering is not implemented.
- Pointer induction and address hoisting still have room for improvement.
- CUDA-style thread-level layout and per-thread swizzle control are not exposed
  (Section 5.1).
- The simdgroup path is not yet a complete old-device strategy.
- The primary validated path is fp16 input, fp16 output, fp32 accumulation.
- There is no instruction-level runtime capability fallback. Path selection is
  a lowering-time shape/scope/policy decision, and the runtime guard only
  selects the MSL language version (Sections 2.1 and 5.1).

Planned work:

- **Shared-staging bypass pass**: detect pure staging of the form
  `global A/B -> shared A/B -> T.gemm`, and rewrite it to the direct-global path
  when semantics are provably unchanged. Mathematically, this is conditional
  substitution:

  ```text
  S[s] = G[phi(s)]
  T.gemm(... S[psi(t)] ...)  =>  T.gemm(... G[phi(psi(t))] ...)
  ```

  The proof must rule out layout transforms, padding, edge fill,
  multi-consumer reuse, and threadgroup communication semantics. If the proof
  fails, the shared path must be preserved.
- **NAX / MLX-style 8-simdgroup cooperative tensor lowering**: `BM=64`,
  `BN=128`, 8 simdgroups / 256 threads, each simdgroup owning a `32 x 32` C
  tile covered by multiple `matmul2d(16, 32, 16)` op-level fragments.
- **Pointer induction and address hoisting**: hoist A/B/C tile base pointers,
  simdgroup offsets, and K-loop pointer increments to reduce repeated affine
  expressions in the hot path.
- **Edge masking and partial tile support**: predicated A/B load or zero-fill,
  predicated C store, out-of-bounds return for padded physical grids after
  swizzle, and branch-free fast paths for aligned cases.
- **Metal-specific policy and tuner**: distinguish at least four policy classes:
  4-simdgroup conservative direct-global, 8-simdgroup NAX / MLX-style,
  shared path for layout / fusion / edge-heavy workloads, and simdgroup
  fallback.

### 5.6 Current Performance Snapshot

These numbers are a snapshot of the current development environment and current
commit. They should not be generalized to all Apple GPUs, OS versions, SDKs, or
library versions. Reproduction commands are in Section 5.8.

| Item | Value |
| --- | --- |
| Date | 2026-06-20 |
| Machine | MacBook Air, Mac17,3 |
| Chip | Apple M5 |
| GPU | Apple M5, 10-core GPU |
| Metal support | Metal 4 |
| macOS | 26.5.1 |
| Xcode | 26.4.1 |
| Python | 3.12.13 |
| PyTorch | 2.12.1, MPS enabled |
| MLX | 0.31.2 |
| TileLang branch | `metal-gemm-perf` |
| TileLang commit | `6f952ed9` |
| TVM submodule | `11c1968acf` |

Benchmark conditions: fp16 input, fp16 output, internal fp32 accumulation,
full-tile shapes, default path `ct_global`, `block_M=64`, `block_N=128`,
`threads=128`. Aggregation method: after warmup, timed runs are executed
consecutively, and TFLOPS is computed from the average latency over
`repeats=100`. This is not median or best-of-N.

| Shape | PyTorch MPS | MLX | TileLang | TileLang / MLX |
| --- | ---: | ---: | ---: | ---: |
| 4096 x 4096 x 4096 | 13.6 TFLOPS | 12.7 TFLOPS | 12.9 TFLOPS | 101% |
| 2048 x 2048 x 2048 | 13.6 TFLOPS | 11.5 TFLOPS | 12.8 TFLOPS | 112% |
| 4096 x 2048 x 4096 | 13.6 TFLOPS | 12.5 TFLOPS | 12.8 TFLOPS | 102% |
| 2048 x 4096 x 4096 | 13.4 TFLOPS | 12.4 TFLOPS | 12.6 TFLOPS | 102% |

Conclusion: on these full-tile shapes, TileLang is at or slightly above MLX and
about 5-6% behind PyTorch MPS. The shared staging path is correct but slower.
MLX-style swizzle is correct but is not currently the fastest default strategy.

### 5.7 Implementation Map

| Area | File | Responsibility |
| --- | --- | --- |
| Language API | `tilelang/language/gemm_op.py` | Metal GEMM operand and offset handling |
| Annotation | `tilelang/language/annotations.py` | `T.use_swizzle(..., order="mlx")` mapping |
| Builtins | `tilelang/language/builtin.py` | cooperative tensor and simdgroup builtin wrappers |
| GEMM lowering | `tilelang/metal/op/gemm/gemm_metal.py` | direct-global/shared dataflow split, warp partition, dtype policy |
| Intrinsic helper | `tilelang/metal/intrinsics/metal_macro_generator.py` | simdgroup and cooperative tensor intrinsic emission |
| Metal pipeline | `tilelang/metal/pipeline.py` | Metal-specific pipeline transform placement |
| Fragment rewrite | `tilelang/metal/transform/metal_fragment_to_simdgroup.py` | legacy fragment accumulator to `metal.simdgroup` |
| Core GEMM op | `src/op/gemm.h`, `src/op/gemm.cc` | GEMM op metadata |
| Metal op lowering | `src/metal/op/gemm.cc`, `src/metal/op/utils.h` | `SelectInst`, validation, scope utilities |
| Metal codegen | `src/metal/codegen/codegen_metal.cc`, `.h` | MSL emission |
| TVM runtime | `3rdparty/tvm/src/runtime/metal/metal_module.mm` | guarded `MTLLanguageVersion4_0` selection |
| Runtime tests | `testing/python/metal/test_metal_gemm_v2.py` | Metal correctness |
| Codegen tests | `testing/python/metal/test_metal_gemm_v2_linux.py` | source-level Metal codegen |
| Simdgroup tests | `testing/python/metal/test_metal_simdgroup_store.py` | simdgroup direct store |
| Benchmark | `benchmark/matmul_metal/benchmark_matmul_metal.py` | PyTorch / MLX / TileLang comparison |

### 5.8 Developer Checklist

When changing lowering, check:

- whether `global/shared/local/metal.simdgroup/metal.cooperative_tensor` scope
  semantics changed;
- whether CUDA shared staging assumptions leaked into the Metal default path;
- whether the direct-global cooperative tensor fast path is preserved;
- whether unsupported shapes produce a clear error or a safe fallback.

When changing codegen, check:

- whether read-only buffers keep `const` and noalias buffers keep `__restrict`;
- whether simdgroup id and lane id are still explicit;
- whether generated MSL avoids repeated physical swizzle arithmetic;
- whether the fp32 accumulation to fp16 output store-cast path still works.

When changing TVM / Metal runtime code, check:

- whether MSL 4.0 is requested only when SDK, runtime OS, and hardware support
  Metal 4;
- whether the MSL 2.3 fallback is preserved and non-Metal-4 devices are not
  regressed.

To reproduce the fp16-output MSL snippet in Section 1.4 from the repository root
(codegen-only, no Metal runtime required):

```bash
TILELANG_DISABLE_CACHE=1 python - <<'PY'
import tilelang
import tilelang.language as T
from tilelang import tvm
from testing.python.metal.test_metal_gemm_v2_linux import matmul_gemm_v2_global_c

func = matmul_gemm_v2_global_c(
    128, 256, 128, 64, 128,
    dtype=T.float16,
    accum_dtype=T.float16,
    threads=128,
)

with tvm.transform.PassContext(), tvm.target.Target('metal'):
    print(tilelang.lower(func, target='metal').kernel_source)
PY
```

The important detail is the output buffer dtype: `C` is `T.float16`, so the
generated source uses `device half* C` and emits the fp32-to-fp16 store cast
shown in Section 1.4.

Recommended tests:

```bash
TILELANG_DISABLE_CACHE=1 python -m pytest testing/python/metal/test_metal_gemm_v2.py -q -x
TILELANG_DISABLE_CACHE=1 python -m pytest testing/python/metal/test_metal_gemm_v2_linux.py -q -x
TILELANG_DISABLE_CACHE=1 python -m pytest testing/python/metal/test_metal_simdgroup_store.py -q -x
```

Default benchmark:

```bash
TILELANG_DISABLE_CACHE=1 python benchmark/matmul_metal/benchmark_matmul_metal.py \
  --m 4096 --n 4096 --k 4096 --warmup 10 --repeats 100
```
