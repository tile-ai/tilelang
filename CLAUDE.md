# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TileLang is a tile-level DSL for generating high-performance GPU/CPU kernels (GEMM, FlashAttention, MLA, etc.) built on top of the TVM compiler infrastructure. It compiles Python-like kernel definitions to CUDA (NVIDIA), HIP (AMD ROCm), Metal (Apple), or CPU backends.

**Primary optimization focus: AMD MI300X/MI355X (gfx942/gfx950) with ROCm.**

## Build Commands

### ROCm build (editable/dev mode)
```bash
bash build.sh
```

### Verify installation
```bash
python -c "import tilelang; print(tilelang.__version__)"
```

## Testing

Tests are in `testing/python/` and run with pytest from the `testing/` directory.

### Run all AMD/ROCm tests
```bash
cd testing
pytest --verbose --color=yes --durations=0 --showlocals --cache-clear \
  --ignore=./python/runtime --ignore=./python/transform \
  ./python
```
Note: `runtime` and `transform` test directories are currently excluded on ROCm (need repair).

### Run a single test file
```bash
cd testing
pytest --verbose python/amd/test_tilelang_test_amd.py
```

### Run AMD-specific tests only
```bash
cd testing
pytest --verbose python/amd/
```

### Run GEMM example (primary AMD test kernel)
```bash
python examples/gemm/example_gemm.py
```

### Manually compile generated HIP kernel and inspect assembly
```bash
bash examples/gemm/compile.sh   # compile cached host_kernel.cu with hipcc
bash examples/gemm/pure.sh      # extract clean assembly
```

## Linting

Pre-commit hooks handle linting (clang-format for C++, ruff for Python, codespell, pymarkdown):
```bash
pre-commit run --all-files
```

Ruff only:
```bash
ruff check tilelang/
ruff format tilelang/
```

## Architecture

### Language layers

```
Python DSL (tilelang/language/)    # User-facing: T.Kernel, T.alloc_shared, T.copy, T.gemm, T.Pipelined
        ↓ parse → TIR
Engine (tilelang/engine/)          # Compilation phases, lowering passes, phase.py orchestrates
        ↓
Intrinsics (tilelang/intrinsics/)  # Hardware macro generators: MFMA (AMD), MMA/WGMMA (NVIDIA)
        ↓
C++ codegen (src/target/)         # codegen_hip.cc (ROCm), codegen_cuda.cc (NVIDIA)
        ↓
Templates (src/tl_templates/)     # hip/gemm.h, hip/copy.h — HIP device code templates
        ↓
JIT (tilelang/jit/)               # Cython adapter wraps compiled kernels for Python invocation
```

### Key source directories

- **`tilelang/language/`** — DSL primitives: `kernel.py`, `allocate.py`, `copy_op.py`, `gemm_op.py`, `loop.py` (Pipelined, Parallel, Serial, Unroll)
- **`tilelang/intrinsics/mfma_macro_generator.py`** — AMD MFMA (Matrix Fused Multiply-Add) TIR macro expansion. Defines `MatrixCoreIntrinEmitter` with `ldmatrix_a_single`, `ldmatrix_b_single`, `mfma_cell`, `mfma_col_slice`, `mfma_slice`
- **`tilelang/intrinsics/mfma_layout.py`** — MFMA layout specifications for gfx9 targets
- **`tilelang/engine/phase.py`** — Compilation pass orchestration. On HIP, skips `MergeSharedMemoryAllocations` to keep shared buffers independent (critical for alias analysis)
- **`tilelang/tileop/gemm/gemm_mfma.py`** — `GemmMFMA.lower()` → `_gemm_ssr()`: SS GEMM lowering with fine-grained S2R/MFMA interleaving, scheduling barriers, and setprio
- **`src/target/codegen_hip.cc`** — HIP code generation: `vmcnt` tracking for `buffer_load_b128...lds`, `s_barrier` emission, static `__shared__` declarations
- **`src/tl_templates/hip/copy.h`** — G2S copy using `buffer_load_b128...lds` (truly async, no VGPR intermediary)
- **`src/tl_templates/hip/gemm.h`** — `MfmaTraits`, `GemmTensorOp`, `tl_setprio_hi/lo` (asm volatile s_setprio with VGPR dependency)
- **`src/transform/lower_tile_op.cc`** — Tile operation lowering including G2S "swizzle-swap" (moves XOR swizzle from LDS store side to global load side)
- **`tilelang/layout/`** — Layout and swizzle representations
- **`tilelang/autotuner/`** — Auto-tuning framework
- **`tilelang/jit/adapter/wrapper.py`** — JIT adapter; `TLHIPSourceWrapper.get_launch_smem_size()` returns 0 for static shared memory

### AMD MI300X/MI355X specific architecture

**G2S (Global→Shared) async copy pipeline:**
1. Uses `buffer_load_b128...lds` instruction (64 lanes × 16 bytes = 1024 bytes per instruction, truly async via vmcnt)
2. "Swizzle-swap" in `lower_tile_op.cc` moves XOR swizzle from LDS store side to global load side so LDS addresses are lane-contiguous
3. `ptx_wait_group(N)` maps to `vmcnt(N × ops_per_group)` in `codegen_hip.cc`
4. Uses `s_barrier` (not `__syncthreads()`) to avoid implicit `vmcnt(0)`
5. Each shared buffer gets independent `__shared__` declaration (not merged into one `extern __shared__`) so LLVM alias analysis doesn't insert spurious `vmcnt(0)` before `ds_read` instructions

**GEMM scheduling (SS variant):**
- Fine-grained S2R/MFMA interleaving for `v_mfma_f32_16x16x32_bf16` on gfx950
- `__builtin_amdgcn_sched_barrier(0)` prevents LLVM from reordering across load/MFMA boundaries
- `tl_setprio_hi/lo` uses `asm volatile` with VGPR constraints to prevent LLVM from optimizing away `s_setprio` or moving MFMA/load instructions across it
- `__builtin_amdgcn_s_barrier()` for workgroup synchronization and round separation
- Typical block tile: 256×256×64 (M×N×K), 512 threads (8 warps), 64 MFMA + 24 ds_read per ki iteration

**Generated kernel location:** `~/.tilelang/cache/<hash>/host_kernel.cu`

### Third-party dependencies (git submodules in `3rdparty/`)

- **TVM** — Compiler infrastructure (modified fork)
- **CUTLASS** — NVIDIA's CUDA templates (include + tools only)
- **Composable Kernel** — AMD's kernel library (include + library only)

## AMD-specific code conventions

- TIR scheduling barriers are emitted via `tir.call_extern("int32", "__builtin_amdgcn_sched_barrier", ...)` — never remove or reorder these
- The `k_pack` parameter controls how many bf16 elements are packed per MFMA instruction (typically 2 for gfx950)
- `num_xcds=8` on MI300X enables XCD remap optimization for GEMM
- ROCm builds set `CMAKE_HIP_STANDARD=17` and define `__HIP_PLATFORM_AMD__`
- Target architectures: `gfx942` (MI300X), `gfx950` (MI350X/MI355X)

## Key documentation

- `docs/g2s-swizzle-swap.md` — G2S buffer_load_b128...lds optimization and alias analysis workarounds (Chinese)
- `examples/gemm/GEMM_SCHEDULING_CONTEXT.md` — Fine-grained S2R/MFMA interleaving design for gfx950 (Chinese)
- `examples/deepseek_mla/amd/` — FlashMLA implementation for MI300X (forward + backward)

## Current optimization status: GEMM on gfx950

### Performance baseline

- **Kernel**: 8192x8192x8192 bf16 GEMM (NT layout), 256x256x64 block tile, 512 threads
- **Current**: ~1020 TFLOPS on MI355X (gfx950)
- **Target**: match HipKittens (~3 VALU/SALU between `buffer_load_dwordx4...lds` instructions)
- **Benchmark command**: `python examples/gemm/example_gemm.py`
- **Cache hash**: `6e6c3722f225bfabc1f384605632692c0b52ab5a2bff53cb8d65cf201367868f`
- **Resource usage**: 35 SGPRs, 204 VGPRs, 0 spills, occupancy 2 waves/SIMD

### Completed optimizations

1. **SRD hoisting** (already in codebase): The 128-bit buffer resource descriptor (`cp_async_make_srd`) is created once per buffer at kernel entry, eliminating 4x `readfirstlane` per `buffer_load_dwordx4` instruction.

2. **G2S voffset/soffset explicit precomputation** (implemented): All G2S voffset (per-lane byte offset) and soffset base (wave-uniform offset) values are precomputed into arrays **before** the pipeline k-loop. The inner loop only does array lookups + `k * stride` addition.
   - **Files modified**: `src/target/codegen_hip.h`, `src/target/codegen_hip.cc`
   - **New classes**: `G2SHoistCollector` (scans pipeline loop body for G2S ops with unrolled loop context), `G2SHoistGroup` (stores precomputed variable names and k-stride)
   - **New method**: `EmitG2SHoistPrecomputation(ForNode*)` — called from `VisitStmt_(ForNode*)` before emitting the pipeline loop header
   - **How it works**:
     1. `G2SHoistCollector` scans the pipeline loop body, finds all `ptx_cp_async` calls, records their source/dst expressions and enclosing unrolled loop vars
     2. For each G2S op, `DetectLinearEquation(offset, {k})` decomposes the source offset into `k*coeff + base`
     3. Before the k-loop, emit precomputed arrays (manually unrolled): substitute both k=0 and unroll_var=0,1,2,3 into the source expression, compute voffset and soff_base for each
     4. Inside the k-loop, `EmitDecomposedG2S` just emits: `cp_async_gs_v2<16>(lds_m0, srd, __g2s_voff[i], __g2s_soff[i] + k * 128)`
   - **Generated code (before k-loop)**:
     ```cpp
     uint32_t __g2s_voff_0[4], __g2s_soff_0[4];  // A: 4 unrolled loads
     // For each i=0..3: compute voffset and soff_base at k=0
     { uint32_t __g = (uint32_t)&A[...k=0,i=0...]; uint32_t __b = readfirstlane(__g);
       __g2s_voff_0[0] = __g - __b; __g2s_soff_0[0] = __b - (uint32_t)A; }
     // ... repeat for i=1,2,3
     uint32_t __g2s_voff_1[4], __g2s_soff_1[4];  // B: 4 unrolled loads
     // ... same pattern
     ```
   - **Generated code (inside k-loop)**:
     ```cpp
     for (int i_3 = 0; i_3 < 4; ++i_3) {
       uint32_t __lds_m0 = readfirstlane((uint32_t)&A_shared[((k+1)&1)*16384 + i_3*4096 + tid*8]);
       cp_async_gs_v2<16>(__lds_m0, __srd_A, __g2s_voff_0[i_3], __g2s_soff_0[i_3] + k * 128);
     }
     ```
   - **Why it's better than k=0 substitution alone**: The previous approach substituted k=0 but still computed voffset inline inside the loop, relying on LLVM LICM to hoist. The new approach **guarantees** hoisting at the codegen level — no LLVM dependency.
   - **LDS dst address**: Left as-is (readfirstlane on shared pointer is cheap and must vary per iteration for double buffering).

### How the G2S decomposition works in detail

The address expression for a G2S load of matrix A looks like:
```
A[blockIdx.y*2097152 + i_3*524288 + (threadIdx.x>>3)*8192 + k*64 + swizzle_bits(threadIdx.x) + 64]
```

Only `k*64` changes per k-iteration. Everything else depends only on blockIdx, threadIdx, and the unrolled loop variable i_3.

`DetectLinearEquation(offset, {k})` decomposes this into:
- `coeff[0]` = 64 (elements per k step)
- `base` = everything else (k-independent)

The codegen then:
1. For each unrolled index (i=0..3), substitutes k=0 AND i_3=i into the `address_of(BufferLoad(...))` PrimExpr
2. Computes `voffset = addr - readfirstlane(addr)` and `soff_base = readfirstlane(addr) - buf_base`
3. Stores into precomputed arrays before the loop
4. Inside the loop, each G2S load is: `cp_async_gs_v2(lds_m0, srd, voff[i], soff[i] + k * stride_bytes)`

### Investigated but not viable

1. **LDS m0 precomputation** (attempted, reverted): Tried precomputing `readfirstlane(&A_shared[...])` for both double-buffer slots into `uint32_t __g2s_lds_N[2][4]` arrays before the k-loop, then using `__g2s_lds_N[k&1][i_3]` inside the loop.
   - **Problem 1**: `s_mov_b32 m0` requires an SGPR input, but array values live in VGPRs. Wrapping with another `readfirstlane` fixes assembly.
   - **Problem 2**: Even with the readfirstlane wrapper, results were incorrect (23.6% element mismatches). The compiler allocated 163840 bytes LDS instead of 131072 — the precomputation referencing `A_shared`/`B_shared` before the k-loop may have affected the compiler's shared memory layout or alias analysis.
   - **Conclusion**: The LDS address computation is lightweight (3-4 SALU: LDS base constant + double-buffer select + unrolled offset), so the potential gain is marginal. Left as inline `readfirstlane`.

### Remaining optimization opportunities

1. **Fine-grained soffset management**: HipKittens uses a single SGPR for soffset and increments it with `s_add_u32` each iteration instead of recomputing `base + k*stride`. This saves one SALU multiply per load. Would require emitting an SGPR variable before the loop and incrementing inside.

2. **Inspect assembly**: Compile to assembly (`bash examples/gemm/compile.sh && bash examples/gemm/pure.sh`) and count instructions between consecutive `buffer_load_dwordx4` to see how close we are to HipKittens' ~3 instructions. The main remaining cost should be the LDS m0 readfirstlane.

3. **Scheduling barrier placement around G2S**: Currently `sched_barrier(0)` is used around S2R/MFMA blocks but not around G2S loads. Adding barriers around the G2S section could prevent LLVM from interleaving G2S address math with other instructions.

### Key TVM/TIR APIs used in codegen

- `arith::DetectLinearEquation(expr, {var})` — Decomposes `expr` into `var*coeff + base`. Returns `[coeff, base]` array or empty on failure. Header: `<tvm/arith/pattern.h>`
- `tir::Substitute(expr, subst_fn)` — Replaces variables in a TIR expression. The lambda `(const Var&) -> Optional<PrimExpr>` returns the replacement or empty Optional to keep original. Header: `<tvm/tir/stmt_functor.h>`
- `tir::builtin::address_of()` — TIR op wrapping `BufferLoad` into a pointer. Args: `[BufferLoad(buffer, [index])]`
- `PrintExpr(PrimExpr)` — CodeGenC method that converts a TIR expression to a C string. Can be called multiple times on different expressions safely.
