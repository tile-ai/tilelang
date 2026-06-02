Handoff: TileLang 6-Role Warp Specialization for FlashAttention SM100

Goal

Match the performance of the reference CUDA binary (969 TFLOPS 1SM, 1371
TFLOPS 2CTA) by faithfully replicating the warp specialization design
from the reference C++ kernel inside the TileLang DSL.

Reference binaries (measured 2026-05-18 on B200):
  - attention_kernel_1sm.so: 969 TFLOPS (b=1 s=32768 h=16 d=128)
  - attention_kernel_1sm_d256.so: 785 TFLOPS (b=1 s=16384 h=16 d=256)
  - attention_kernel.so (2-CTA): 1371 TFLOPS (b=1 s=16384 h=16 d=128)

Reference source:
  /home/yu.cheng/workspace/avo/kernels/attention_kernel_1sm.cu (826 lines)
Our TileLang kernel:
  tilelang/examples/flash_attention_sm100/attention_kernel_1sm.py
TileLang repo to modify:
  /home/yu.cheng/workspace/tilelang/ (synced via mutagen)

---
Current Status (updated 2026-05-18)

Headline: Non-split (3-role) kernel = 753 TFLOPS. 4-role split has
correctness bug (O1 tile wrong). Root cause IDENTIFIED, partial fix
implemented but not complete.

Performance table (BF16, non-causal, b=1, d=128):
  ┌──────────────────┬──────────┬─────────┬────────┐
  │ Shape (s × h)    │  Mine    │  Avo    │ % avo  │
  ├──────────────────┼──────────┼─────────┼────────┤
  │ 4096  × 8        │ 584      │ 744     │ 78%    │
  │ 8192  × 16       │ 651      │ 779     │ 84%    │
  │ 16384 × 16       │ 753      │ 936     │ 80%    │
  └──────────────────┴──────────┴─────────┴────────┘

---
ROOT CAUSE of the 4-role split correctness bug

The 4-role split (main_kq2_split) separates the correction WG (tid
256-383) from the MMA WG (tid 384-511). The correction WG reads O from
TMEM, multiplies by scale, writes back. The MMA WG then accumulates new
P*V into the corrected O via tcgen05mma with accumulate=1.

PROBLEM: The tcgen05.st.sync.aligned.32x32b.x32 instruction (used by
the correction WG to store back to O_tmem) has COLUMN-ADDRESS-DEPENDENT
cross-WG TMEM visibility behavior:

  | Store instruction | O0 (cols ~256) | O1 (cols ~384) |
  |-------------------|----------------|----------------|
  | x32 (default)     | ✅ 0.002       | ❌ 0.13        |
  | x16 (MAX_LOGN=4)  | ❌ 0.097       | ✅ 0.001       |

x32 stores provide cross-WG visibility for O0's column range but NOT
O1's. x16 stores fix O1 but break O0.

Additionally, the O0 commit (tcgen05_mma_arrive after O0's PV gemm)
interferes with O1's TMEM visibility. When O0's correction stores use
x16 AND the commit fires, O1 becomes correct — but then O0 breaks.

The "no intermediate commit" pattern (matching avo's single commit
after both tiles) also doesn't work in TileLang because the single-
wait pattern loses fence coverage for both tiles regardless of manual
fence placement.

The avo reference avoids all these issues by:
1. Using per-warp tcgen05_ld_16/st_16 (x16) with explicit row offsets
2. Single commit after both O0+O1 PV gemms (no intermediate commit)
3. Software-pipelined stores: all x16 stores issued, then one wait_st
4. Per-tile tcgen05_fence_after_sync immediately before each tile's ops

---
INFRASTRUCTURE ALREADY IMPLEMENTED

1. T.tcgen05_gemm(mbar=None) — no-commit gemm API
   Files: tilelang/language/gemm_op.py (mbar optional)
          tilelang/cuda/op/gemm/gemm_tcgen05.py (no_commit handling)
          tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py

2. tcgen05_ld_x16 / tcgen05_st_x16 — x16 store/load interface
   Files: src/tl_templates/cuda/copy_sm100.h (x16 templates)
          src/op/builtin.h + builtin.cc (op registration)
          src/backend/cuda/codegen/codegen_cuda.cc (codegen)
          src/backend/cuda/op/copy.cc (annotation-based selection)
          src/transform/inject_tcgen05_fence.cc (op recognition)

   Usage: T.copy(src, dst, annotations={"use_x16": True})

3. inject_tcgen05_fence.cc fixes:
   - before_thread_sync injection restored (needed for cross-WG)
   - duplicate push_back bug fixed (was doubling every statement!)
   - x16 ops recognized as tcgen05/TMEM operations

---
WHAT STILL NEEDS TO BE DONE

The core unsolved problem: making BOTH O0 and O1 simultaneously correct
in the cross-WG correction scenario.

Option A: Fix the tcgen05.st cross-WG visibility at the instruction level
  - Root cause is that x32 stores don't provide visibility for high
    TMEM columns (384+) cross-WG, while x16 doesn't for low columns.
  - Possible fix: use a DIFFERENT TMEM allocation layout that puts both
    O tiles in a column range where ONE instruction width works for both.
  - The codegen reorders allocations, so DSL order doesn't control
    physical columns. Need to investigate copy.cc/allocator logic.
  - Or: implement avo's exact pattern — per-warp stores with explicit
    row offsets, async pipeline (all stores then one wait_st, no per-
    store wait). This requires bypassing TileLang's copy_sm100.h template
    which has per-store fence_view_async_tmem_store() built in.

Option B: Fix the single-commit + single-wait pattern
  - avo uses ONE commit after both tiles, ONE mb_pv wait in correction.
  - Our version fails (0.55) because tcgen05_after_thread_sync() fence
    doesn't provide sufficient coverage when O0+O1 ops are sequential.
  - Need to understand WHY the same pattern works in avo's CUDA but not
    in TileLang's generated CUDA (identical instructions but different behavior).
  - Likely cause: avo's per-warp load pattern (x16 with explicit row offset
    + separate wait_ld before use) establishes different HW state vs our
    full-WG x32 load with built-in wait_ld.

Option C: Implement avo's correction_warp_fn as inline code
  - Write the correction WG logic as raw CUDA (inline PTX) directly in
    the generated code, bypassing TileLang's copy lowering entirely.
  - We proved that calling a __device__ function via T.call_extern breaks
    fence propagation. But INLINE code within the same outlined function
    would work.
  - This requires either: modifying the codegen to emit inline PTX for
    marked operations, OR implementing Changes 3+4 in copy.cc (partial-
    row support + warp-granular thread alignment).

RECOMMENDED PATH: Option A (allocation layout fix) or Option C (inline
correction). Option B requires understanding a subtle HW behavior that
may not be solvable without per-warp operations.

---
Key Files Cheat Sheet

Purpose: Target kernel (Python)
Path: examples/flash_attention_sm100/attention_kernel_1sm.py
────────────────────────────────────────
Purpose: Reference C++ kernel (1SM)
Path: /home/yu.cheng/workspace/avo/kernels/attention_kernel_1sm.cu
────────────────────────────────────────
Purpose: Reference C++ kernel (2CTA)
Path: /home/yu.cheng/workspace/avo/kernels/attention_kernel.cu
────────────────────────────────────────
Purpose: TileLang TMEM copy lowering
Path: src/backend/cuda/op/copy.cc
────────────────────────────────────────
Purpose: TileLang TMEM fence injection
Path: src/transform/inject_tcgen05_fence.cc
────────────────────────────────────────
Purpose: TileLang TMEM PTX wrappers (x32 + x16 variants)
Path: src/tl_templates/cuda/copy_sm100.h
────────────────────────────────────────
Purpose: TileLang gemm lowering (mbar=None support)
Path: tilelang/cuda/op/gemm/gemm_tcgen05.py
────────────────────────────────────────
Purpose: TileLang barrier wrappers
Path: src/tl_templates/cuda/barrier.h
────────────────────────────────────────
Purpose: TileLang op registration
Path: src/op/builtin.h, src/op/builtin.cc
────────────────────────────────────────
Purpose: TileLang CUDA codegen
Path: src/backend/cuda/codegen/codegen_cuda.cc
