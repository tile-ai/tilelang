Handoff: TileLang 6-Role Warp Specialization for FlashAttention
  SM100

  Goal

  Match the performance of the reference CUDA binary (912 TFLOPS) by
  faithfully replicating the 6-role warp specialization design from
  the reference C++ kernel inside the TileLang DSL.

  - Reference source:
  /home/yu.cheng/avo/kernels/attention_kernel_1sm.cu (826 lines)
  - Reference binary (target perf):
  /home/yu.cheng/avo/kernels/attention_kernel_1sm.so → ~912 TFLOPS
  - Our TileLang kernel: /data1/home/yu.cheng/tilelang/examples/flash_
  attention_sm100/attention_kernel_1sm.py
  - TileLang repo to modify: /home/yu.cheng/tilelang/ (symlinked dev
  tree; not the install path)

  The 6-role design (one CTA, 16 warps, 512 threads):

  ┌────────┬─────────┬───────────────────────────────────┬────────┐
  │ Warps  │   TID   │               Role                │  Regs  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 0-3    │ 0-127   │ softmax0 (Q-tile 0)               │ 184    │
  │ (WG0)  │         │                                   │ (inc)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 4-7    │ 128-255 │ softmax1 (Q-tile 1)               │ 184    │
  │ (WG1)  │         │                                   │ (inc)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 8-11   │ 256-383 │ correction / O-rescale            │ 64     │
  │ (WG2)  │         │                                   │ (dec)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 12     │ 384-415 │ MMA issuer (QK + PV via           │ 80     │
  │        │         │ tcgen05.mma.cta_group::1)         │ (dec)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 13     │ 416-447 │ TMA producer (Q/K/V loads)        │ 80     │
  │        │         │                                   │ (dec)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 14     │ 448-479 │ Epilogue (TMA store O)            │ 80     │
  │        │         │                                   │ (dec)  │
  ├────────┼─────────┼───────────────────────────────────┼────────┤
  │ 15     │ 480-511 │ Empty (donates regs)              │ 80     │
  │        │         │                                   │ (dec)  │
  └────────┴─────────┴───────────────────────────────────┴────────┘

  Register budget: 8×184 + 4×64 + 4×80 = 2048 regs/thread → 65536
  regs/SM (full SM occupancy).

  ---
  Current Status (updated 2026-05-18)

  Headline: best stable kernel = 753 TFLOPS at s=16384 h=16
  (~83% of the 912 TFLOPS target, ~81% of avo at the same shape).
  Goal NOT yet achieved.

  Performance table (BF16, non-causal, b=1, d=128)

  ┌──────────────────┬──────────┬─────────┬────────┐
  │ Shape (s × h)    │  Mine    │  Avo    │ % avo  │
  ├──────────────────┼──────────┼─────────┼────────┤
  │ 4096  × 8        │ 584      │ 731     │ 80%    │
  │ 8192  × 16       │ 651      │ 768     │ 85%    │
  │ 16384 × 16       │ 753      │ 928     │ 81%    │
  └──────────────────┴──────────┴─────────┴────────┘

  What works (committed direction; uncommitted in working tree)

  - Change 1 (setmaxnreg) — IMPLEMENTED.
    T.set_max_nreg(count, mode) lowers to
    setmaxnreg.{inc|dec}.sync.aligned.u32 N. Tuned register split:
    mma_load WG gets 232, idle donor WG drops to 24. ~8-9% delta on
    the 3-role baseline (701 → 753 TFLOPS at s=16384 h=16).
  - Change 2 (cross-WG TMEM fence) — PARTIAL FIX.
    Extended IsPlainBarrierArrive in
    src/transform/inject_tcgen05_fence.cc:72-76 to also recognize
    ptx_arrive_barrier_lane0. Without this, fence injection silently
    skipped every lane-elect arrive, so any kernel using
    mbarrier_arrive(..., lane0=True) for cross-WG handoff was broken
    by construction.

  What's broken (THE blocker — the 4-role split)

  - main_kq2_split (sister prim_func in attention_kernel_1sm.py)
    separates the correction WG (tid 256-383) from the mma+TMA+epi WG
    (tid 384-511). Compiles cleanly; produces max_abs = 0.3-0.7 (wrong
    output).
  - Root cause: cross-WG TMEM coherency. The correction WG's
    tcgen05_st writes to O_tmem are NOT visible to the mma WG's
    subsequent tcgen05.mma reads, despite proper fence injection at
    the PTX level after the Change 2 fix.
  - SASS evidence: ptxas drops FENCE.VIEW.ASYNC.T before UTCHMMA even
    though it is present in the emitted PTX (asm volatile is not
    enough — ptxas treats the tcgen05 proxy fence as schedulable and
    reorders it out of the critical interval).
  - Variants attempted (all failed):
      lane0 vs warp-wide arrives;
      explicit .release.cta on mbarrier.arrive;
      mb_scale_consumed handshake;
      outline_warp_spec_branches ON vs OFF;
      tcgen05.wait::st.sync.aligned after the store;
      64-col vs 32-col chunking;
      explicit before/after fences manually emitted in the DSL.
  - Standalone minimal repros (test_single_wg_tmem.py,
    test_cross_wg_tmem.py) still fail at TileLang layout inference
    with "Failed to find a suitable instruction for tcgen05.st" — so
    we cannot bisect outside the FA context until Change 3 + Change 4
    relax the copy.cc layout constraints.

  Working tree (uncommitted)

  - src/transform/inject_tcgen05_fence.cc — Change 2 (3-line fix)
  - examples/flash_attention_sm100/attention_kernel_1sm.py —
    setmaxnreg in kq_stages=2 path, sister main_kq2_split (~580
    lines) for the 4-role split (currently broken).
  - examples/flash_attention_sm100/test_{single,cross}_wg_tmem.py —
    failed minimal repros.

  Pending tasks (none of these have been started)

  - Change 3 (TMEM partial-row support, copy.cc:822 row range check)
  - Change 4 (TMEM warp-granular thread alignment, copy.cc:781-786)
  - Full 6-role rewrite (depends on Changes 3 + 4)

  ---
  TOP PRIORITY (do this first): unblock the 4-role split

  Everything else is gated on this. The 4-role split
  (main_kq2_split in attention_kernel_1sm.py) is the smallest
  intermediate step toward the 6-role layout, and it currently
  produces wrong output (max_abs = 0.3-0.7) because the correction
  WG's TMEM stores are not visible to the mma WG's subsequent
  tcgen05.mma. Until this single handoff works, Changes 3/4 and the
  full 6-role rewrite cannot be validated.

  Concrete plan to land this:

  1. Verify what is actually in the cubin at the cross-WG handoff:
     cuobjdump --dump-sass on the latest split build and confirm
     whether FENCE.VIEW.ASYNC.T appears between the correction WG's
     STTM.x32 and the mma WG's UTCHMMA. Compare against the avo
     reference cubin (which is known to work) at the same point.
  2. If FENCE.VIEW.ASYNC.T is missing, the fix is in the codegen,
     not the DSL:
       - Emit the tcgen05 proxy fence as a SASS-level barrier ptxas
         cannot reorder. Options, in order of preference:
           (a) wrap the fence + the dependent op in a single inline
               asm block so ptxas treats them as one instruction;
           (b) add an artificial register dependency between the
               fence and the subsequent tcgen05.mma operand;
           (c) emit a stronger fence (fence.proxy.async.shared::cta
               + tcgen05.fence::after_thread_sync) at the consumer
               side, immediately before the tcgen05.mma issue.
       - Implementation lives in src/tl_templates/cuda/tcgen_05.h
         and the codegen site for tcgen05_mma in
         src/backend/cuda/codegen/ptx.cc.
  3. If FENCE.VIEW.ASYNC.T IS present but the kernel still produces
     wrong output, the bug is in the mbarrier protocol, not the
     fence — re-audit the arrive count and phase parity on the
     mb_corr / mb_o / mb_scale chain.
  4. Acceptance: main_kq2_split path reaches max_abs < 0.01 on the
     standard test config; perf at s=16384 h=16 should land in the
     820-870 TFLOPS range (interim milestone). Without the 4-role
     split working, Changes 3 and 4 should not be attempted, since
     they add complexity to a layout we cannot yet validate.

  ---
  Required Changes — Priority Order

  Each change is feasible inside the local TileLang dev tree
  (/home/yu.cheng/tilelang/). Effort is rough days of work for a
  competent compiler engineer.

  Change 1: setmaxnreg builtin (small, ~half day, high ROI even before
   6-role)

  Why: avo's 6-role split depends on aggressive register rebalancing.
  Without it, even a successful 6-role split won't help — softmax will
   still spill.
  Files:
  - Add T.set_max_nreg(count: int, mode: "inc"|"dec") Python intrinsic
   (tilelang/language/builtins.py or similar)
  - Emit asm("setmaxnreg.{inc|dec}.sync.aligned.u32 N;") in CUDA
  codegen
  - Optionally: auto-inject inside outline_warp_spec_branches pass
  based on per-branch reg estimate
  Test: standalone kernel that calls T.set_max_nreg(184, "inc") and
  inspect generated .cu.

  Change 2: Cross-WG TMEM fence / release scope — the real blocker
  (~1-2 days; root-cause dive needed first)

  Why: Without this, 6-role split produces wrong output even when
  register layout is correct.
  Investigation needed first (don't blindly patch):
  1. Build the broken split kernel (revert from git stash if needed)
  and run under nsight-compute --target-processes all --section
  MemoryWorkloadAnalysis,SchedulerStats to inspect TMEM scoreboard
  state at the handoff point.
  2. Compare PTX (not just SASS) for the mbarrier.arrive instruction
  emitted by Barrier::arrive() in the split vs baseline.
  3. Verify whether inject_tcgen05_fence.cc actually injects
  tcgen05.fence::before_thread_sync before the cross-WG arrive —
  IsPlainBarrierArrive currently only matches ptx_arrive_barrier /
  ptx_arrive_cluster_barrier, NOT lane-elect arrives.

  Likely fix (one of):
  - Extend IsPlainBarrierArrive in
  src/transform/inject_tcgen05_fence.cc to recognize lane-elect arrive
   forms.
  - Add new builtin T.barrier_arrive_release() that emits raw
  mbarrier.arrive.release.cta.shared::cta.b64 _, [bar]; bypassing the
  Cutlass ClusterTransactionBarrier wrapper.
  - Or change src/tl_templates/cuda/barrier.h::mbarrier_arrive to use
  .release.cta modifier explicitly.

  Test: revert to the failing split kernel (search for
  tl::ptx_arrive_barrier_lane0 calls in 4-role variant in git
  stash/history); verify max_abs < 0.01.

  Change 3: TMEM partial-row support (~1 day)

  File: src/backend/cuda/op/copy.cc:822
  if (tmem_phy_row_min != 0 || tmem_phy_row_max != 127) return;  //
  remove
  Replace with: allow [row_min, row_max] to be 32-multiple sub-range;
  encode row_min into bits [16:23] of the TMEM dst_addr (TMEM PTX
  addressing format).
  Also update tcgen05_st_32dp32bNx/ld wrappers in
  src/tl_templates/cuda/copy_sm100.h:200-263 to accept a row_offset
  parameter (or have the codegen bake it into tmem_start_col).

  Change 4: TMEM warp-granular thread alignment (~2 days)

  File: src/backend/cuda/op/copy.cc:781-786
  ICHECK(FloorMod(T.thread_bounds->min, WARPGROUP_SIZE) == 0 &&
         num_threads % WARPGROUP_SIZE == 0)
  Relax to WARP_SIZE (32). Then downstream num_useful_wgs =
  num_threads / WARPGROUP_SIZE and expandTcgen05Layout need to accept
  warp counts.
  Risk: layout inference inside expandTcgen05Layout may assume
  WG-granular fragment shapes; touch carefully.

  Change 5 (optional, for debuggability): Independent fragment layout
  for TMEM IO (~3-5 days)

  Add T.fragment_for_tmem(tmem_buf, slice) API so register fragments
  can derive their layout from a TMEM allocation without needing a
  tcgen05_gemm anchor. Unlocks standalone unit tests for TMEM
  round-trip.

  ---
  Test Configuration

  Correctness target

  - Test script:
  examples/flash_attention_sm100/test_attention_kernel_1sm.py (or
  whatever the canonical test entry is — find via pytest
  --collect-only in that directory).
  - Config: batch=1, seqlen=1024, num_heads=8, head_dim=128,
  dtype=BF16, non-causal.
  - Pass criterion: max_abs(out - ref) < 0.01, mean_abs < 0.001 (ref =
   PyTorch SDPA or hand-rolled FA).

  Performance target

  - Same shape; run repeated and measure median TFLOPS.
  - Baseline: 701 TFLOPS (current 3-role).
  - Reference: 912 TFLOPS (avo .so).
  - Acceptable interim milestones:
    - After Change 1 (setmaxnreg): aim for ~750 TFLOPS (~7%) by
  reducing softmax spill.
    - After Changes 2+3+4 (6-role unlocked): aim for ~880+ TFLOPS
  (~25% over baseline).

  Reference comparison harness

  # Build avo reference once
  cd /home/yu.cheng/avo/kernels && make
  # Run benchmark on same shape
  python /data1/home/yu.cheng/tilelang/examples/flash_attention_sm100/
  bench_vs_avo.py
  # (may need to create this; load both .so via ctypes, time identical
   inputs)

  SASS / cubin inspection

  - Dump SASS: cuobjdump --dump-sass /tmp/k_baseline.cubin >
  /tmp/sass.txt
  - Key counters to track (grep these in SASS):
    - UTCHMMA (tcgen05 MMA issues; baseline=128)
    - LDTM.x32, STTM.x32 (TMEM load/store; baseline=32 / 16)
    - FENCE.VIEW.ASYNC.T, FENCE.VIEW.ASYNC.S (TMEM/SMEM proxy fences;
  baseline=24 / 17)
    - MEMBAR.ALL.CTA, BAR.SYNC, ATOMS.OR (sync primitives)

  ---
  Key Files Cheat Sheet

  Purpose: Target kernel (Python)
  Path: examples/flash_attention_sm100/attention_kernel_1sm.py
  ────────────────────────────────────────
  Purpose: Reference C++ kernel
  Path: /home/yu.cheng/avo/kernels/attention_kernel_1sm.cu
  ────────────────────────────────────────
  Purpose: Reference helpers
  Path: /home/yu.cheng/avo/kernels/fmha_2cta_raw.cuh
  ────────────────────────────────────────
  Purpose: TileLang TMEM copy lowering
  Path: src/backend/cuda/op/copy.cc:770-900
  ────────────────────────────────────────
  Purpose: TileLang TMEM fence injection
  Path: src/transform/inject_tcgen05_fence.cc
  ────────────────────────────────────────
  Purpose: TileLang TMEM PTX wrappers
  Path: src/tl_templates/cuda/copy_sm100.h, tcgen_05_st.h,
    tcgen_05_ld.h
  ────────────────────────────────────────
  Purpose: TileLang barrier wrappers
  Path: src/tl_templates/cuda/barrier.h
  ────────────────────────────────────────
  Purpose: Generated CUDA (baseline)
  Path: /tmp/cu_baseline.cu
  ────────────────────────────────────────
  Purpose: Generated CUDA (broken 4-role split)
  Path: /tmp/cu_split.cu
  ────────────────────────────────────────
  Purpose: SASS dumps
  Path: /tmp/sass_baseline.txt, /tmp/sass_split.txt
  ────────────────────────────────────────
  Purpose: Failed standalone repros
  Path: examples/flash_attention_sm100/test_{single,cross}_wg_tmem.py

  ---
  Suggested Execution Order

  1. Change 1 (setmaxnreg) — independent, ship it, measure perf delta
  on 3-role baseline.
  2. Change 2 root-cause dive — Nsight Compute + PTX diff first; only
  patch after you've identified whether the missing piece is fence
  injection, release scope, or arrive-count.
  3. Change 3 + 4 (partial-row + warp-granular TMEM) — needed together
   to enable single-warp mma_warp_fn / producer_warp_fn /
  epilogue_warp_fn as in avo.
  4. Rewrite kernel to 6-role — once Changes 2-4 land, rewrite
  attention_kernel_1sm.py with 6 separate if tid < ...: ... elif ...
  branches, each calling one role's logic, mirroring the avo .cu
  structure section-by-section.
  5. Change 5 (independent fragment layout) — only if minimal repros
  are needed for further debugging.

  Out-of-Scope (do not pursue until 6-role lands)

  - kKVStages=3 (3-stage KV pipeline) — orthogonal, +10-15%, but
  easier after 6-role
  - Persistent block — orthogonal, depends on 6-role epilogue overlap
  - Softmax microopts (FMA_F32X2, EXP2_POLY_2, fmax3 tree) — ~5%, only
   if everything else is done
