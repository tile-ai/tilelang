# Project Plan — TT Backend MVP (Matrix Multiplication Dry Run)

## Scope & Goals
- Deliver the minimal Tenstorrent backend path that lowers a TileLang GEMM into Metalium-ready host/kernels without executing on hardware.
- Lock the MVP operating point to: contiguous per-core schedule, interleaved DRAM tilization via TensorAccessor, bf16 tensors, and no user-authored annotations.
- Provide a compile-only "dry run" that emits reader/compute/writer kernels, a host program stub, and scheduling metadata (`tt.plan.json`).
- Keep the existing CUDA/HIP/CPU backends untouched; the TT path activates only when the `tenstorrent` target is requested.

## Default Operating Point Assumptions
- **Sharding/layout:** Use **DRAM interleaved tensors** backed by the TT-Metalium `TensorAccessor` (see `tt_metal/tt_metalium/runtime/tensor_accessor.*`). Each tile is 32×32 bf16; interleaving handles per-core striping without manual address swizzling.
- **Schedule:** Static, contiguous tile ranges per core (`policy="contiguous"`, `order="row_major"`). No K-panel chunking or multicast.
- **Kernels:** One reader, one compute, one writer kernel per active Tensix core using a depth-2 circular buffer pipeline.
- **Host runtime:** Generates runtime args `[start_id, count, grid_x, grid_y, kt_tiles]` for the compute kernel; reader/writer derive DRAM strides solely from interleaved descriptors.

## Workstream 1 — Frontend Integration & Target Selection
**Outcome:** TileLang recognizes `target="tenstorrent"`, synthesizes default TT annotations, and routes the module through a TT-specific lowering stack.

**Implementation**
- Extend `tilelang/utils/target.py` to register `"tenstorrent"` and skip auto-detection logic (explicit opt-in only).
- Add a target adapter in `tilelang/engine/lower.py` and `tilelang/engine/__init__.py` that dispatches to a `tilelang.engine.tt.lower` helper when the TT target is active; reuse existing CUDA/HIP branches otherwise.
- Introduce `python/tilelang_tt/target.py` with a small helper that stamps default TT schedule/sharding attrs (contiguous/interleaved) when user code omits them.
- Wire the helper into the standard lowering entry point (`tilelang/engine/lower.lower`) right after target determination.
  - **Justification:** The TT default synthesis is backend-specific and would clutter the generic TileLang frontend if inlined; isolating it in `tilelang_tt` keeps other targets pristine.

**Testing**
- New Python unit test `tests/python/tt/test_target_registration.py` that checks:
  - `determine_target("tenstorrent", return_object=True)` returns a `Target` named `tenstorrent`.
  - Lowering a toy PrimFunc with `target="tenstorrent"` injects the default TT attrs in the resulting IRModule.

## Workstream 2 — Schedule & Sharding Metadata
**Outcome:** Inject TT schedule/shard metadata describing contiguous per-core ranges and DRAM interleaved tilization.

**Implementation**
- Add `src/tt/transform/infer_tt_schedule.cc` implementing `InferDefaultTTSchedule`.
  - Reads `T.Kernel` metadata, enumerates tiles (`grid_x * grid_y`), partitions them by core count, and stores `tt.schedule` plus runtime-arg schemas.
  - **Justification:** No existing transform computes TT runtime metadata; adding a dedicated pass avoids overloading GPU-centric passes such as `LowerL2Persistent`.
- Add `src/tt/transform/infer_tt_shard.cc` providing `InferDefaultTTShard`.
  - Generates interleaved DRAM descriptors referencing TensorAccessor stride rules; for new files we depend on TT-metal headers during codegen but keep the pass pure metadata.
  - Marks non-multiple-of-32 axes for later padding.
- Update `python/tilelang_tt/__init__.py` to expose these passes so they can be invoked from Python.
- Register both passes in the lowering sequence applied by Workstream 3.

**Testing**
- C++ unit tests under `tests/cpp/tt/test_infer_tt_schedule.cc` and `tests/cpp/tt/test_infer_tt_shard.cc` using `TVM_REGISTER_GLOBAL` to invoke the passes on synthetic PrimFuncs; assert emitted attrs match expected contiguous ranges and interleaved descriptors.
- Python regression `tests/python/tt/test_inferred_metadata.py` to ensure a TileLang matmul lowered with TT target carries the metadata expected by later passes.

## Workstream 3 — TIR Transform Pipeline
**Outcome:** Convert the annotated PrimFunc into TT-ready IR with persistent loops, interleaved addressing, and TT-specific intrinsics.

**Implementation**
1. **`GridToPersistentTT` (`src/tt/transform/grid_to_persistent.cc`):**
   - Wrap the kernel body with `for (i = 0; i < count; ++i)` recovering `(bx, by)` from `start_id + i`.
   - Replace symbolic `bx/by` bindings, annotate `tt.runtime_args`.
   - **Justification:** Existing `persist_threadblock` (GPU) assumes CUDA thread semantics and cannot express TT runtime arg wiring.
2. **`TTShardToCoreMap` (`src/tt/transform/shard_to_core_map.cc`):**
   - Use schedule metadata to pick a rectangular `CoreRangeSet`; attach `tt.core_ranges` and per-core `(start_id, count)` arrays for host emission.
   - **Justification:** TileLang has no notion of Tensix topology today; dedicating a pass keeps the knowledge localized.
3. **`TilePadTT` (`src/tt/transform/tile_pad.cc`):**
   - Insert pad/unpad ops where shapes are not tile-multiples; prefer reader-side zero fill when possible.
4. **`MemorySpaceLowerTT` (`src/tt/transform/memory_space_lower.cc`):**
   - Lower shared/fragment buffers to circular buffers with depth=2; tag them with `tt.cb` attributes.
   - Convert `T.copy` to TT enqueue/dequeue intrinsics.
5. **`TensorizeTT` (`src/tt/transform/tensorize_matmul.cc`):**
   - Match matmul loop nests and replace with `tt.matmul_tiles(cb_a, cb_b, cb_c)`.
6. **`VerifyTTIR` (`src/tt/transform/verify.cc`):**
   - Ensure required attrs, runtime args, and CB invariants are present before codegen.
- Update `python/tilelang_tt/pipeline.py` to define an ordered pass list: inference (WS2) → transforms (steps 1–6) → host/codegen stubs.

**Testing**
- For each transform, add focused C++ tests in `tests/cpp/tt/` that apply the pass and compare the transformed IR against expected snippets (use `tvm::support::AsText`).
- Python-level test `tests/python/tt/test_tir_pipeline.py` runs the full pipeline on an MVP GEMM and asserts the final IR has the persistent loop, CB attrs, and Tensorize call.

## Workstream 4 — Code Generation & Runtime Glue
**Outcome:** Emit Metalium-compatible reader/compute/writer kernels and a host program stub capable of constructing an interleaved TensorAccessor view.

**Implementation**
- Introduce `src/tt/codegen/emit_kernels.cc` to walk TT-annotated PrimFuncs and produce C++ text for compute kernels; include headers from TT-metal for `TensorAccessor` and `CircularBuffer` definitions.
- Add `src/tt/codegen/emit_reader_writer.cc` to generate DRAM reader/writer kernels that program TensorAccessor iterators using interleaved layout metadata.
- Create `src/tt/codegen/emit_program.cc` building the host Program: allocate CBs, set runtime args, instantiate kernels on the `CoreRangeSet`, and dump `tt.plan.json`.
  - **Justification:** Existing CUDA/HIP codegen paths rely on NVCC/HIPCC FFI; TT requires a distinct BYOC module that integrates with Metalium headers and the dry-run artifact flow.
- Provide Python glue in `python/tilelang_tt/codegen.py` registering `target.build.tilelang_tt` and `target.build.tilelang_tt_without_compile` with TVM.

**Testing**
- Golden-file comparisons in `tests/python/tt/test_codegen_artifacts.py` that run the pipeline, inspect generated `compute.cpp`, `reader.cpp`, `writer.cpp`, and `tt.plan.json`, and diff against checked-in templates under `tests/python/tt/golden/`.
- Unit test `tests/python/tt/test_tensor_accessor_interleaving.py` verifying emitted reader/writer indices align with expected interleaved offsets for small matrices (compare against handcrafted TensorAccessor calculations).

## Workstream 5 — Tooling, Testing, and Validation
**Outcome:** Establish reproducible dry-run validation, including the TileLang GEMM MVP acceptance test.

**Implementation**
- Add command-line hook `python/tilelang_tt/cli.py` (optional) to dump artifacts for ad-hoc inspection during development.
- Integrate clang-format/clang-tidy checks for emitted sources in CI (reuse `format.sh`).
- Extend CI configuration to add a `TT_MVP_DRYRUN` job executing the tests below and archiving artifacts.
- Ensure the pipeline returns structured metadata so downstream tooling can inspect per-core work splits.

**Testing**
- **TileLang MVP GEMM test:** Implement `tests/python/tt/test_matmul_mvp.py` that builds the canonical TileLang GEMM (from README Phase 0), lowers it with `target="tenstorrent"`, asserts all passes succeed, and validates that the generated `tt.plan.json` assigns the expected interleaved tiles. This test is the final acceptance gate for the MVP.
- Additional smoke test `tests/python/tt/test_dry_run_cli.py` invoking the optional CLI to confirm artifact emission.

## Workstream 6 — Documentation & Follow-Up
**Outcome:** Users and contributors understand the TT pipeline, defaults, and next steps.

**Implementation**
- Update `README.md` Phase 0 section to reference the new interleaved defaults, TensorAccessor dependency, and dry-run instructions.
- Add a HOWTO in `docs/tenstorrent/` (e.g., `docs/tenstorrent/dry_run_walkthrough.md`) detailing the CLI/output layout.
- Document API changes (`tenstorrent` target flag, new Python helpers) in `docs/api_reference.md` (or the appropriate API doc).

**Testing**
- Documentation lint job verifying new Markdown (spelling, links) via existing docs tooling.
- Manual review checklist ensuring instructions match the behavior validated in Workstream 5.

## Milestones & Sequencing
1. Land Workstreams 1–2 (target detection + metadata inference) with unit tests.
2. Implement Workstream 3 transforms sequentially, gating each with its dedicated C++ tests; land once `VerifyTTIR` passes on the MVP matmul.
3. Add Workstream 4 codegen and ensure golden artifacts stabilize; update CI.
4. Finalize Workstream 5 acceptance tests (including `test_matmul_mvp.py`) and enable the dry-run CI job.
5. Publish documentation updates (Workstream 6) concurrent with enabling the TT target for early adopters.

## Acceptance Criteria
- Running `pytest tests/python/tt/test_matmul_mvp.py` succeeds and produces serialized kernels/host artifacts using interleaved TensorAccessor layout.
- All per-workstream unit tests (C++ and Python) pass in CI, and golden artifacts remain stable across runs.
- Documentation clearly states defaults, limitations, and links to the TensorAccessor reference for contributors.

