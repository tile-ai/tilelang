# Ascend tests

Keep each regression at the first boundary that owns its behavior. Shared
TileLang behavior belongs in `testing/python`. Preserve upstream tests; add shared
regressions only for necessary coverage of this fork's changes to shared code.

| Directory | Contract | Typical check |
| --- | --- | --- |
| `analysis` | Ascend source-program legality | Call the checker directly; check acceptance or a useful diagnostic |
| `transform` | Individual Ascend IR transformations | Before IR → one pass → relevant After IR structure |
| `auto_schedule` | Scheduling, storage ownership, core placement, synchronization and reuse | Explicit scheduled input with selected stages, cores and versions |
| `layout` | Ascend fractal geometry and numerical matrix paths | Inspect layout mappings; execute small kernels when physical data placement matters |
| `runtime`, `target` | Ascend launch ABI, current stream and target/compiler selection | Numerical runtime checks or isolated configuration tests |
| `language` | Ascend frontend, emitted device code and numerical behavior | Use lowering for codegen contracts; compile and execute for numerical contracts |

## Pass and device boundaries

- `test_ascend_normalize_fractal_storage` protects padded storage, aliases and
  access-pointer remapping without running layout inference or scheduling.
- `test_ascend_copy_bounds` checks clamped regions and semantic fills before
  scheduling. `test_ascend_lower_copy` checks DMA bursts, physical pitches,
  transfer guards and L1/L0 alignment on explicit layouts.
- `test_ascend_lower_gemm` checks logical MAD dimensions independently of
  physical allocation sizes. `test_ascend_insert_nd2nz` checks scatter layouts,
  padding, alignment and VF boundaries.
- `test_ascend_fractal_storage_integration` retains the one cross-pass check
  that padded alias storage determines the physical multibuffer pitch.

Use the architecture's vector width in numerical SIMD fixtures. One or two
vectors normally suffice for an instruction test; pipelines and persistent
loops belong only in tests that protect those behaviors. Compare all defined
output elements, and check neighboring storage when testing a partial write.
Codegen tests may lower to source; invoke the device compiler when the test
executes the result or specifically protects compiler acceptance.

## Scheduling tests

The scheduling suite is organized by the behavior being protected:

- `test_control_flow`, `test_schedule_units`, `test_tasks`, `test_estimate_latency`:
  stable control snapshots, task boundaries,
  explicit scheduling constraints and the Ascend hardware cost model.
- `test_buffer_ownership`, `test_buffer_epochs`, `test_epoch_guards`,
  `test_buffer_plan_validation`, `test_buffer_materialization`: storage owners,
  version clocks, guard scope and physical version indexing.
- `test_core_resolution`, `test_special_registers`: scalar availability,
  protocol participants and hardware-register dependencies.
- `test_sync_dependencies`, `test_sync_boundaries`, `test_flag_allocation`,
  `test_cross_core_flags`: dependency completion, conditional/empty loops,
  flag pairing, reservation and spilling.
- `test_access_footprints`, `test_buffer_lifetimes`,
  `test_sync_access_footprints`, `test_conflict_hints`: physical access coverage,
  live generations, aliases and explicit conflict contracts.
- `test_schedule_integration`, `test_reuse_integration`: the small set of
  checks that need interactions between passes, layouts, kernel splitting or
  alias-contract propagation.

`_ir.py` provides shared IR builders for source and scheduled inputs.
`auto_schedule/_scheduled_ir.py` adds task, stage and buffer-ring fixtures
without invoking compiler passes.
These are intermediate IR fixtures, not executable kernels: an InsertSync
fixture may begin with a read whose producer belongs to an earlier stage.
Keep the relevant access, guard and core metadata explicit.

For larger source-level footprint matrices, `_ir.schedule` prepares the input
through the requested boundary. Pass `auto_schedule=False` for synchronization
and reuse tests: this preserves the supplied order, stages and version counts
without running the latency estimator or solver. In this mode,
`tl.EstimateLatency` is not an available snapshot boundary. Then call the tested
pass explicitly. This helper performs no device lowering or compilation.

Assert buffer/variable identity, necessary index expressions, guard dominance,
core masks, conflicting lifetimes and matching set/wait events. Do not pin
compiler-generated names, arbitrary flag IDs, allocator offsets or the order
of independent tasks. A dependency order or a user-specified stage is a valid
contract. Full structural equality is useful only when the entire small output
is relevant. Inspect IR nodes instead of matching the IR printer's text.

Combine cases that vary one behavior in a parameter table with descriptive IDs.
Reduce application regressions to the required access, control-flow or layout
pattern. Do not import example kernels or external kernel collections; name
tests after the compiler behavior they protect.
Keep fixtures near their assertions; share builders only when their semantics
are identical. Frontend kernels should use `T.Tensor` and current VF APIs.
Numerical kernels must initialize every consumed value and have clear ownership
of output elements.

## Running

Use an Ascend-enabled build from the checkout being tested. Verify the imported
`tilelang` package and native libraries belong to that checkout, particularly
when using an independent worktree alongside an editable installation.

Host-only pass and integration checks (no NPU execution):

```bash
ASCEND_NPU_ARCH=dav-3510 python -m pytest \
  testing/ascend/analysis testing/ascend/transform testing/ascend/auto_schedule -q
```

The integration cases generate Ascend source and need the normal Ascend
compiler environment. Runtime reuse regressions are kept separately:

```bash
python -m pytest testing/ascend/language/test_tilelang_ascend_buffer_reuse.py -q
```
