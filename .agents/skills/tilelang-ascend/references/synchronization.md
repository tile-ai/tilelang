# Schedule and synchronize Ascend kernels

Contents: [scheduling modes](#choose-the-scheduling-mode),
[versions and owners](#buffer-versions-and-offsets),
[conditional epochs](#conditional-execution-and-epochs),
[manual multibuffering](#manual-multibuffering-limited-support),
[control flow](#control-flow-and-stage-shifting),
[conflict hints](#dependency-analysis-and-conflict-hints),
[storage protocol](#trace-the-complete-storage-protocol).

## Choose the scheduling mode

| Mode | Frontend expression | Compiler responsibility |
|---|---|---|
| Automatic | Ordinary kernel with `T.Pipelined` / `T.Persistent` as needed | Task order, core assignment, synchronization, and multibuffering |
| Constrained scheduling | `T.Stage` around complete tasks | Preserve manual stage and per-PIPE order constraints while retaining dependency analysis, sync, and multibuffering |
| Fully manual | Explicit core scopes, buffer slots, and flags with `TL_ENABLE_AUTO_SCHEDULE=False` | Skip the scheduling path; the program supplies the required storage and synchronization protocol |

Start with `examples/ascend/example_gemm.py` for automatic scheduling and
`examples/ascend/flash_attention/example_gqa_manual_schedule.py` for explicit
stages. Stage constraints and buffer rings are covered by
`testing/ascend/auto_schedule/test_schedule_units.py` and `test_sync_dependencies.py`.
Fully manual mixed protocols are covered in `testing/ascend/language/test_tilelang_ascend_nd2nz_scatter.py`.
The presence of low-level flags does not automatically select fully manual mode.

For fully manual kernels, use these APIs as part of a complete protocol:

| Purpose | API |
|---|---|
| Signal/wait between pipes on one core | `T.ascend_set_flag` / `T.ascend_wait_flag` |
| Signal/wait between cores | `T.ascend_cross_core_set_flag` / `T.ascend_cross_core_wait_flag` |
| Barrier for a selected pipe | `T.ascend_pipe_barrier` |

Match flag IDs and include initialization, reuse and final waits. Legacy
`T.ascend_get_buf(pipe, id)` / `T.ascend_rls_buf(pipe, id)` are also available for
explicit buffer handoffs; use them only with a complete manual protocol.

### Constrained scheduling details

`T.Stage` selects manual constraints independently for each scheduled child list
when any direct child task or loop carries a stage. All tasks in a stage scope
inherit its value; unwrapped siblings default to 0. Nested child lists without
staged children remain automatic. Each hardware PIPE follows the materialized
source order; different pipes remain free to overlap and reorder.

```python
for k in T.Pipelined(K_TILES, num_stages=2, annotations={"enable_offset": True}):
    with T.Stage(0):
        T.copy(A[k], a_l1)
        T.copy(B[k], b_l1)
    with T.Stage(1):
        T.gemm(a_l1, b_l1, accum, transpose_B=True)
```

Nonzero stages require `enable_offset=True`. `num_stages` bounds automatic
buffer-version selection, not frontend stage values. Do not combine body
`T.Stage` scopes with `order=` or `stage=` on `T.Pipelined`.

Wrap complete scheduler tasks: `with T.Stage(1), T.SimtVF(...):` has the correct
nesting. Never put `T.Stage` inside a Task, SimtVF, SimdVF, SBlock, or parallel
loop. Preserve the generated schedule when diagnosing an unexpected order.

### Tune performance with T.Stage

Use `T.Stage` when profiling and generated code suggest that task order or
insufficient overlap limits performance. Keep `TL_ENABLE_AUTO_SCHEDULE` enabled:
the frontend supplies stage and order constraints while the compiler manages
dependencies, synchronization, and buffer versions.

1. **Establish the baseline.** Save a correct automatically scheduled kernel,
   its generated source, and measured latency for the target cases. Initially
   keep tile sizes, core partitioning, VF code, and numerical operations fixed
   so the comparison isolates scheduling changes.
2. **Choose a scheduling hypothesis.** List the loop's tasks by PIPE and their
   input/output buffers. Identify a specific delay to reduce, such as a late
   input transfer or an output store blocking buffer reuse. Check whether the
   delay comes from required dependencies, limited storage, or task order.
   Stage changes cannot remove required work or dependencies.
3. **Express a small set of candidates.** Reorder independent same-PIPE tasks
   in source, or place producers and consumers in different `T.Stage` scopes
   with `enable_offset=True`. Account for every task in the affected child
   list, including siblings that default to stage 0. A load/VF/store pipeline
   can use stages 0/1/2; compare it with fewer stages for the actual workload.
   Wrap complete tasks, for example `with T.Stage(1), T.SimtVF(...):`.
4. **Check the materialized schedule and storage.** Inspect generated order,
   cross-iteration overlap, prologue/epilogue guards, waits, and buffer versions.
   Verify that the intended change actually occurred. Longer producer-consumer
   lifetimes may require more storage; check UB/L1/L0 capacity. Tune
   `num_stages` or explicit version counts separately when storage limits the
   candidate. More stages or versions do not necessarily improve throughput.
5. **Validate and measure.** Check correctness for affected tails, short loops,
   and repeated launches, then compare with the saved baseline using the same
   harness across the target cases. Use
   [performance measurement](../SKILL.md#12-performance-measurement-and-tuning) for measurement.
   Select by repeatable device-time improvement and report per-case regressions;
   a smaller modeled II alone is not a performance result. Retain automatic
   scheduling when manual constraints provide no measured benefit.

### Buffer versions and offsets

Eligibility, version count, indexing mode, and task stage are separate choices.
`T.annotate_buffer_versions` applies to a storage and all Buffer aliases sharing
it; aliases must agree on their count and mode.

| Annotation value | Meaning |
|---|---|
| `1` | Opt out of multibuffer eligibility, including explicit owner claims; retain ordinary data dependencies |
| Integer >= 2 | Fix the physical version count; select the indexing mode automatically |
| `(count, mode)` | Fix the count and mode; `(1, "auto")` retains eligibility and owner-exclusion dependencies with one physical version |
| `"auto"`, `"iteration"`, or `"counter"` | Select the mode and let the scheduler choose the count |

```python
T.annotate_buffer_versions(
    {
        carry: 1,  # Preserve ordinary dependencies on values carried across iterations.
        scratch: (1, "auto"),  # Keep owner-based scheduling with one physical slot.
        sparse: (2, "counter"),  # Two slots, advanced on active buffer epochs.
    }
)
```

The scheduler selects automatic counts from task lifetimes and memory limits,
with `num_stages` as the search upper bound. For an automatic loop without that
annotation the bound is one. A manual `T.Stage` child list without an explicit
`num_stages` uses `max_stage + 1`; an explicit bound does not limit stage numbers.
Fixed counts still require valid ownership and sufficient capacity. See
`testing/ascend/auto_schedule/test_buffer_version_annotations.py` for API coverage.

Automatic buffers do not need a manually added leading stage dimension. An
explicit leading dimension can instead be part of the application's layout;
do not remove it mechanically or accidentally request buffering twice.

With the current solver, `enable_offset=False` (the default) constrains each
PIPE's task start times to `max(start) - min(start) < II`. Raw solver stage
indices can still differ if that interval straddles an II boundary, but C++
reconstruction assigns automatic tasks stage 0 when offset is disabled.
`enable_offset=True` removes the span constraint and retains cross-iteration
stages. For example, the two Cube operations in attention may issue in a
staggered order instead of finishing both for iteration i before starting i+1.
Cross-PIPE overlap exists in both modes. Manual nonzero `T.Stage` values require
`enable_offset=True`; buffer-version selection remains a separate control.

### Owners and eligibility

An owner is a loop whose iteration contains a complete buffer epoch: its writes
and the reads consuming that value. Automatic inference finds the deepest
mutually non-nested loops covering all ordinary accesses, with writes before
reads. Separate sibling loops can own the same storage. If an outer loop loads
a tile consumed by an inner loop, that outer loop owns its versions: inner-loop synchronization must use the owner's iteration, not the
consumer's local iteration. If an inner loop writes rows and an outer task
consumes the whole buffer, the owner must likewise include both. See
`testing/ascend/auto_schedule/test_buffer_ownership.py` for ownership cases.

The current write-first heuristic is storage-granular: a write to one region
does not prove that every later-read region was overwritten. When untouched
regions must retain values across iterations, use `{buf: 1}` to opt out, or
choose an owner whose epoch overwrites the complete read footprint.

When conservative inference misses a valid owner, it can be declared explicitly:

```python
for i in T.Pipelined(N, num_stages=2, annotations={"multi_buffer_eligible": [buf]}):
    ...  # The complete write-before-read epoch for buf.
```

An explicit claim disables automatic owner inference for that storage across
the kernel. Declare every owner needed to cover its ordinary accesses; owners
must not nest. This annotation asserts eligibility, rather than proving a
complex access pattern safe. A bare `{buf: 1}` overrides explicit claims and
emits a warning once per storage.

Ordinary accesses outside all owners are rejected by version planning. An
owner-external `T.fill` can initialize all versions only when it can be rewritten
independently: its value, bounds, and guard must not read the target storage.
Broadcast fills of multi-version L1 storage are currently unsupported. Fill
broadcast initializes each slot; it does not propagate later updates between
slots.

### Indexing modes

| Mode | Clock and current restrictions |
|---|---|
| `iteration` | Flattened lexical loop iteration; requires exactly one owner. Explicit selection is allowed with guards, but irregular loop nests emit a warning. |
| `counter` | Counter advances for active buffer epochs and can span disjoint owners. Within each owner, all direct children accessing the same storage must have the same stage. |
| `auto` | Uses iteration for regular, unconditional single-owner epochs; selects counter when needed for conditional, irregular, or multiple-owner execution and compatible stages. |

Irregular paths include outer-dependent loop bounds, conditionally entered
loops, and loop breaks. If a single owner's accesses span stages, `auto` can
fall back to iteration; explicit counter mode rejects that pattern. Multiple
owners require counter mode, so a cross-stage owner cannot use that fallback.
The restriction concerns each storage's accesses, not every task in the loop:
`enable_offset=True` alone does not make counter mode invalid. Inspect stages
before combining `(2, "counter")` with a load/VF/store split across stages.

### Conditional execution and epochs

For a loop whose load, VF, and store share `if active(i)`, lexical iteration i
and the count of executed buffer epochs differ when the condition is false:

| Mode | Slot selection | Synchronization behavior |
|---|---|---|
| `iteration` | Lexical iteration modulo the version count | Buffer-version handshakes follow lexical epochs, including iterations that skip guarded data work |
| `counter` | Active-epoch counter modulo the version count | The owner clock and associated handshakes follow the prepared active guard; skipped epochs do not advance the clock |

For more complex guards, the active guard is derived from storage accesses; it
need not be identical to one source-level `if`. A late-defined or cross-stage
guard cannot always be used as a loop-local active guard. Thus the presence of
an `if` alone does not guarantee counter selection.

Both modes need producer-to-consumer readiness and consumer-to-next-writer
release. Generated code may distribute guards across individual tasks and may
hoist or combine synchronization. Check initialization and final drain as well
as steady-state slot indexing; an isolated set/wait sketch omits those paths.

### Manual multibuffering: limited support

**Manual multibuffering is not recommended for general use.** The current
analysis supports simple patterns; complex manually indexed accesses and
control-flow combinations may be analyzed incorrectly and produce incorrect
synchronization. Prefer compiler-managed multibuffering.

For understanding existing kernels, `T.annotate_manual_multi_buffer(buf)`
declares already-indexed physical slots, with the ring size inferred from a
constant leading dimension. `T.annotate_manual_multi_buffer({flat: 3})` supplies
the ring size explicitly for storage without a dedicated leading version axis.
The user writes every slot index; the annotation does not reshape or reallocate
storage, or validate an arbitrary manual buffering protocol. AutoSchedule can
remain enabled to analyze dependencies and insert synchronization. This differs
from disabling the entire scheduling path for a fully manual flag kernel.
Guarded manual-ring synchronization is checked in
`testing/ascend/auto_schedule/test_sync_dependencies.py`; this does not establish
support for arbitrary manual protocols.

## Control flow and stage shifting

Complex `if` conditions and memory-reading loop bounds are first captured in
Bind variables, preserving their evaluation before the body can mutate the
memory. Schedule-unit construction then represents branch work as guarded
tasks; a condition already expressed as a Var needs no extra snapshot.

With offsets, bindings needed by several stages may be cloned into those
stages. This can repeat expensive GM reads or other expressions. Do not assume
an immutable variable name makes its bound memory load immutable. A
`T.alloc_var` alternative creates mutable `local.var` storage with dependencies;
it is not automatically multibuffered and can constrain stage offsets.

The current pipeline accepts nested while/for control flow by normalizing while
loops for scheduling and restoring them afterward. An inner `T.Pipelined` loop
can overlap work; the synthetic while loop itself is not an arbitrary pipeline
offset boundary. `testing/ascend/transform/test_ascend_restore_while_loops.py`
checks restored control flow, and `testing/ascend/auto_schedule/test_schedule_integration.py`
checks nested-loop exits across core splitting. Preserve
`T.loop_break` ordering and examine zero-trip loops and early exits when changing
buffer clocks. A one-trip loop can still be an owner or synchronization boundary;
do not remove it solely because its extent is one.

## Dependency analysis and conflict hints

Dependency analysis uses read/write regions with enclosing loop, guard, assume,
and binding context, both within and across iterations. Mutable loads can be
independent unknowns at different access points. Views/reshapes sharing storage
but using different Buffer objects retain conservative dependencies when their
logical coordinates cannot be compared directly.

Use hints only when the program's data contract establishes the stated relation.
For example, if `perm` is known to be a permutation and each iteration writes
one distinct row:

```python
for i in T.serial(N):
    T.assume_no_conflict(out[perm[i], 0:TILE], cross=True, level=0)
    T.copy(tile, out[perm[i], 0:TILE])
```

Here `level=0` selects the outermost enclosing loop. The region must match the
actual access footprint: a single indexed element does not describe a tile.

| Argument | Scope |
|---|---|
| `cross=True` / `False` / `None` | Cross-iteration only / same-iteration only / both |
| `level=0, 1, ...` | One common enclosing loop, counted from outermost |
| `level=-1` | Root sequence only |
| `level=None` | Root sequence and every common enclosing loop |
| `group="name"` | Pair exactly two one-region declarations in different scopes |

A bare Buffer matches its whole storage, so `T.assume_no_conflict(out)` can
suppress much more than one redundant output barrier. The hint is a user
assertion and has no runtime check. `T.assume_conflict` forces matching accesses
to be treated as conflicting while retaining the usual RAW/WAR/WAW rules;
read-only pairs still have no dependency. If contradictory hints match, conflict
wins. See `tilelang/ascend/language/schedule_hint.py` for the API.

## Trace the complete storage protocol

For a suspicious access or wait, build a table with:

- Producer and consumer task, core, PIPE, and issue order.
- Physical allocation, layout, offset, accessed byte range, and alias relation.
- Owning loop, logical iteration, version counter, selected slot, and guard.
- Readiness signal, reuse signal, flag identity, and initialization/drain path.

Follow both directions: the consumer must wait for produced data, and the next
writer must wait until previous readers have released the reused storage.
Cover RAW, WAR, and WAW hazards according to actual overlapping regions.
There is not necessarily one flag pair per dependency or a barrier for every
same-PIPE pair: implicit pipe ordering and existing completion relations can
make explicit synchronization redundant. A logical buffer name, task token,
or equal shape alone does not describe physical overlap. Different staging
sizes or layouts can reuse overlapping storage.

Keep task stages, dependency distance, and storage slot count distinct. A
single physical slot does not automatically imply that every dependence has
distance one. Conversely, a distance annotation alone does not establish that a
particular flag protocol can represent the required outstanding operations.

## Allocation ownership

Treat each buffer version as live until every consumer finishes. Different
Buffer views can alias the same physical allocation; shape or source order
alone does not prove that reuse is safe. Keep slot selection, producer/consumer
guards, and initialization/drain paths consistent when changing manual kernels.
For an unexpected allocation or overlap, retain the generated source and
lowering trace with a minimal case; see [lowering traces](../../../../docs/tools/lower_trace.md).

## Audit flags through control flow

Inspect generated per-core code after schedule and sync lowering. Source order
across asynchronous pipes does not by itself prove completion order. For each
wait, find the matching set on every reachable path and check the correct
producer/consumer PIPE, event kind, owner, iteration, and flag reuse.

Pay particular attention to:

1. Dynamic loops with zero trips: an internal set can be skipped while an
   external drain wait still executes.
2. Conditional producers and unconditional consumers, or guards changed by
   stage shifting and prologue/epilogue generation.
3. Nested-loop ownership and several producers/readers sharing an allocation.
4. Initialization, steady-state reuse, and the final outstanding operations.
5. Cross-core fan-out/fan-in, active AIV count, and which sub-core owns a signal.

Do not treat one hardware flag as an unbounded counting semaphore. Check the
specific intra-core or cross-core primitive and target SDK semantics before
assuming signals can accumulate or waits can be paired arbitrarily far apart.
Likewise, a scalar special-register dependency needs the correct issuing PIPE;
an entry in the dependency graph is only part of the protocol.

Incorrect latency estimates may expose a scheduling or protocol defect; timing
alone must not stand in for a required synchronization relation. Adding waits
until one case passes can hide the cause or introduce deadlocks. Derive the
necessary edge and remove only synchronization proved redundant under the
complete ownership, guard, and storage model.

## UnitFlag and tail lifecycles

For kernels using UnitFlag synchronization between GEMM and output transfers,
check the target SDK's pairing and reset requirements. Preserve the matching
buffer slots, accessed regions, and guards when changing tile sizes or tails.
Check zero-size operation behavior for each instruction involved.

Validate tail handling across repeated launches and buffer reuse. Check that
the declared access regions cover the generated instructions' physical
footprint and that each invocation completes the required synchronization.

## Validate a kernel change

Start with a small supported kernel and check all defined outputs. Include the
tail, guard, repeated launch, or slot-reuse case affected by the change. Preserve
the original failing program before changing tiling, barriers or version counts.
Extract a standalone case when application code hides the failure. Use
[performance measurement](../SKILL.md#12-performance-measurement-and-tuning)
for timing after correctness is established.
