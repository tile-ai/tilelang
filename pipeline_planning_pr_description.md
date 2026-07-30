# Rebuild automatic pipeline planning around a dependency DAG

## Summary

This PR replaces the producer-closure heuristics in `PipelinePlanning` with a single source-ordered dependency DAG and a weighted longest-path schedule.  The main goal is correctness: every inferred `stage` and `order` must preserve all
visible buffer hazards, scalar def-use dependencies, and opaque async-control ordering before `InjectSoftwarePipeline` realizes the schedule.

The change directly fixes:

- [#2720](https://github.com/tile-ai/tilelang/issues/2720), where a dependency chain crossing buffer and scalar edges produced an invalid stage assignment;
- [#1792](https://github.com/tile-ai/tilelang/issues/1792), where a copy index loaded from a `local.var` buffer but the address-computation dependency was omitted from planning;
- [#2668](https://github.com/tile-ai/tilelang/issues/2668), where a shared-buffer read followed by an overwrite was incorrectly pipelined across a loop-carried WAR lifecycle and silently miscomputed.

Tests on [#2595](https://github.com/tile-ai/tilelang/issues/2595) exposed a bug in pipeline-rejection. This is not fixed in this PR.

## Drawbacks with the previous planner

Statements were only assigned to two stages, `0` and `num_stages`, making in-stage dependencies insufficient. Furthermore, some dependencies were not correctly resolved (for example, the three issues fixed here).

## New design

### 1. One DAG for all top-level pipeline statements

Each normalized top-level statement in a `T.Pipelined` body becomes one node. Edges always point from a smaller source index to a larger source index, so source order is a topological order by construction.

The graph contains:

- buffer RAW edges (write -> later read);
- buffer WAR edges (read -> later write);
- buffer WAW edges (write -> later write);
- scalar def-use edges;
- conservative edges around opaque state-changing control calls.

Buffer regions belonging to the same `Buffer` are compared conservatively (this preserves issue [#2759](https://github.com/tile-ai/tilelang/issues/2759)). Distinct `Buffer` views that share one data variable are rejected inside `T.Pipelined`, because downstream injection and buffer versioning track dependencies by `Buffer` identity.
The diagnostic asks users to use one consistent `Buffer` view throughout the pipeline or move the aliased access outside `T.Pipelined`.

### 2. Weighted longest-path stage assignment

Logical levels are computed in one forward pass over the DAG.  Only an unconditional global-to-shared copy advances a successor by one level.  Scalar bindings, ordinary synchronous computation, control glue, WGMMA, and TCGen operations have zero scheduling weight. This intentionally models the overlap currently supported by the injector: global-memory transfer can be issued ahead of its consumer.  WGMMA / tcgen05 are treated as synchronized operations due to their complex synchronization methods (pipelines are still available with handwritten annotations).

The logical levels are scaled into the requested stage range.  If the longest dependency chain is deeper than the requested distance, adjacent logical levels are merged instead of violating an edge.

### 3. Strong same-stage constraints

Two dependency classes cannot safely cross skewed loop iterations:

- A materialized scalar Bind and its users must share a stage because the injector versions buffers, not arbitrary registers.
- A read-before-write WAR pair must share a stage.  It describes a cyclic buffer lifecycle: consume the value entering this iteration, then overwrite it for the next iteration.  A simple `stage(read) <= stage(write)` inequality is insufficient because different stages execute different logical iterations.

These equalities and ordinary DAG inequalities are propagated to a fixed point.

### 4. Terminal stages and periodic-boundary retiming

Statements with no in-pipeline successor are initially placed in the final consumer stage.  For an ordinary inferred pipeline, `num_stages` describes the producer/consumer distance, so the provisional range may be `[0, num_stages]`.  A consumer-only endpoint can be retimed once to `num_stages - 1`, preserving the previous buffer-saving boundary rotation.

The retiming is disabled when the final stage still produces an internal buffer value.  It is also disabled for manual warp specialization.

This feature is applied in old `T.pipeline` implementation and therefore preserved. The switch is defaulted to `True` to preserve compatibility.

However, enabling this feature may make the meaning of `num_stages` ambiguous and may also hurt performance with small values. Therefore, an extra parameter controls this behavior. You may disable it with:

```python
for k in T.Pipelined(
    extent, num_stages=3, compact_terminal_stage=False
):
    ...
```

Manual `T.ws` uses `[0, num_stages - 1]` directly because `num_stages` is also the physical ring-buffer/barrier slot count.  Allowing `stage == num_stages` would alias stage 0 after modulo indexing and could use the wrong barrier phase.

## Conditional control flow and warp specialization

`IfStmtBinding` splits a no-`else` multi-statement body into separately guarded top-level statements where legal.  The planner sees the buffer/scalar accesses inside each guarded statement, but a conditionally executed copy is not marked
as an implicit async producer.  An `if/else` remains an atomic scheduling unit and its branch accesses are analyzed conservatively.

An outer runtime guard around the whole pipeline remains outside the generated pipeline and does not make each inner statement conditional.

Manual `T.ws` scopes keep their warp roles.  Pipeline planning only assigns their time stage/order and uses the restricted physical ring range described above.  Automatic producer/consumer WS is a separate earlier pass and is not redesigned here.

## Explicit PTX async limitation

This PR **DOES NOT** make manually written `T.ptx_cp_async` pipelines fully versionable.

`PipelinePlanning` can understand `tl.access_ptr` read/write masks, but `InjectSoftwarePipeline` does not currently multi-version buffers referenced through `tl.access_ptr`.  If an explicit PTX copy/control chain is split from its consumer, the PTX instruction can keep writing the original shared slot while the consumer reads a versioned slot.

This was discovered while running the relevant unit test `testing/python/issue/test_tilelang_issue_tma_no_ws.py:203`. An issue will be raised separately.

To handle this and similar situations, this PR treats PTX statements conservatively, keeping relevant consumers in the same stage. In addition, a warning is raised when PTX statements appear in `T.Pipelined`.

## Compatibility

The new implementation generates more efficient code and is more decoupled.

Compatibility is kept in the largest extent:
- Most `T.Pipelined` usages in unit tests produce exactly the same ordering as the older version.
- On only one test with `tcgen05` - gemm pipeline does the 2 implementations differ - the code generated is reduced from a 3-buffer pipeline to a 2-buffer pipeline because of better analysis in dependencies.

## Tests

Added relavent unit tests. All related tests passed.
