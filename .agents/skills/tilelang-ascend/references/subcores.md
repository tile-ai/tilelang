# Independent AIV execution

Use `T.MixedKernel(..., sids=2)` with automatic scheduling enabled when AIV0
and AIV1 need different work. `T.Stage` can constrain this schedule without
disabling dependency analysis, synchronization, or buffer versioning. For
explicit core scopes and flags with scheduling disabled, follow the
[fully manual protocol](synchronization.md#choose-the-scheduling-mode).

## Choose roles and storage

`sid` is the AIV sub-core ID, not a SIMT thread ID. Both AIVs share one Cube
stream. Keep common GEMM and input loads outside `sid` guards when they should
execute once for the pair. `sids=1` selects only AIV0; `sids=2` retains both
AIVs even for a program with no Cube work.

| Pattern | Expression and ownership |
|---|---|
| Fixed roles | `if sid == 0` prepares an input through UB→L1; `if sid == 1` handles the L0C→UB epilogue |
| Disjoint input partitions | Each AIV processes its own slice and copies it into a disjoint region of the common L1 tile |
| Alternating roles | A guard such as `sid == i % 2` chooses the AIV for iteration `i`; the selected AIV can own a complete tile |

UB and sub-core-dependent scalar state are local to each AIV. The compiler
projects their storage identities, views, layouts, and version annotations;
do not add a leading `sid` dimension merely to separate the two UB address
spaces. L1, L0, and GM storage remain common, including aliases of that
storage. A private UB buffer does not make its eventual GM/L1 destination
private. Check per-AIV UB capacity and the common Cube buffer capacity.

For example, given `A[STEPS, 32, 32]` and `B[32, 32]` in bfloat16 and
`C[STEPS, 32, 32]` in float32, alternate complete output tiles:

```python
with T.MixedKernel(1, sids=2) as (_, sid):
    a_l1 = T.alloc_l1((32, 32), "bfloat16")
    b_l1 = T.alloc_l1((32, 32), "bfloat16")
    accum = T.alloc_l0c((32, 32), "float32")
    output = T.alloc_shared((32, 32), "float32")
    T.annotate_buffer_versions({output: (2, "counter")})
    T.copy(B, b_l1)
    for i in T.Pipelined(STEPS, num_stages=2):
        T.copy(A[i, :, :], a_l1)
        T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
        if sid == i % 2:
            T.copy(accum, output)
            T.copy(output, C[i, :, :])
```

Each AIV's counter follows its own active buffer epochs. Keep all accesses to
the same counter-indexed storage at one stage within each owner; explicit
`counter` mode still has the [usual versioning restrictions](synchronization.md#indexing-modes).
The output buffer above holds a full tile because only one AIV drains each
iteration. For a cooperative half-tile epilogue, use the `T.dual_copy` shape
contract instead.

## Route L0C-to-UB copies

L0C→UB copies execute on AIC/Fixpipe even when written under an AIV guard.
The guard selects the destination AIV; it does not move Fixpipe execution onto
the Vector core.

| Form | Routing |
|---|---|
| `T.copy(accum, ub)` | In a projected program, implicit routing targets each participating AIV; a `sid` guard can select just one |
| `T.copy(accum, ub, sub_blockid=target)` | Selects the explicit destination together with the enclosing guards |
| `T.dual_copy(accum, half_ub)` | One hardware dual-destination instruction with a common UB offset and generation; its control path must be uniform across sub-cores |

Use guarded ordinary copies for independently selected destinations. Do not
put hardware `dual_copy` under a `sid`-dependent guard or loop, or replace it
mechanically with two ordinary copies: their storage and synchronization
contracts differ. Software dual copies on GM↔UB and UB→L1 remain per-AIV
copies with `sid`-indexed addresses.

An explicit destination must agree with its guard. For example,
`if sid == 0: T.copy(accum, ub, sub_blockid=1)` can eliminate the copy during
projection; there is currently no dedicated conflict diagnostic. Guard the
matching consumer as well when only one AIV receives data.

Put projected Fixpipe copies with different destinations in separate
`T.Task` scopes. Implicit routing or a route computed inside the task can
produce multiple AIC executions. If the task also has an external side effect
such as an L0C→GM store, atomic accumulation, or an effectful external call,
the compiler rejects projection that could repeat it. Split those effects
into separate tasks and snapshot a dynamic route before the Fixpipe task.
A constant route or enclosing guard that leaves only one task replica does
not have this replay problem.

Within `T.PerCoreTask`, preserve the
[block-index dispatch contract](../SKILL.md#25-per-core-task-contract).
Each candidate keeps its own destination and guard. Projected dispatch must
use immutable values available before the group. Explicit synchronization
inside a group with implicit or task-local Fixpipe routing is rejected when
projection could change its participation or execution count. Move that
synchronization outside the group, or give each candidate an explicit
`sub_blockid` computed before the group.

Explicit UnitFlag controls remain caller-managed; see
[UnitFlag and tail lifecycles](synchronization.md#unitflag-and-tail-lifecycles)
before projecting accumulator drains.

## Shared dependencies and loop exits

Direct AIV0↔AIV1 synchronization is not supported. Conflicting accesses to
common GM or L1 storage must already be ordered by real AIC data dependencies;
otherwise the remaining dependency is rejected after dependency optimization.
An unrelated Cube operation between two accesses does not order them.
Alternating writers with `sid == i % 2` also does not by itself make reuse of
one shared destination safe. Prefer disjoint destinations or a supported
dataflow through AIC; do not suppress a real hazard with conflict hints.

The two AIVs may have different loop bounds, mutable counters, and early exits.
An exit must also reach the Cube stream when that stream consumes the
corresponding sub-core's values, including uses after the loop. Independent
exits that ambiguously control common Cube work or Cube work from both
sub-cores are rejected. Put the shared Cube work in a separate loop or use a
common exit condition.

## Examples and validation

`testing/ascend/auto_schedule/test_schedule_integration.py` contains complete
kernels for fixed roles, disjoint input halves, alternating epilogues, changing
roles, distinct loop bounds, and early exits. `test_core_resolution.py` covers
routing, buffer views, and UnitFlag projection; `test_tasks.py` covers
`T.PerCoreTask`, and `test_cross_core_flags.py` covers supported dependency
ordering and rejected direct AIV synchronization.

When adapting a kernel, inspect each AIV's guards, UB versions, GM/L1 write
regions, and Fixpipe destinations in the generated source or lowering trace.
Validate short and odd iteration counts, inactive branches, repeated buffer
reuse, and both requested AIV counts when the kernel supports them.
