# Review Load-Balancing Rules

## Purpose

This document standardizes per-group rule capacity, cross-reference merge capacity, concurrency limits, and subagent isolation. `steps/common.plan-design.md` reads these rules to generate the review plan. File reviews continuously backfill available slots according to plan priority, while subsequent PR reviews retain the original wave-by-wave execution model.

This document neither selects applicable rules nor performs code reviews.

## Per-Group Rule Capacity

Each `references/*.md` file declares the maximum number of rules that one subagent may handle in a single task through the `<review_load>` block in its header:

```text
<review_load>
General review subagent rule capacity limit: N
</review_load>
```

`N` is the maximum total number of rules that one group generated from that reference may contain:

- A smaller capacity indicates that the rules require more detailed analysis, so fewer are assigned to each group;
- A larger capacity indicates that the rules are better suited to batch inspection, so more may be assigned to each group;
- Rules within a group must still be reviewed independently, one by one. The capacity limit never permits merging or omitting review steps.

To adjust a capacity, modify only the `<review_load>` block in the corresponding reference. Do not duplicate each reference's specific value in this document.

If the capacity declaration is missing, is not a positive integer, or cannot be parsed, `plan-design` must stop grouping and report the specific file. It must not guess a default value.

## Cross-Reference Merge Capacity

Rules from different references may be merged only when their applicable targets, code scope, and issue root cause are the same. The merged capacity is the minimum capacity among all source references:

```text
merged group capacity = min(capacity of each reference represented in the group)
```

The capacity limits the total number of rules assigned to the agent; it is not calculated separately for each reference. For example, if two references both have a capacity of 3, the merged group may contain at most 3 rules in total, not 3 from each reference for a total of 6.

The following rules must not be mixed across groups:

- Red-line rules from `tilelang-red-line.md` must form independent groups and must not be merged with ordinary performance, domain, or documentation rules;
- Markdown rules must not be merged with Python or TileLang code rules;
- Host-only rules must not be merged with kernel-only rules;
- Rules must not be merged when their formal review file scopes differ and the same code evidence cannot validate them.

## Reference Handling

Every reference in the target skill that currently contains valid `<applicability>` and `<review_load>` declarations enters the ordinary rule-grouping process.

`references/doc-style.md` contains ordinary Markdown review rules, has a side classification of `N/A`, and enters general rule grouping according to its declared capacity.

Auxiliary documents without a valid `<applicability>` declaration cannot independently generate review groups. They are read as supporting material only when explicitly referenced by a rule that has already matched.

## Concurrency Limit and Continuous Backfilling for File Reviews

At any time, the total number of running format-check and rule subagents must not exceed:

```text
min(number of concurrency slots available to this review, 6)
```

The “number of concurrency slots available to this review” is the number of slots available for this review when Stage 1 begins. Regardless of how many slots the environment provides, at most six review subagents may run simultaneously. Before every dispatch, the workflow must also verify that the environment currently has a free slot.

File reviews execute in the following order:

1. Flatten the planned waves into a dispatch queue that preserves priority and stable ordering. The initial format task occupies one slot, and rule groups fill the remaining slots.
2. Whenever a task completes, first verify its expected YAML. If it is missing, use the newly available slot to redispatch that task once before dispatching anything else.
3. After confirming a result, immediately dispatch the next group in the queue into the available slot without waiting for other running tasks. If the YAML remains missing after redispatch, stop dispatching new groups.
4. After all tasks return, perform a global completeness check against the plan's full expected result set. Only after it passes may the collector be closed and the reporting stage begin.

Every dispatch uses a new, isolated subagent. A subagent that completed a task must not be reused for another group or a redispatch. This prevents rules, group identifiers, evidence, and YAML submission parameters from contaminating subsequent rounds.

## Dispatch Priority

Subject to both the per-group capacity and concurrent-running limits, schedule groups in the following order:

1. Red-line rules from `tilelang-red-line.md`;
2. Review categories explicitly specified by the user through `scope_hint`;
3. Domain rules whose trigger conditions matched, such as MoE, TopK, or quantization;
4. Other rules that are enabled by default and applicable.

Within the same priority, schedule groups with more direct risk evidence and more explicit file scopes first. If they cannot be distinguished, retain the stable order generated by `plan-design`.

“Highest priority” determines dispatch order only. It does not permit exceeding the available slots, the six-agent concurrency limit, or per-group capacity. Priority does not preempt running tasks; the highest-priority queued group is dispatched before lower-priority queued groups.

When a Markdown-only input matches only documentation rules, `doc-style.md` enters the initial task set directly and is not delayed by the code-rule priorities above.

## File Reviews and PR Reviews

File reviews and subsequent PR reviews share these invariants:

- Per-group capacity comes from the reference's own declaration;
- Cross-reference merge capacity is the minimum source capacity;
- Red-line rules, rules for different sides, and rules for different file types must not be merged;
- At most six subagents may run at any time, subject to the number of slots actually available;
- Every dispatch uses a new subagent; a completed task's subagent is never reused.

File reviews follow this document's continuous-backfilling rules. Subsequent PR reviews retain the original process of waiting wave by wave and checking completeness; this change does not modify their processing rules.

If a PR review first partitions groups by file scope, it must continue to apply this document's rule-capacity and wave rules within each file group. File grouping must not allow the concurrency limit to be exceeded.
