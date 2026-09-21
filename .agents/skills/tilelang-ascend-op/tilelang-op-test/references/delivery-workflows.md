# ST Delivery Workflows

Three operator-maturity scenarios share a single `tilelang-op-test` entry point. The scenario determines which evidence to start from and which maintainable assets must be delivered, but it does not lower the credibility threshold.

## Delivery from Scratch

Use this workflow to create a new operator from requirements, a prototype, a design, or a user description.

Begin ST work before implementing the kernel: freeze the public contract, build an independent PyTorch/CPU reference, and design an acceptance matrix based on the supported scope. Do not derive semantics from a kernel that does not yet exist. Derive the operator implementation and ST independently from the same contract.

Deliverables include:

- A contract inventory covering formulas, inputs/outputs, dtypes, shapes, layouts, optional dependencies, exceptions, side effects, accuracy, and backend scope;
- An independently reviewable reference, together with small exact examples or invariants that validate the reference itself;
- A coverage assessment across all ten dimensions, with explicit rationale for every `not_applicable` status;
- Pytest cases that execute through the planned public API and use stable parameter IDs;
- Test collection and execution evidence produced after the operator implementation is complete;
- A final `PASS`, `FAIL`, or `NOT_VERIFIED` report.

If the public API or kernel does not yet exist, tests may be designed and written against the agreed interface, but the report status may state only that they are designed or implemented, not executed. When the implementation is missing, classify the delivery as `NOT_VERIFIED`, not as passing.

## Existing-Test Remediation

Use this workflow when the repository already contains the operator, tests, or both. Discover existing tests, reconstruct the contract from more reliable sources, review credibility, identify missing dimensions, add the smallest set of discriminating cases, and then run targeted tests and the relevant regression suite.

Do not equate the existence of historical pytest cases with coverage. Preserve credible existing cases. Strengthen weak assertions only after the expected behavior has been established. Identify contract gaps explicitly instead of treating the current implementation behavior as the correctness standard.

## Delivery Acceptance

Use this workflow on a fixed commit after completing a new feature, bug fix, backend migration, or performance optimization. Treat committed pytest cases, references, and specifications as maintainable test assets, and generate an acceptance report tied to the current run.

The acceptance report must tie its conclusions to the commit and worktree state, target backend/device, exact commands and nodeids, collection/execution/result counts, random-seed strategy, coverage gaps, and failure classification.

- `PASS`: Every mandatory and applicable dimension is supported by credible cases executed on the specified target, with no unresolved contract gaps.
- `FAIL`: Any mandatory correctness, invalid-input rejection, state, gradient, compilation/runtime, or other delivery requirement is violated.
- `NOT_VERIFIED`: Evidence is insufficient because tests were not collected or executed, every relevant case was skipped, a required environment was unavailable, or a required contract remains unclear.

A compilation or device failure may block delivery while numerical correctness remains unknown. State both facts in the report.

## Maintainable Assets and Per-Run Artifacts

Maintainable repository assets include contracts/specifications, independent references, pytest cases, and stable test configuration. Per-run artifacts include collection results, logs/JUnit output, environment data, and acceptance reports. A complete operator delivery requires both: tests make results reproducible, while reports demonstrate what actually occurred for the current version.

Performance and memory targets belong to ST acceptance only when the contract explicitly requires them. Reuse the profiling skill for measurements, and rerun correctness ST after performance optimization.
