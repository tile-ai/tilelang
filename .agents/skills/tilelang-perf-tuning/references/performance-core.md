# Performance Core

This file is used in every tuning engagement and keeps only the general rules that can change candidate ranking or acceptance conclusions.

## Fact Priority

1. The current operator, public wrapper/dispatch, tests, and locked cases;
2. This round's compilation, precision, and device-side profiling;
3. The actually imported TileLang source, the target backend lowering, and running implementations in the same repository;
4. Historical candidates, data from other devices, and design material.

Lower-priority material never overrides higher-priority measurements. Locate the actual TileLang:

```bash
python -c 'import pathlib,tilelang; print(pathlib.Path(tilelang.__file__).resolve()); print(getattr(tilelang,"__version__","unknown"))'
```

APIs, parameters, layouts, synchronization, L2 policies, and multi-version capabilities must be confirmed in that source, its lowering, same-version examples, or current compilation output. When no basis can be found, mark the item as pending verification or `DESIGN_ONLY`; never implement from memory.

## Baseline Model

Record per case:

```text
semantic path → dispatch → primfunc/kernel → task mapping → tile loop → execution domain
```

Also establish five cost categories:

- **Logical work**: dimensions each output depends on, duplicated computation, common subexpressions, loop invariants;
- **Addressing/control**: integer multiply/divide/modulo, bit operations, casts, branches, indexing, and per-thread live state;
- **Data movement**: effective and physical reads/writes across GM/UB/L1/L0, intermediate materialization, repeated transfers, synchronization;
- **Parallel structure**: number of independent tasks, actual core count, longest/shortest per-core work, full waves, and tail waves;
- **Numeric work**: Vector/Cube operations, reduction order, accumulation dtype, and approximate instructions.

A drop in logical work is only an upper bound on the gain; it must not be reported as a measured speedup.

### Resource Ledger

Physical capacities come from current device evidence; the compiler's effective capacities are confirmed from the current lowering or compilation output. Never inherit fixed safety caps from historical machines.

```text
explicit footprint = Σ(product(padded_shape) × dtype_bytes × versions)
total footprint = explicit + resident data + proven temporaries + justified safety margin
remaining capacity = effective upper bound of the current lowering - total footprint
```

Recompute after adding or removing `SimtVF`, buffer versions, padding, LUT/index, temporary layouts, or execution domains. GM tails access valid elements only, while UB is still allocated by the real SIMD/DMA-reachable footprint.

## Bottleneck Attribution

Pipe ratios are clues, not source-level root causes. Concrete attribution combines at least one of source-level work, generated IR/instructions, memory traffic, timelines, or isolated experiments.

| Type | Evidence | Preferred candidates |
|---|---|---|
| GM/MTE | Heavy GM round trips, small bursts, actual time close to or far above the transfer lower bound | Fusion, contiguous transfers, residency, L2 policy, less materialization |
| Vector | VEC time dominates and the compute/reordering work explains it | Register reuse, fewer casts/traversals, SIMD dataflow, fast math |
| Scalar/scheduling | Dynamic indexing, branches, loop control, or fixed segments dominate | Hoisting invariants, affine tasks, merging work, compile-time generic specialization |
| Parallelism | Fewer tasks than cores, or a visible tail imbalance | Inner flatten, adjust core count/granularity, makespan mapping |
| Cube | Cube time dominates or MMAD utilization is low | M/N/K tiling, L1/L0C, K pipelining, output paths |
| Launch | Kernels are short and fixed launch/multi-kernel overhead dominates | Merge kernels/dispatch while keeping the whole-chain methodology |
| Mixed | Different cases or stages have different dominant bottlenecks | Branch by general structure and adjudicate per case |

Logical bandwidth is source bytes divided by elapsed time; it is not physical HBM utilization. L2 hits, atomic read-modify-write, repeated consumption, and write allocation change physical traffic.

## Candidate Ranking

Check in the following order, but let the current bottleneck and the gain upper bound decide the actual ranking:

1. Eliminate cross-dimension duplicated computation, repeated reductions, or redundant kernels;
2. Eliminate GM round trips, full intermediate materialization, and repeated reads;
3. Resident weights, states, LUTs, indices, or broadcast parameters;
4. Flatten independent tasks and improve inter-core tail balance and fixed-cost amortization;
5. Search for grounded tile/core choices and physical Vector/Cube routes;
6. Test pipelining when both MTE and Compute show visible time and each core has enough iterations;
7. Test local arithmetic/lowering formulations only after generated-instruction differences appear.

When two consecutive candidates gain less than about 3% while a clear gap to the target remains, stop equivalent micro-tuning and re-examine algorithmic reuse, dataflow, live state, task mapping, and historically verified structures. When more than 1.2× is needed, do not keep prioritizing candidates that can only explain low-single-digit gains, unless one is a necessary prerequisite of a high-yield route.

## Execution Domain and Numerics

- Regular contiguous elementwise work generally prefers `SimdVF`; discrete indexing, complex branching, atomics, or thread-dependent semantics use `SimtVF`. Verified paths in this repository take precedence over abstract preference.
- `T.Kernel` keeps a one-dimensional block grid; multi-axis tasks are expressed through scheduling structures.
- fp16/bf16 reductions, norms, softmax statistics, and GEMM accumulation start from fp32 state by default.
- Internal lower precision, HF32, hardware division, fast math, and approximate instructions may be tried experimentally, but must not change the reference, the original tolerances, or the public semantics; keep them only after both targeted precision checks and the full regression pass.
- Tail blocks must not read uninitialized lanes; full-register accesses must be zeroed, padded, clamped, or correctly masked.
- JIT/host branching may rely only on general properties: dtype, alignment, capacity, task count, contiguity, tails, or hardware limits; never speculate on public case IDs or exact test shapes.

## Historical Reuse

First check the operator's git history, existing tuning directories, and same-repository implementations. When reusing, verify the current interfaces and cases, TileLang/backend/device versions, data residency and layouts, intermediate materialization and synchronization, tiles/cores/stages, tails, launcher parameters, the original performance methodology, and source SHAs.

A compatible historical candidate still only enters the candidate pool; it never becomes B* directly. The presence of template files, identical names, or a historical "verified" label never replaces current re-validation.

## Precision and Performance Gates

- Each candidate first verifies the targeted shapes it actually activates; the existing relevant full test suite runs only once, at the end.
- Never widen tolerances, lower reference precision, skip failing shapes, or call a subset pass a full pass.
- Distinguish numeric failures, compilation failures, kernel execution failures, profiler failures, and deliberate `NotImplementedError`.
- Keep inputs, dtypes, output semantics, device, frequency, warmup, repeats, concurrency, and timing level identical between baseline and candidates.
- Promotion reports the geometric mean or the user-designated metric, the worst case, and every regressed case; an average must not hide an individual case.
- Without a valid measurement on the current device, call it a candidate or a design; never claim a gain.
