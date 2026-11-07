# Ascend Hardware Resources and Effective Compiler Capacity Discovery

Tiling, residency, and multiversion pipelines must distinguish physical hardware capacity, execution-domain reservations in the current lowering, and explicit kernel allocations. Do not inherit a "safe limit" from historical machines or a single operator.

## 1. Record the Current Hardware

Reuse the caller-provided `full_soc`, `npu_arch`, and complete `evidence`. If evidence is missing or incomplete, or if the device or configuration has changed, load the `npu-arch` Skill by name and use its bundled detection script to reacquire it. Accept only consistent evidence for the Ascend950PR/DT family with `npu_arch=3510`. Stop and report if the Skill is missing, detection fails, or evidence conflicts.

Record the complete model, architecture, CANN version, Vector/Cube core counts, and physical UB/L1/L0 capacities, together with the source of each value. Prefer actual-device returns for runtime-queryable items. Use complete `full_soc` to match SKU-dependent parameters such as core count, frequency, L2, Memory, peak bandwidth, and theoretical compute. Do not select the 1.4/1.6 TB/s tiers of Ascend950PR or the 4 TB/s tier of Ascend950DT solely from the PR/DT product family. Mark them unconfirmed when the exact model cannot be matched. This guide does not maintain a duplicate hardware-parameter table.

## 2. Query Effective Compiler Capacity

First locate the TileLang actually imported by the task rather than inferring its version from a directory name:

```bash
python -c 'import pathlib, tilelang; print(pathlib.Path(tilelang.__file__).resolve())'
```

Search the corresponding source for resource limits and execution-domain reservations:

```bash
rg -n "GetSharedMemoryLimit|UnifiedBuffer|HasSimtVF|SIMT_VF|ThreadContext" <tilelang_root>/src/ascend
```

The currently verified `src/ascend/transform/auto_schedule.cc` inspects the final kernel body: if any `SIMT_VF` block remains, it reserves a 32 KiB thread context from the 248 KiB UB, reducing the shared-memory limit to 216 KiB. Pure `SimdVF` does not trigger this reservation. These values are facts only for the current source; recheck other versions and architectures. A branch eliminated at compile time has no effect, but initialization, tail, or fallback `SimtVF` still present in IR triggers the kernel-wide reservation. Therefore, recompute capacity after any execution-domain change. This does not imply that `SimtVF` itself should be removed.

## 3. Establish a Resource Ledger

```text
Physical capacity: Confirmed device documentation or tools in the task environment
Effective compiler limit: Result calculated by current lowering for the final kernel body
Explicit footprint: sum(aligned_extent * dtype_bytes * buffer_versions)
Resident footprint: LUT / index / state / metadata
Implicit or temporary overhead: Record only the portion proven by current source, IR, or compilation errors
Safety margin: Specific protected resource and numerical basis
Remaining capacity: Effective compiler limit - the footprints above
```

When physical capacity and the compiler limit disagree, first verify the architecture, import path, and version. A safety margin must protect a specific resource; do not replace the formula with an arbitrary cap. Use the ledger to enumerate complete contiguous units, the maximum legal capacity, and neighboring SIMD/DMA-aligned candidates. Recompute after changes to `SimtVF`, buffer versions, padding, LUT/index data, or temporary materialization, and validate using generated IR and runtime results.
