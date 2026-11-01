# TileLang Operator Red-Line Issue Checklist

<applicability>
Language: Python, TileLang DSL
Side: All
Domain: false
Enabled by default: true

Applicable scenarios: High-frequency red-line coding issues encountered in TileLang operator development
Introduction: A checklist of common TileLang red-line issues. Eight rules cover division by zero, indexing, integer overflow, initialization, data races, resource release, and GM address ranges
Categories (All): Division-by-zero guards, index bounds, no signed-integer overflow, no unsigned-integer wraparound, no reads of uninitialized data, and avoiding data races
Categories (Host): Resource release
Categories (Kernel): Use sufficiently wide integer types for GM memory offsets and sizes
</applicability>

<review_load>
General review subagent rule capacity limit: 3
</review_load>

## Purpose

Review TileLang host and kernel code for red-line correctness, safety, and resource-management issues.

## Quick Index

### Applicable to Both Host and Kernel `[Applies to: All]` (6 Rules)

| # | Issue type | Category | Severity |
|---|------------|----------|----------|
| 1 | Guard against division by zero in division/remainder operations | Numeric safety | High |
| 2 | Validate GM/UB/fragment indices | Memory safety | High |
| 3 | Prevent signed-integer overflow | Numeric safety | High |
| 4 | Prevent unsigned-integer wraparound | Numeric safety | High |
| 5 | Do not read or write back uninitialized data | Memory safety | High |
| 6 | Avoid data races among threads, cores, and pipeline stages | Concurrency safety | High |

### Host Only `[Applies to: Host]` (1 Rule)

| # | Issue type | Category | Severity |
|---|------------|----------|----------|
| 7 | Reliably release files, subprocesses, and temporary resources | Resource management | High |

### Kernel Only `[Applies to: Kernel]` (1 Rule)

| # | Issue type | Category | Severity |
|---|------------|----------|----------|
| 8 | Use sufficiently wide integer types for GM memory offsets and sizes | Memory safety | High |

---

## Detailed Rules

### 1 Ensure That Division and Remainder Operations Cannot Divide by Zero `[Applies to: All]`

**Review strategy**

Scan `/`, `//`, `%`, `T.ceildiv`, and normalization expressions, and trace the source of each divisor.

| Divisor source | Decision |
|----------------|----------|
| Nonzero compile-time constant | PASS |
| Specialization parameter constrained to be positive by a factory/assertion | PASS |
| User Tensor shape, attribute, or dynamic intermediate value | Must have wrapper validation or a kernel guard |
| Derived expression such as `value - 1` | Prove again that the result is nonzero |

**Incorrect example**

```python
num_groups = T.ceildiv(hidden, group_size)
```

**Correct example**

```python
assert group_size in (16, 32, 64, 128)
num_groups = T.ceildiv(hidden, group_size)
```

---

### 2 Ensure That External Data Used as an Index Is Within Bounds `[Applies to: All]`

**Review strategy**

Trace gather indices, expert IDs, stride offsets, and dynamic slices. A guard must dominate the GM load/store; reading out of bounds and then masking the output is still incorrect.

**Incorrect example**

```python
value = x[index[row]]
if index[row] < num_rows:
    out[row] = value
```

**Correct example**

```python
if 0 <= index[row] and index[row] < num_rows:
    out[row] = x[index[row]]
```

---

### 3 Prevent Signed-Integer Overflow `[Applies to: All]`

**Issue description**

If `row * stride + col`, a shape product, byte count, flattened offset, or `T.view` length overflows in an intermediate expression, storing the final result in a wider variable cannot recover the correct value. The fact that integers in a Python factory do not overflow does not prove that the lowered IR expression still uses a sufficiently wide dtype.

**Review strategy**

1. Derive the maximum absolute value of every multiplication, addition, and alignment expression from the largest publicly supported shape.
2. Check the dtype of every operand in the IR and ensure widening occurs before multiplication.
3. Inspect every `T.cast`, function parameter, `T.alloc_var`, and narrow SIMD integer for unjustified narrowing.

**Exclusion rules**: Assign `PASS` when a compile-time constant expression is evaluated by Python and proven range-safe before entering the IR, or when a narrow type is used only for a lane/index already proven to be in range.

**Decision method**: Assign `FAIL` when an intermediate value can overflow within the valid input range and affect an address, loop count, partitioning operation, or output. If the public maximum shape is unspecified, mark the issue for confirmation and state the boundary information required.

---

### 4 Prevent Unsigned-Integer Wraparound `[Applies to: All]`

**Issue description**

If the lowered value is an unsigned integer, `value - 1` wraps to a large positive number when `value == 0`. That result often corrupts tail-tile lengths, reverse loops, and address offsets.

**Review strategy**

- Inspect subtraction, expanded ceil-div expressions, reverse loops, `n - offset`, and alignment formulas.
- Trace each operand's signedness; do not assume that an operand is signed merely because the Python source does not visibly use `uint`.
- A lower-bound guard must dominate the subtraction itself; comparing after wraparound is ineffective.

**Decision method**: Assign `FAIL` when zero or a smaller left operand is reachable and the wrapped result participates in memory access, a loop, or output. Assign `PASS` when a complete upstream lower-bound proof exists.

---

### 5 Do Not Read or Write Back Uninitialized Data `[Applies to: All]`

**Issue description**

`T.alloc_shared`, `T.alloc_fragment`, `T.alloc_var`, and `torch.empty` do not imply the zero-initialization required by an algorithm. Correct behavior on a full tile does not prove that every valid output is covered in a tail tile, conditional branch, or first pipeline iteration.

**Review strategy**

1. Starting from each buffer/scalar allocation, verify that every reachable path assigns a value before the first read.
2. For full-width register accesses, verify that UB padding lanes are initialized; checking only the valid GM slice is insufficient.
3. When int64/packed data is written through a narrow view, verify that every constituent word/byte is defined.
4. For reducers, partial outputs, and atomic targets, inspect both host-side and kernel-side initialization.

After finding an uninitialized read, continue tracing whether its value affects a valid GM writeback, address, branch, synchronization operation, or exception. If it occurs only in an invalid tail row and is provably unobservable, record it for confirmation or as a robustness recommendation; do not immediately classify it as a functional `FAIL`.

**Correct example**

```python
# When int64 output is written through int32 UB, the high 32 bits must also be initialized.
with T.SimdVF():
    T.clear(acc_ub)
```

**Decision method**: Assign `FAIL` when an uninitialized value can reach a valid output, address, or control flow. Assign `PASS` when every valid lane on every path is written before it is read.

---

### 6 Avoid Data Races `[Applies to: All]`

**Review strategy**

1. Every thread under `T.Parallel`/`T.SimtVF` writes to a unique address or uses the correct atomic operation.
2. `T.Persistent` does not allow different cores to write the same output.
3. Multistage buffer versions, producer/consumer order, and lifetimes are correct.
4. An atomic operation's target address, dtype, initial value, and invocation count conform to the algorithm contract.

**Issue description**

Assigning a different task ID to every core does not automatically prove that write addresses are disjoint. Flattening/unflattening, shared experts, partial reductions, and in-place aliases may cause multiple execution units to write the same location. A non-atomic read-modify-write cannot rely on coincidentally favorable execution order.

**Exclusion rules**: Assign `PASS` when the task domain and address formulas prove that all write sets are pairwise disjoint, or when the current TileLang/PTO lowering explicitly supports the atomic dtype/operation being used and the algorithm permits arbitrary execution order.

**Decision method**: Assign `FAIL` when two concurrently executable units can write the same location without synchronization, or when the atomic target's initial value is incompatible with the accumulation semantics. If evidence for the target atomic lowering is unavailable, mark the issue for confirmation.

---

### 7 Reliably Release Files, Subprocesses, and Temporary Resources `[Applies to: Host]`

**Issue description**

When testing, building, or profiling fails, unclosed files, subprocesses, temporary directories, or device resources contaminate subsequent cases, especially under pytest-xdist and multi-device workloads.

**Review strategy**: Inspect resource lifetimes along normal-return, exception, timeout, and user-interruption paths. Prefer context managers. When a resource must cross scopes, use `try/finally` covering every exit.

**Decision method**: Assign `FAIL` when a reachable exit path omits a close, wait, or cleanup operation. Assign `PASS` when a standard context manager fully owns the resource.

---

### 8 Use Sufficiently Wide Integer Types for GM Memory Offsets and Sizes `[Applies to: Kernel]`

**Issue description**

GM offsets, total element counts, and byte counts must cover the complete range of the largest publicly supported shape. Unjustified narrowing to 32 bits after entering the kernel or lowering causes the latter part of a large Tensor to access incorrect addresses.

**Review strategy**

1. Derive the maximum flattened offset, stride product, and byte count.
2. Check whether operands are converted to sufficiently wide integer types before multiplication.
3. Distinguish a SIMD lane index used only within a small range from a final GM address. The former may be narrow; the latter must cover the full address range.

**Exclusion rules**: Assign `PASS` when the public interface and factory specialization strictly limit the total element count to the range of the target narrow type, and the constraint covers every entry point.

**Decision method**: Assign `FAIL` when the largest valid input can truncate an offset/size that is then used for GM access. If the maximum input is unclear, mark the issue for confirmation.

---

## Review Checklist

- [ ] Dynamic divisors have a nonzero guarantee
- [ ] GM/UB/fragment indices and slices are in bounds
- [ ] Signed-integer operations do not overflow
- [ ] Unsigned-integer operations do not wrap around
- [ ] Every value written back is initialized on every path
- [ ] No data races exist among threads, cores, or stages
- [ ] Host resources are released on exception paths
- [ ] Integer types for GM offsets and sizes cover the complete range
