# Control Flow

This guide covers the control‑flow primitives in TileLang and how they lower to
efficient GPU code. You will use these to structure loops, handle boundaries,
and express pipelined compute.

## Overview
- Conditionals: `if` / `elif` / `else`, ternary (`x if c else y`)
- Loops: `T.serial`, `T.unroll`, `T.Parallel`, `T.Pipelined`
- While loops: `while` with a TIR condition
- Flow control: Python `break` / `continue`
- Safety: automatic OOB guards via the LegalizeSafeMemoryAccess pass

The examples assume `import tilelang.language as T`.

## Conditionals

Standard Python `if`/`elif`/`else` is supported inside `@T.prim_func` kernels.
Conditions should be TIR expressions (e.g., `i < N`). Python plain booleans are
treated as compile‑time constants and will be folded.

```python
for i in T.serial(N):
    if i < N:            # TIR condition
        C[i] = A[i] + B[i]
    else:
        pass

# Ternary
x = (A[i] if i < N else 0)
```

Short‑circuit boolean ops are supported. For multi‑dimensional bounds, use
`T.any_of` / `T.all_of` for clarity:

```python
if T.all_of(i < M, j < N):
    C[i, j] = A[i, j] + B[i, j]
```

Boundary handling note
- The LegalizeSafeMemoryAccess pass automatically inserts guards when an access
  may be out‑of‑bounds, and elides them when proven safe. You can often omit
  explicit `if` checks for simple edge handling, but keep them when you need
  custom logic or clarity.

## Loops

### Serial

`T.serial` creates a plain for‑loop. Common forms:

```python
for i in T.serial(N):
    ...                     # 0..N-1

for i in T.serial(0, N, 2):
    ...                     # 0, 2, 4, ...
```

### Unroll

`T.unroll` requests loop unrolling for small trip counts.

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

Advanced: TileLang forwards unroll hints to TIR; factor/explicit knobs are
available for expert tuning.

### Parallel (elementwise)

`T.Parallel(ext0, ext1, ...)` builds nested loops that map well to elementwise
operations. The body receives all indices in one `for` header:

```python
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

Optional hints:
- `coalesced_width=` controls memory coalescing width (used for vectorization checks).
- `loop_layout=` accepts a `T.Fragment` to annotate the layout of the entire
  nested parallel loop. The annotation is attached to the outermost loop only
  and must have `InputDim == number of nested parallel extents`.

### Pipelined (software pipelining)

`T.Pipelined(iters, num_stages=...)` overlaps producer/consumer stages (e.g.,
Global→Shared copies with compute). This is the backbone of GEMM/attention
pipelines.

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_s)  # stage: copy A tile
    T.copy(B[ko * BK, bx * BN], B_s)  # stage: copy B tile
    T.gemm(A_s, B_s, C_f)             # stage: compute
```

### Persistent (advanced)

`T.Persistent(domain, wave_size, index, group_size=...)` exposes persistent
thread‑block style looping. It is an advanced construct that TileLang lowers in
later passes and is typically used by specialized templates.

## While Loops

`while` is supported when the condition is a TIR expression. Avoid infinite
loops; TileLang will error if it detects a constant‑true condition.

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

## Break and Continue

Use Python `break`/`continue` to exit or skip within `T.serial`/`T.unroll`/
`T.Parallel`/`while` loops. Keep the body clean after a `break`/`continue` for
readability; the compiler will ignore the dead path.

## Putting It Together: Residual Tile Handling

Below is a typical edge pattern for a 2D kernel. With LegalizeSafeMemoryAccess,
the explicit guard can be omitted when you don’t need a custom edge path.

```python
for i, j in T.Parallel(M, N):
    gi = by * BM + i
    gj = bx * BN + j
    if T.all_of(gi < M, gj < N):     # optional in many cases
        C[gi, gj] = A[gi, gj] + B[gi, gj]
```

## Debugging Conditions

Use `T.print` to inspect values under predicates. For buffers, TileLang prints
from a single thread to avoid duplicate outputs.

```python
if i == 0:
    T.print(C, msg='C tile:')
```
