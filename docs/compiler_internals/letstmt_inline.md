# LetStmt Inlining in TileLang

This document explains how `LetStmt` inlining works in TileLang's simplification pipeline, which is an important optimization that affects code generation and performance.

## Overview

A `LetStmt` (Let Statement) is a temporary variable binding in the IR (Intermediate Representation). During compilation, TileLang's simplifier may choose to inline these temporary variables to simplify the code. TileLang also provides a standalone `LetInline` pass that performs eager substitution before the main legalization pipeline. However, not all `LetStmt` nodes can be safely inlined.

## When Does LetStmt Get Inlined?

The inlining logic is implemented in `src/transform/simplify.cc`. A `LetStmt` will be inlined if **both** of the following conditions are met:

### 1. The value satisfies `CanInlineLetStmt`

The `CanInlineLetStmt` helper returns `true` when:

- **The value is a constant** (`is_const_number(op->value)` returns true)
- **The value is a variable** (`op->value.as<VarNode>()` returns a node)
- **The value is an integer expression without side effects**:
  - The value has `int` dtype
  - The side effect level is `kPure` or lower (no observable side effects)

```cpp
bool CanInlineLetStmt(const LetStmtNode *op) {
  if (is_const_number(op->value))
    return true;
  if (op->value.as<VarNode>())
    return true;
  // Won't face the deep expression explosion problem as in Let expression.
  // attempt to inline as much as possible if the value integer type(can be
  // index).
  if (!op->value.dtype().is_int())
    return false;
  return SideEffect(op->value) <= CallEffectKind::kPure;
}
```

### 2. The variable is NOT used in buffer definitions

Even if `CanInlineLetStmt` returns true, the variable will **not** be inlined if it's used in a buffer's definition (shape, strides, elem_offset, or data fields).

This protection exists because:
- Buffer definitions are not updated during the simplification pass
- If a variable used in a buffer definition is inlined, later references to that buffer would fail to find the variable definition
- This would cause compilation errors or incorrect behavior

The mutator checks this before dropping the binding:

```cpp
bool used_in_buffer_def = used_in_buffer_def_.count(op->var.get());

if (can_inline && !used_in_buffer_def) {
    return body;  // Inline: remove LetStmt and return body directly
}
```

## Example: Why Buffer Definition Variables Are Protected

Consider this code:

```python
let stride = M * 16
let buffer_a = Buffer(data, shape=[M, N], strides=[stride, 1])
buffer_a[i, j] = ...
```

- `stride` satisfies `CanInlineLetStmt` (it's an int expression with no side effects)
- However, `stride` is used in `buffer_a`'s `strides` field
- If we inline it, the buffer definition becomes `strides=[M*16, 1]`
- But the Buffer object's fields are not updated during simplification
- Later code accessing `buffer_a` would fail to find the `stride` variable

Therefore, `stride` is added to `used_in_buffer_def_` and will **not** be inlined.

## How Variables Are Collected

The `CollectVarsUsedInBufferDefinition` helper traverses all `BufferLoad` and `BufferStore` nodes and collects variables used in their buffer definitions:

```cpp
void VisitBuffer(const Buffer &buf) {
  // Collect variables that should remain defined
  VarUseDefAnalyzer usage(Array<Var>{});
  usage(buf->data);
  for (const auto &dim : buf->shape) {
    usage(dim);
  }
  for (const auto &dim : buf->strides) {
    usage(dim);
  }
  usage(buf->elem_offset);

  // Track for use in LetStmtNode mutator
  for (const auto &var : usage.undefined_) {
    used_in_buffer_def_.insert(var.get());
  }
}
```

## Practical Example: Temporary Variable Issue

Consider this TileLang code:

```python
for i in T.Parallel(block_N):
    idx = bx * block_N + i
    tmp = T.max(A[idx], 1)
    B[idx] = tmp / 2
    A[idx] = tmp * 2
```

In this case:
- `tmp` is an integer-like temporary variable
- It satisfies `CanInlineLetStmt` (pure int expression)
- It's **not** used in any buffer definition
- Therefore, `tmp` **will be inlined**

This means the IR becomes:

```python
for i in T.Parallel(block_N):
    idx = bx * block_N + i
    B[idx] = T.max(A[idx], 1) / 2
    A[idx] = T.max(A[idx], 1) * 2
```

If this causes issues (e.g., `A[idx]` being read twice with different values due to the first write), it indicates a potential problem with the inlining heuristic or the code pattern.

## Controlling Let Inlining via Pass Config

TileLang exposes an explicit pass configuration key, `tilelang.PassConfigKey.TL_FORCE_LET_INLINE` (`"tl.force_let_inline"`), that allows users to force the eager `LetInline` pass to run before the legalization pipeline begins. When enabled, the pipeline invokes `tilelang.transform.LetInline()` at the start of `LowerAndLegalize` (see `tilelang/engine/phase.py`). This knob is useful when debugging LetStmt-related issues or when deterministic inlining behavior is desired across different environments.

```python
from tilelang import transform
from tilelang.engine.phase import LowerAndLegalize

with transform.PassContext(
    config={transform.PassConfigKey.TL_FORCE_LET_INLINE: True}
):
    lowered_mod = LowerAndLegalize(input_mod, target)
```

If the flag is left unset (the default), the eager pass is only applied when downstream transforms opt in (for example, by calling `_Simplify(..., inline_let=True)` inside Tile operators). The guard in `tilelang/engine/phase.py` ensures the eager pass is only triggered when the user explicitly requests it.

## Summary

The LetStmt inlining mechanism is a **conservative optimization** that:
1. Aggressively inlines simple, pure integer expressions to simplify the IR
2. Protects variables used in buffer definitions to avoid breaking buffer access
3. Helps reduce IR complexity and improve code generation
4. Can be forced through `TL_FORCE_LET_INLINE` when deterministic eager inlining is required

Understanding when inlining happens is crucial for:
- Debugging compilation issues
- Understanding generated code
- Writing efficient TileLang programs
- Identifying potential optimization opportunities or bugs

## Related Files

- `src/transform/simplify.cc`: Main Simplify implementation
- `src/transform/frontend_legalize.cc`: Standalone LetInline pass
- `tilelang/engine/phase.py`: Pipeline integration for eager LetInlining
- `testing/python/transform/test_tilelang_transform_let_inline.py`: Regression coverage for the pass
