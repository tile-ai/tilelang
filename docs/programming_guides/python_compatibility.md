# Python Compatibility

TileLang is a Python-embedded DSL, but not all Python syntax is supported inside
TileLang DSL. This guide clarifies what works, what doesn't, and how
to translate common Python patterns into TileLang equivalents. Specially, we focus on
the kernel part (scripts inside `with T.Kernel`) semantics. For host-side semantics when
using eager-style JIT, please stay tuned for our upcoming documentation.

The following codes use the conventional aliases:

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## Control Flow & Loops

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `for i in range(n)`     | ✅        | Maps to `T.serial(n)`                    |
| `for i in range(a,b,s)` | ✅        | Maps to `T.serial(a, b, s)`              |
| `for x in list/tuple`   | ✅        | Compile-time iteration; expands the loop body |
| `while condition`       | ✅        |                                          |
| `if` / `elif` / `else`  | ✅        |                                          |
| `x if cond else y`      | ✅        | Ternary expression                       |
| `break` / `continue`    | ✅        |                                          |
| `enumerate()` / `zip()` | ✅        | Compile-time iteration over Python iterables |

### Device loops and compile-time iteration

In a `for` statement, calling Python's built-in `range` creates a serial device
loop, including when its bounds are dynamic. Aliases of the built-in `range`
behave the same way; a user-defined function named `range` is not replaced.
Explicit `T.serial`, `T.Parallel`, and other TileLang loop constructors retain
their existing behavior. Iterating a Python `range` object also creates a serial
loop.

Other Python iterables, including lists, tuples, `enumerate`, `zip`, and
generators, are consumed during IR construction and expand the loop body once
per item. They must be finite. Nested loops can mix compile-time iteration with
device loops. Compile-time loops support `break` and `continue` under
compile-time conditions; runtime-dependent `break`/`continue` targeting these
loops is not yet supported. Use a serial loop for that case.
Device-side `break` and `continue` in `T.serial` / `for ... in range(...)`
remain supported, including when that device loop is nested inside a
compile-time Python loop. The restriction concerns the loop targeted by the
control statement, not whether its condition is dynamic in general.

List, tuple-via-generator, dictionary, and set comprehensions use Python
iteration rather than device loops:

```python
scale_idx_table = [
    ((j >> 4) & 7) | ((j >> 7) << 3)
    for j in range(block_K)  # block_K must be known during IR construction
]
for index, value in enumerate(scale_idx_table):
    output[index] = value
```

Comprehension iteration bounds and filters must be evaluable during IR
construction, but the resulting elements may be symbolic expressions.
Comprehension variables do not leak into the enclosing scope. Generator
expressions retain Python's lazy evaluation semantics and must be consumed
while their captured IR values remain in scope.

A Python list is not a device buffer: indexing it with a runtime TIR variable
does not create a device-side lookup table. Use a buffer or compute the index
expression directly when runtime lookup is needed.

## Data Access

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `a[i]` indexing         | ✅        | Multi-dim indexing supported: `a[i, j, k]` |
| `a[i:j]` slicing        | ✅        | Creates `BufferRegion`                   |
| `a[-1]` negative index  | ✅        |                                          |

## Assignment & Arithmetic Operations

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `x = expr`              | ✅        |                                          |
| `+`, `-`, `*`, `/`, `%` | ✅        | Maps to device-side arithmetic operations |
| `+=`, `-=`, `*=`, etc.  | ✅        | Augmented assignment                     |
| `a = b = c`             | ❌        | Use separate assignments                 |

## Functions & Classes

As a kernel script language, TileLang doesn't support functions or classes. You can use `@T.macro` to define reusable code blocks, which will be inlined at compile time like `__device__` function.

## Statements & Built-in Functions

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `with`                  | ⚠️        | Only `T.Kernel`, `T.ws`                  |
| `assert`                | ⚠️        | Use `T.device_assert` or `T.assert`      |
| `print()`               | ⚠️        | Use `T.print()`; `print` works for Python expressions |
| `len()`                 | ❌        | Use `buffer.shape[dim]`                  |
| `type()`, `isinstance()`| ❌        |                                          |
