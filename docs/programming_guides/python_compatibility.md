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
| `for x in list`         | ❌        | Use index-based loop                     |
| `while condition`       | ✅        |                                          |
| `if` / `elif` / `else`  | ✅        |                                          |
| `x if cond else y`      | ✅        | Ternary expression                       |
| `break` / `continue`    | ✅        |                                          |
| `enumerate()` / `zip()`    | ❌     |                                          |

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
