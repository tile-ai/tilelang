# TileLang Semantics

TileLang is a Python-embedded DSL, but not all Python syntax is supported inside
`@T.prim_func` kernels. This guide clarifies what works, what doesn't, and how
to translate common Python patterns into TileLang equivalents.

The examples assume `import tilelang.language as T`.

## Quick Reference

### Control Flow

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `for i in range(n)`     | ✅        | Also `T.serial(n)`                       |
| `for i in range(a,b,s)` | ✅        | Also `T.serial(a, b, s)`                 |
| `for x in list`         | ❌        | Use index-based loop                     |
| `while condition`       | ✅        |                                          |
| `if` / `elif` / `else`  | ✅        |                                          |
| `x if cond else y`      | ✅        | Ternary expression                       |
| `break` / `continue`    | ✅        |                                          |

### Data Access

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `a[i]` indexing         | ✅        | Multi-dim: `a[i, j, k]`                  |
| `a[i:j]` slicing        | ✅        | Creates `BufferRegion`                   |
| `a[-1]` negative index  | ✅        |                                          |

### Assignment

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `x = expr`              | ✅        |                                          |
| `+=`, `-=`, `*=`, etc.  | ✅        | Augmented assignment                     |
| `a = b = c`             | ❌        | Use separate assignments                 |

### Functions & Structures

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `*args`, `**kwargs`     | ❌        | Use explicit parameters                  |
| List comprehension      | ❌        | Use explicit loop                        |
| Dict comprehension      | ❌        |                                          |
| Generator expression    | ❌        |                                          |
| `lambda`                | ❌        | Define a named function                  |
| `class`                 | ❌        |                                          |
| `try` / `except`        | ❌        |                                          |
| Recursion               | ❌        |                                          |
| Closures                | ❌        |                                          |

### Statements

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `with`                  | ⚠️        | Only `T.Kernel`, `T.ws`                  |
| `import`                | ❌        | Not inside kernel                        |
| `assert`                | ⚠️        | Use `T.device_assert` or `T.assert`      |

### Built-in Functions

| Python Feature          | Supported | Notes / Alternative                      |
|-------------------------|:---------:|------------------------------------------|
| `print()`               | ⚠️        | Use `T.print()`; `print` works for Python expressions |
| `len()`                 | ❌        | Use `buffer.shape[dim]`                  |
| `type()`, `isinstance()`| ❌        |                                          |

## Loops

### What Works

TileLang provides specialized loop constructs that map to efficient GPU patterns.
Python's `range()` is also supported and maps to `T.serial` internally.

```python
# Python range (supported, maps to T.serial)
for i in range(N):
    C[i] = A[i] + B[i]

# Serial loop (equivalent to range)
for i in T.serial(N):
    C[i] = A[i] + B[i]

# With start, stop, step
for i in T.serial(0, N, 2):   # or range(0, N, 2)
    C[i] = A[i] * 2

# Unrolled loop (for small, known trip counts)
for k in T.unroll(4):
    acc += A[k] * B[k]

# Grid loop (serial nested loops)
for i, j in T.grid(M, N):
    C[i, j] = A[i, j] + B[i, j]

# Parallel nested loops (parallel version)
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

# Pipelined loop (overlapped producer/consumer)
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.copy(B[ko * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_frag)
```

### What Doesn't Work

```python
# ❌ Iterating over collections
for x in my_list:         # Error
    ...

# ❌ enumerate/zip
for i, x in enumerate(A): # Error
    ...
```

### Migration Examples

| Python | TileLang |
|--------|----------|
| `for i in range(N)` | Works as-is (or `T.serial(N)`) |
| `for i in range(0, N, 2)` | Works as-is (or `T.serial(0, N, 2)`) |
| `for i, j in product(range(M), range(N))` | `T.grid(M, N)` (serial) or `T.Parallel(M, N)` (parallel) |

## Conditionals

### What Works

Standard `if/elif/else` and ternary expressions are supported:

```python
# Simple conditional
if i < N:
    C[i] = A[i]
else:
    C[i] = 0

# Ternary expression
value = A[i] if i < N else 0

# Multiple conditions with logical operators
if i < M and j < N:
    C[i, j] = A[i, j]
```

### What Doesn't Work

```python
# ❌ Python objects as conditions
if my_list:               # Error - Python object, not a valid condition
    ...

# ❌ Membership testing
if x in my_set:           # Error
    ...
```

## Buffer Operations

### What Works

```python
# Multi-dimensional indexing
value = A[i, j, k]
A[i, j] = value

# Slicing (creates BufferRegion for T.copy)
T.copy(A[i:i+BM, j:j+BN], A_shared)

# Negative indexing
last = A[-1]
A[-1] = 0

# Shape access
M = A.shape[0]
N = A.shape[1]

# Type casting
x = value.astype("float32")
```

### What Doesn't Work

```python
# ❌ Python len()
n = len(A)                # Error - use A.shape[0]

# ❌ Step in slice
A[::2]                    # Error - step not supported

# ❌ Ellipsis
A[..., 0]                 # Error
```

## Variables and Assignment

### What Works

```python
# Simple assignment
x = A[i] + B[i]

# Augmented assignment
acc += A[i] * B[i]
count -= 1

# Multiple statements
x = 1
y = 2

# Allocating buffers
shared_buf = T.alloc_shared((M, N), "float16")
local_var = T.alloc_var("int32", init=0)
```

### What Doesn't Work

```python
# ❌ Chained assignment
a = b = c = 0             # Error - use separate lines

# ❌ Tuple unpacking (except in Parallel loops)
a, b = get_pair()         # Error

# ❌ Walrus operator
if (n := compute()) > 0:  # Error
    ...
```

## Functions

### What Works

You can define helper functions outside the kernel and call them:

```python
def dequantize(packed, scale, zero):
    return (packed - zero) * scale

@T.prim_func
def kernel(A: T.Tensor((N,), "int8"), ...):
    with T.Kernel(...) as bx:
        for i in T.serial(N):
            value = dequantize(A[i], scale, zero)  # OK
```

### What Doesn't Work

```python

# ❌ Nested function definitions inside kernel
@T.prim_func
def kernel(...):
    def helper():         # Error
        ...

# ❌ *args, **kwargs
def func(*args):          # Error in kernel context
    ...
```

## Operators

### What Works

```python
# Arithmetic
x = a + b - c * d / e
x = a % b                 # modulo
x = a ** 2                # power

# Comparison
if a < b and c >= d:
    ...

# Bitwise
x = a & b                 # and
x = a | b                 # or
x = a ^ b                 # xor
x = a << 2                # left shift
x = a >> 2                # right shift
x = ~a                    # not

# Logical
if a and b:
    ...
if not flag:
    ...
```

### Type-Specific Notes

```python
# Integer division - use T.ceildiv for ceiling
tiles = T.ceildiv(N, BLOCK)

# Explicit rounding control for floats
result = T.ieee_add(a, b, "rn")  # round to nearest
```

## Common Patterns

### Iterating with Index

```python
# enumerate() is not supported, use index-based loop:
for i in range(N):          # or T.serial(N)
    result[i] = process(data[i])
```

### Conditional Accumulation

```python
# Generator expressions not supported, use explicit loop:
total = T.alloc_var("float32", init=0)
for i in range(N):          # or T.serial(N)
    if data[i] > 0:
        total[0] += data[i]
```

### Nested Loops

```python
# Sequential (range or T.serial both work):
for i in range(M):
    for j in range(N):
        C[i, j] = A[i, j] + B[i, j]

# grid (serial version, might be slow)
for i, j in T.grid(M, N):
    C[i, j] = A[i, j] + B[i, j]

# Parallel (better for performance):
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

### Early Exit

```python
# break/continue work as expected:
for i in range(N):          # or T.serial(N)
    if found(i):
        break
```

### Max/Min Finding

```python
# Python max()/min() not supported, use explicit loop:
max_val = T.alloc_var("float32", init=T.min_value("float32"))
for i in range(N):          # or T.serial(N)
    max_val[0] = T.max(max_val[0], data[i])

# Or use built-in reduction:
T.reduce_max(data, out, dim=0)
```

## Error Messages

When you use unsupported syntax, TileLang will report an error. Common messages:

| Error Message | Cause | Fix |
|--------------|-------|-----|
| `Expect the for loop to be one of: range, T.serial, ...` | Using `for x in collection` (not range) | Use `range()`, `T.serial()`, or `T.Parallel()` |
| `Consequential assignments like 'a = b = c' are not supported` | Chained assignment | Use separate assignment statements |
| `Annotation should be Var` | Wrong type annotation | Use `T.Tensor` or `T.Buffer` |

## Integer Division and Modulo

TileLang's `//` and `%` operators follow Python semantics (`floordiv`/`floormod`),
not C/C++ semantics. If you need C-style truncation behavior, use `T.truncdiv()` and
`T.truncmod()` explicitly.

**Note**: Unlike Triton, which always uses `truncdiv`/`truncmod` (C-style, inconsistent with
Python), TileLang preserves Python's expected behavior for `//` and `%`.

TileLang provides multiple division and modulo operations with different rounding
behaviors. Understanding these is important when working with negative numbers.

### truncdiv / truncmod (C-style)

Rounds toward zero. The remainder has the same sign as the dividend.

```python
T.truncdiv(-7, 2)   # = -3  (toward zero: -3.5 → -3)
T.truncmod(-7, 2)   # = -1  (since -7 = (-3) * 2 + (-1))

T.truncdiv(7, -2)   # = -3
T.truncmod(7, -2)   # = 1   (since 7 = (-3) * (-2) + 1)
```

### floordiv / floormod (Python-style)

Rounds toward negative infinity. The remainder has the same sign as the divisor.

```python
T.floordiv(-7, 2)   # = -4  (toward -∞: -3.5 → -4)
T.floormod(-7, 2)   # = 1   (since -7 = (-4) * 2 + 1)

T.floordiv(7, -2)   # = -4
T.floormod(7, -2)   # = -1  (since 7 = (-4) * (-2) + (-1))
```

### Comparison Table

| a | b | truncdiv | truncmod | floordiv | floormod |
|---|---|----------|----------|----------|----------|
| 7 | 2 | 3 | 1 | 3 | 1 |
| -7 | 2 | -3 | -1 | -4 | 1 |
| 7 | -2 | -3 | 1 | -4 | -1 |
| -7 | -2 | 3 | -1 | 3 | -1 |

### Default Behavior

- Python's `//` operator maps to `floordiv`
- Python's `%` operator maps to `floormod`
- `T.ceildiv(a, b)` computes ceiling division: `⌈a / b⌉`

### When to Use Which

- **floordiv/floormod**: Default choice, matches Python semantics
- **truncdiv/truncmod**: When you need C/C++ compatible behavior
- **ceildiv**: For computing grid sizes: `T.ceildiv(N, BLOCK)` gives the number
  of blocks needed to cover N elements
