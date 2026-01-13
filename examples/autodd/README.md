# AutoDD - Automatic Delta Debugging for TileLang

AutoDD (Automatic Delta Debugging) is a built-in debugging tool for TileLang that automatically simplifies complex Python programs to the minimal code needed to reproduce a specific error. This is extremely useful for debugging large, complex TileLang programs.

## What is Delta Debugging?

Delta Debugging is an automated debugging technique with the core idea:
1. Given a program that triggers a bug
2. Systematically remove code fragments from the program
3. Check if the simplified program still triggers the same bug
4. Eventually obtain the minimal code that triggers the bug

AutoDD uses a Probability Distribution Driven Delta Debugging (PDD) algorithm for efficient search of minimized code.

## Why AutoDD?

When developing TileLang programs, bugs are often hidden in complex code:

- **Lots of irrelevant code**: Real projects may have hundreds of lines of configuration, helper functions, logging, etc.
- **Hard to locate**: Error messages may point to underlying TVM/CUDA rather than TileLang code
- **Tedious debugging**: Manually deleting code to locate bugs is very time-consuming

AutoDD automates this process, reducing hundreds of lines of code to just a few dozen, directly exposing the root cause of the problem.

## Usage

### Basic Usage

```bash
python -m tilelang.autodd <source_file> --err-msg "<error_message>" -o <output_file>
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `source` | Path to the input Python source file |
| `--err-msg` | Error message to match (searched in stdout or stderr) |
| `-o, --output` | Path to the minimized output file |
| `--backend` | Execution backend: `runner` (faster) or `subproc` (more stable), default `runner` |
| `--timeout` | Timeout for each task in seconds, default 60 |
| `-j, --jobs` | Number of parallel jobs, default 1 |

### Example

Run AutoDD on `tilelang_buggy.py` in this directory:

```bash
# Use 4 parallel jobs, search for "Dimension mismatch" error
python -m tilelang.autodd tilelang_buggy.py --err-msg "Dimension mismatch" -o minimized.py -j 4

# Or use subprocess backend (more stable but slower)
python -m tilelang.autodd tilelang_buggy.py --err-msg "Dimension mismatch" -o minimized.py --backend subproc
```

## Example Files

### `tilelang_buggy.py`

A complex TileLang program with a bug (~200 lines), containing:
- Multiple useless helper functions (`calculate_optimal_block_size`, `get_memory_requirements`, etc.)
- A complex configuration class (`MatmulConfig`)
- Unused benchmark code (`benchmark_pytorch`)
- **A GEMM shape mismatch bug**

The bug is on line 124:
```python
B_shared = T.alloc_shared((block_M, block_N), dtype)  # Wrong! Should be (block_K, block_N)
```

### `tilelang_minimized_expected.py`

The expected output after AutoDD simplification (~30 lines). The simplified code clearly shows the root cause of the bug:

```python
def buggy_matmul(...):
    @T.prim_func
    def matmul_kernel():
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug!
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.gemm(A_shared, B_shared, C_local)  # Error occurs here
```

## How AutoDD Works

AutoDD uses AST (Abstract Syntax Tree) analysis and multiple rewrite rules to simplify code:

### 1. Fast Reducers
- **Statement removal**: Directly remove statements that don't affect bug reproduction
- **If statement simplification**: Simplify `if cond: body` to `body`
- **For loop simplification**: Bind loop variables to constants

### 2. Canonicalizers
- **With statement expansion**: Convert `with expr as var` to explicit assignment
- **Function argument extension**: Add `*args, **kwargs` for compatibility

### 3. Simplifiers
- **Assignment simplification**: Replace complex expressions with constants
- **Function call simplification**: Simplify `f(x)` to `x`
- **Binary operation simplification**: Simplify `a + b` to `a` or `b`

### 4. Slow Reducers
- **Expression removal**: Remove arbitrary expressions
- **Argument removal**: Remove function arguments
- **Integer reduction**: Gradually reduce large integers

## Use Cases

1. **TileLang kernel debugging**: Simplify complex TileLang programs to locate bugs
2. **Bug report submission**: Generate minimal reproduction code for easier issue tracking
3. **Understanding errors**: Easier to understand the nature of errors after removing irrelevant code
4. **Regression testing**: Simplified code can serve as regression test cases

## Notes

1. **Error message matching**: The `--err-msg` parameter needs to exactly match a string in the error output
2. **Timeout setting**: For programs with long compilation times, you may need to increase `--timeout`
3. **Parallel jobs**: Increasing `-j` can speed up the simplification process but consumes more resources
4. **Backend selection**: If the `runner` backend is unstable, try the `subproc` backend

## References

- [Delta Debugging Paper](https://www.st.cs.uni-saarland.de/papers/tse2002/)
- [TileLang Documentation](https://github.com/tile-ai/tilelang)
