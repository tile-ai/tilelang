# TileLang Python Secure Coding Guidelines

<applicability>
Language: Python
Side: Host
Domain: false
Enabled by default: true
</applicability>

<review_load>
General review subagent rule capacity limit: 5
</review_load>

## Purpose

Review TileLang Python host code for secure-coding and resource-management issues.

## Quick Index

| Rule ID | Rule name | Category | Severity |
|---------|-----------|----------|----------|
| 1.1 | Guard against zero divisors in division and modulo operations | Numeric safety | High |
| 2.1 | Do not leak sensitive data through exceptions | Exception handling | High |
| 3.1 | Validate and normalize external file paths | File operations | High |
| 4.1 | Do not use `shell=True` with subprocess | Command execution | High |
| 5.1 | Reliably release files, temporary directories, and subprocesses on every path | Resource management | High |

## Applicable Scenarios

- **Test code** (`tests/`): Data generation, API invocation, accuracy assertions, and benchmarks.
- **Tool/build scripts** (`scripts/`, pytest plugins): Compilation, paths, and result processing.
- **PyTorch interface code** (the current operator's host wrapper, public interface, and independent reference): User Tensor and parameter handling.

---

### 1. Safe Numeric Operations

##### Rule 1.1 Guard Against Zero Divisors in Division and Modulo Operations

**Applicable scenarios**: User Tensor shapes, public parameters, or environment variables participate in `/`, `//`, `%`, or ceil-div operations.

**Issue description**: Python wrappers and factories often calculate groups, tiles, strides, and output shapes before entering the kernel. A zero divisor makes the call fail before compilation. Even if its source variable is nonzero, a derived divisor may become zero after subtraction or integer division.

**Review method**: Scan arithmetic expressions and trace each divisor back to its source. Check whether the guard protects the same variable, dominates the operation, and covers empty Tensors and zero-dimensional shapes. A nonzero literal or a specialization parameter already asserted positive on the same path may be excluded.

```python
# Risky
num_groups = hidden // group_size

# Correct
assert group_size > 0 and hidden % group_size == 0
num_groups = hidden // group_size
```

Nonzero constant divisors do not require redundant validation.

**Decision method**: Assign `FAIL` when a publicly reachable input can make the divisor zero and the call chain contains no guard. A docstring requirement that the value be nonzero, or test data that is always nonzero, is not a guard.

---

### 2. Exception Handling

##### Rule 2.1 Do Not Leak Sensitive Data Through Exceptions

**Issue description**: Exceptions, logs, and assertion messages may appear in CI, shared reports, or user interfaces. Printing complete Tensors, tokens, keys, all environment variables, or unnecessary local paths for diagnostic purposes increases the exposure of sensitive information.

**Review method**: Check `raise`, `assert` messages, logging, `print`, and forwarded subprocess errors. Trace whether formatted arguments originate from user data, environment variables, or large Tensors. Shapes, dtypes, parameter names, and necessary scalar bounds are generally safe to output.

**Exclusion rules**: Outputting nonsensitive kernel source or a small statistical summary behind an explicit debug switch is acceptable when disabled by default. “Used only for testing” does not exempt a sensitive-data leak.

**Decision method**: Assign `FAIL` when sensitive values or unbounded large objects can reach default log or exception paths. Mark the issue for confirmation when data sensitivity cannot be determined.

---

### 3. Safe File Operations

##### Rule 3.1 Validate and Normalize External File Paths

**Issue description**: Paths from CLI arguments, environment variables, or configuration may contain `..`, symbolic links, empty values, or unexpected absolute paths. A tool that uses such paths to read, overwrite, or delete files may escape its intended scope.

**Review method**: Normalize paths with `Path.resolve()` before performing file operations. According to the tool's actual responsibilities, validate existence, file type, and allowed root directories. When restricting directory scope, apply a reliable check such as `relative_to` to the normalized path; do not compare string prefixes. Create temporary files and directories with `tempfile`.

**Exclusion rules**: If a tool is publicly designed to accept arbitrary user paths, it need not restrict them to the repository root. It must still normalize them and avoid passing them to a dangerous shell command or broad deletion operation.

**Decision method**: Assign `FAIL` when an untrusted path can cause unauthorized reads or writes, path traversal, or broad damage. Treat a missing existence check that has no security impact as a functional issue.

---

### 4. Safe Command Execution

##### Rule 4.1 Do Not Use `shell=True` with subprocess

**Issue description**: When the shell parses a command string, spaces, quotes, redirections, command substitutions, and control characters can all alter command boundaries. Concatenating any external input into that string may enable command injection.

**Review method**: Check `subprocess.run/Popen/check_*`, `os.system`, and indirect command helpers. Trace the sources of the command, arguments, environment, and cwd. Prefer an argument list with `shell=False`, and handle the return code, timeout, and stderr explicitly.

```python
# Incorrect
subprocess.run(f"pytest {path}", shell=True)

# Correct
subprocess.run(["pytest", str(path)], check=True)
```

**Decision method**: Assign `FAIL` when external input can reach a shell command string. A completely static `shell=True` command with no input is not evidence of injection, but should still be reported as removable risky usage.

---

### 5. Resource Management

##### Rule 5.1 Reliably Release Files, Temporary Directories, and Subprocesses on Every Path

**Issue description**: When a test fails, compilation raises an exception, an operation times out, or a user interrupts execution, leaked file handles, temporary directories, and background subprocesses can contaminate later cases and may occupy devices or ports.

**Review method**: Inspect resource lifetimes along normal-return, exception, timeout, and interruption paths. Prefer context managers for files and temporary directories. Record the exact handle/PID for a long-running subprocess, then terminate and wait for it in `finally`; kill that exact process only if necessary.

**Exclusion rules**: Assign `PASS` when resource ownership is explicitly transferred to the caller and both the interface documentation and every caller fulfill the close responsibility.

**Decision method**: Assign `FAIL` when a reachable exit path omits a close, wait, or cleanup operation. Do not hide unclear ownership by broadly cleaning up processes by name.
