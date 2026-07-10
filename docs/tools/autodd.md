# AutoDD

AutoDD reduces a failing Python program while preserving a selected failure
signature. It repeatedly rewrites the program's Python abstract syntax tree
(AST), executes each candidate, and keeps candidates whose standard output or
standard error still contains the requested text. The result is a smaller,
standalone reproducer that is easier to inspect, report, and turn into a
regression test.

AutoDD is intended for deterministic failures that can be reproduced by
running one Python file. It minimizes Python source, not TIR after lowering.

## Quick Start

First, run the source program directly and choose a short, stable, and specific
substring from its output. Then invoke AutoDD:

```bash
python -m tilelang.autodd examples/autodd/tilelang_buggy.py \
  --err-msg "T.gemm K shape check failed" \
  -o minimized.py
```

Run the minimized file to confirm that it still exposes the intended failure:

```bash
python minimized.py
```

AutoDD considers a candidate interesting when `--err-msg` occurs in captured
standard output or standard error. Matching is case-sensitive and does not
require a nonzero exit status. Use a distinctive failure substring so an
unrelated warning or log message cannot satisfy the test accidentally.

## Command-Line Options

```text
python -m tilelang.autodd source --err-msg MSG -o OUTPUT \
  [--backend {runner,subproc}] [--timeout SEC] [-j N]
```

| Argument | Description | Default |
| --- | --- | --- |
| `source` | Readable Python file to minimize | Required |
| `--err-msg MSG` | Case-sensitive substring searched in stdout and stderr | Required |
| `-o OUTPUT`, `--output OUTPUT` | File updated with the best accepted candidate | Required |
| `--backend runner` | Execute candidates in reusable spawned worker processes | `runner` |
| `--backend subproc` | Execute every candidate with a fresh `python3` process | - |
| `--timeout SEC` | Per-candidate execution timeout in seconds | `60` |
| `-j N`, `--jobs N` | Number of candidates evaluated concurrently | `1` |

The `runner` backend avoids starting a new interpreter for every candidate and
is usually faster. Its worker processes are reused, so imported module and
native runtime state can persist between candidates. Use `subproc` when the
failure depends on process-global state, a worker exits unexpectedly, or the
faster backend produces inconsistent results.

Increasing `--jobs` can shorten a reduction, but each worker may compile and
run a kernel independently. Limit concurrency when candidates compete for the
same GPU or consume substantial host memory.

## Preserve Required Code

Some setup, synchronization, or failure-triggering code must remain intact for
the reproducer to be meaningful. Mark such code as frozen so AutoDD does not
rewrite it.

For one physical line, append an annotation:

```python
initialize_runtime()  # autodd: freeze
```

For a statement block, use paired annotations:

```python
# autodd: freeze-start
config = load_config()
initialize_runtime(config)
# autodd: end-freeze
```

AutoDD converts these annotations to an internal no-op context manager and
adds its import when necessary. The single-line form must be on a statement
that occupies one physical line. Use the block form for multiline calls.

You can also express the boundary directly:

```python
from tilelang.autodd import __freeze__

with __freeze__:
    config = load_config()
    initialize_runtime(config)

shape = __freeze__(compute_required_shape())
```

`__freeze__` is an identity operation at runtime. AutoDD protects the marked
subtree and its enclosing statements so a parent control-flow rewrite cannot
discard the frozen code. If the file uses `from __future__` imports, add the
explicit `__freeze__` import after them; this prevents annotation preprocessing
from inserting an import ahead of the required future-import position.

## Reduction Process

AutoDD applies several groups of AST transformations until no candidate makes
further progress:

1. Remove statements and simplify `if` and `for` constructs.
2. Canonicalize constructs such as `with ... as ...` and function arguments.
3. Simplify assignments, calls, binary expressions, and arguments.
4. Reduce integer constants and try slower expression-level removals.

Every proposed source file must remain syntactically valid and reproduce the
selected output substring before AutoDD accepts it. The output file is updated
as smaller candidates are found.

## Practical Guidance

- Make the source file self-contained. Candidates run from temporary files, so
  do not rely on the source file's own directory being added to `sys.path`.
- Remove nondeterminism before minimizing. Intermittent failures can cause
  AutoDD to accept or reject candidates incorrectly.
- Start with the default backend and one job. Add `-j 4` after confirming that
  independent candidates can share the available compute resources.
- Raise `--timeout` above the normal compile-and-run time. A timeout is reported
  as output, so avoid using a generic timeout string as the target signature.
- Freeze only the smallest required region. Large frozen regions limit how far
  the program can be reduced.
- Treat the generated file as a reproducer, then review and format it before
  adding it as a regression test.

A complete shape-mismatch example is available in `examples/autodd/`, including
the original program and an expected minimized result.
