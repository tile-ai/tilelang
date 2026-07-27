# Pass Diff

:::{admonition} Superseded — use IR Lower Trace
:class: warning

This tool has been superseded by **IR Lower Trace** (`TL_LOWER_TRACE`), documented
in {doc}`lower_trace`. New users should use IR Lower Trace directly — it provides
phase context, codegen capture, multi-run accumulation, and an enhanced HTML
report. `TILELANG_PASS_DIFF` is retained only for backward compatibility.
:::

Pass Diff shows how TileLang's TIR changes as compiler passes run. It captures
the IR before and after each pass, computes a unified diff, and writes the
result to the terminal, an interactive HTML report, or both.

Use the environment-variable workflow to observe the complete lowering
pipeline without modifying a program. Use the Python API when you want to
apply and inspect a specific pass sequence.

## Observe the Lowering Pipeline

Set `TILELANG_PASS_DIFF` before starting the Python process:

```bash
# Print colored diffs while passes run.
TILELANG_PASS_DIFF=terminal python my_script.py

# Write an HTML report.
TILELANG_PASS_DIFF=html python my_script.py

# Print terminal diffs and write an HTML report.
TILELANG_PASS_DIFF=both python my_script.py
```

TileLang reads this setting when `tilelang` is imported and installs a process-
wide hook around TVM pass execution. Setting the variable after importing
TileLang does not enable the hook for that process.

| Variable | Accepted values | Default |
| --- | --- | --- |
| `TILELANG_PASS_DIFF` | `terminal`, `html`, `both`; `1`, `true`, `yes`, and `on` also select terminal output | `0` (disabled) |
| `TILELANG_PASS_DIFF_OUTPUT` | Directory for pipeline HTML reports | `tmp/pass_diff_output` |

False-like values (`0`, `false`, `no`, `off`, or an empty string) disable the
hook. An unrecognized nonempty value falls back to terminal output.

HTML reports use a timestamped filename such as
`pass_diff_20260710_153000.html`. The report is refreshed after each pass so a
partial result is available even during a long compilation, then finalized at
process exit.

```{figure} ../_static/img/pass_diff_html.png
:width: 600
:alt: Pass Diff HTML report showing TIR before and after a compiler pass
:align: center

Pass Diff HTML report.
```

The HTML viewer provides side-by-side before and after source, collapsible pass
sections, expandable unchanged context, copy controls, light and dark themes,
and aggregate insertion and deletion counts.

:::{note}
The pipeline hook wraps `tvm.ir.transform.Pass.__call__` for the entire process,
so it observes TVM passes outside the immediate `tilelang.compile()` call as
well. Capturing IR and, in HTML mode, rewriting the report after every pass adds
debugging overhead. Leave the feature disabled for normal builds and benchmarks.
:::

## Compare Selected Passes

The `pass_diff` Python API accepts a `PrimFunc` or `IRModule` and applies the
provided passes in order:

```python
import tilelang
from tilelang import tvm
from tilelang.utils.pass_diff import pass_diff

steps = pass_diff(
    func,
    [
        ("AnnotateDeviceRegions", tvm.tirx.transform.AnnotateDeviceRegions()),
        ("SplitHostDevice", tvm.tirx.transform.SplitHostDevice()),
        ("ThreadSync", tilelang.transform.ThreadSync("shared")),
    ],
    mode="both",
    context=5,
    html_path="tmp/selected_passes.html",
)
```

A single unnamed pass is also accepted:

```python
steps = pass_diff(func, tilelang.transform.ThreadSync("shared"))
```

| Parameter | Description | Default |
| --- | --- | --- |
| `func_or_mod` | Starting `PrimFunc` or `IRModule`; a `PrimFunc` is wrapped as `main` | Required |
| `passes` | One pass, a list of passes, or a list of `(name, pass)` pairs | Required |
| `mode` | `terminal`, `html`, or `both` | `terminal` |
| `context` | Unchanged context lines included around each diff hunk | `3` |
| `html_path` | Output file used by `html` and `both` modes | `pass_diff_report.html` |

The return value contains one dictionary per pass with these fields:

| Field | Meaning |
| --- | --- |
| `name` | Explicit step name, or a name derived from the pass type |
| `before_script` | Complete TIR script before the pass |
| `after_script` | Complete TIR script after the pass |
| `diff_lines` | Unified diff as a list of strings |
| `insertions` | Number of added lines |
| `deletions` | Number of removed lines |
| `changed` | Whether the pass changed the rendered TIR |

For example, a test can assert that a transformation introduced an expected
intrinsic:

```python
steps = pass_diff(func, tilelang.transform.ThreadSync("shared"))
assert steps[0]["changed"]
assert "tvm_storage_sync" in steps[0]["after_script"]
```

Do not enable the process-wide `TILELANG_PASS_DIFF` hook when using the
programmatic API unless you intentionally want both layers of reporting. The
API invokes each pass normally, so the global hook would capture those same
invocations again.

## Choosing an Output Mode

- Use `terminal` for a small number of passes and immediate feedback.
- Use `html` when a lowering pipeline produces many changes or you need to
  compare complete before and after scripts.
- Use `both` when iterating interactively but retaining a shareable report.
- Increase `context` only for the programmatic API; the process-wide hook uses
  three context lines.

Pass Diff compares the textual form returned by `IRModule.script()`. A textual
change is useful evidence about a pass, but it does not by itself establish a
semantic or performance change.
