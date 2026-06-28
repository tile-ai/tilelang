# Debugging Tile Language Programs

<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/LeiWang1999">Lei Wang</a>
</div>

## Overview

A Tile Language program (hereafter referred to as a *program*) is transformed into a hardware-executable file through several stages:

1. The user writes a Tile Language program.
2. The program undergoes multiple *Passes* for transformation and optimization (the *lower* stage, see `tilelang/engine/lower.py`), finally producing an intermediate representation (e.g., LLVM or C for CPU, CUDA for NVIDIA GPUs, etc.).
3. The generated code is compiled by the respective compiler (e.g., nvcc) into a hardware-executable file.

```{figure} ../_static/img/overview.png
:width: 300
:alt: Overview of the compilation process
:align: center

```

During this process, users may encounter roughly three categories of issues:

- **Generation issues**: The Tile Language program fails to generate a valid hardware-executable file (i.e., errors during the lowering process).
- **Correctness issues**: The resulting executable runs, but produces incorrect results.
- **Performance issues**: The executable runs with performance significantly below the expected theoretical hardware limits.

This tutorial focuses on the first two issues—how to debug generation and correctness problems. Performance tuning often requires using vendor-provided profiling tools (e.g., **Nsight Compute**, **rocProf**, etc.) for further hardware-level analysis, which we will address in future materials.

Below, we take matrix multiplication (GEMM) as an example to demonstrate how to write and debug a Tile Language program.

## Matrix Multiplication Example

In **Tile Language**, you can use the **Tile Library** to implement matrix multiplication. Here's a complete example:

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    # ...existing code...

# 1. Define the kernel (matmul) with the desired dimensions
func = matmul(1024, 1024, 1024, 128, 128, 32)

# 2. Compile the kernel into a torch function
# ...existing code...
```

## Debugging Generation Issues

TileLang essentially performs *progressive lowering*. For example, a `T.copy` may first be expanded into `T.Parallel` (see the pass `LowerTileOP`), which is then expanded again, eventually resulting in lower-level statements that can be translated to CUDA C code.

```{figure} ../_static/img/ir_transform_diagram.png
:width: 400
:alt: IR transformation diagram
:align: center

```

When the code fails to generate (for instance, a compilation error occurs), you do **not** necessarily need to jump directly into C++ passes to debug. Instead, you can first inspect the intermediate representations (IR) in Python by printing them.

For example, consider a case where a simple `T.copy` in 1D causes the lowering process to fail. The snippet below illustrates a simplified version of the problem (based on community Issue #35):

```python
@T.prim_func
def main(Q: T.Tensor(shape_q, dtype)):
    # ...existing code...
```

The TileLang lower process might yield an error such as:

```text
File "/root/TileLang/src/cuda/codegen/codegen_cuda.cc", line 1257
ValueError: Check failed: lanes <= 4 (8 vs. 4) : Ramp of more than 4 lanes is not allowed.
```

This indicates that somewhere during code generation, an unsupported vectorization pattern was introduced (a ramp of 8 lanes). Before diving into the underlying C++ code, it is helpful to print the IR right before code generation. For instance:

```python
device_mod = tir.transform.Filter(is_device_call)(mod)
# ...existing code...
```

## Debugging Correctness Issues

Sometimes, the kernel compiles and runs but produces incorrect results. In such cases, there are two main strategies to help debug:

1. **Use post-processing callbacks to inspect or modify the generated CUDA code.**
2. **Use the built-in `T.print` debugging primitive to inspect values at runtime.**

### Post-Processing Callbacks for Generated Source

After code generation (in the codegen pass), TileLang calls a callback function (if registered) to allow post-processing of the generated source code. In `src/cuda/codegen/rt_mod_cuda.cc`:

```cpp
std::string code = cg.Finish();
if (const auto *f = Registry::Get("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).operator std::string();
}
```

Hence, by registering a Python function named `tilelang_callback_cuda_postproc`, you can intercept the final CUDA code string. For example:

```python
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, _):
    print(code) # print the final CUDA code
    code = "// modified by tilelang_callback_cuda_postproc\n" + code
    return code

kernel = tilelang.compile(matmul, target="cuda")
kernel_source = kernel.get_kernel_source()
print(kernel_source)
'''
// modified by tilelang_callback_cuda_postproc
#include "cuda_runtime.h"
...
'''
```

### Runtime Debug Prints with `T.print`

TileLang provides a built-in debugging primitive called `T.print` for printing within kernels. Be mindful of concurrency and thread synchronization when using it in GPU code. Below are some examples showing how to print buffers, variables, and other data inside TileLang programs.

1. **Printing an Entire Buffer**

```python
def debug_print_buffer(M=16, N=16):
    # ...existing code...
```

2. **Conditional Printing**

```python
def debug_print_buffer_conditional(M=16, N=16):
    # ...existing code...
```

3. **Printing Thread Indices or Scalar Values**

```python
def debug_print_value_conditional(M=16, N=16):
    # ...existing code...
```

4. **Printing Fragment (Register File) Contents**

```python
def debug_print_register_files(M=16, N=16):
    # ...existing code...
```

5. **Adding a Message Prefix**

```python
def debug_print_msg(M=16, N=16):
    # ...existing code...
```

The output messages will include something like:

```text
msg='hello world' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): 0
```

### Visual Layout Inference For TileLang
 The **Visual Layout Inference** tool automatically generates visual diagrams that illustrate the mapping between logical indices, thread IDs, and register file locations.

When TileLang performs layout inference, it determines how fragment buffers are distributed across threads. The visual layout tool captures this information and generates:
1. **Textual output**: A human-readable description of the layout mapping
2. **Visual diagrams**: Color-coded plots showing the thread-to-data mapping

The visual layout inference tool is controlled through the `TL_LAYOUT_VISUALIZATION_ENABLE` and `TL_LAYOUT_VISUALIZATION_FORMATS` pass configuration. By default, `TL_LAYOUT_VISUALIZATION_ENABLE` is **disabled** to avoid performance overhead during compilation.

When enabled, `TL_LAYOUT_VISUALIZATION_FORMATS` accepts string values to control output formats:
- "txt": Text output only (same as default)
- "all": Generates all formats (TXT, PDF, PNG, SVG)
- "png": Generate PNG format only
- "pdf": Generate PDF format only
- "svg": Generate SVG format only
- "txt,svg": Generate multiple formats (comma-separated) in addition to text output

The output messages of "txt" will include something like:
```
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

## IR Lower Trace: Full-Pipeline IR Tracking

TileLang programs are lowered through a sequence of compiler *passes*, each of which may transform the IR. Understanding exactly what each pass changes is essential for debugging incorrect transformations, unexpected optimizations, or missing passes.

**IR Lower Trace** is the recommended tool for this task. It transparently captures the IR before and after *every* pass in the compilation pipeline — including the final codegen step that produces C/CUDA/HIP source — and renders a human-readable diff report in the terminal and/or a self-contained HTML page. No code changes are required: enabling a single environment variable is enough.

Compared to the older **Pass Diff** tool (`TILELANG_PASS_DIFF`), IR Lower Trace adds:

- **Phase context** — each pass is tagged with its pipeline phase (e.g. `pipeline_c`, `phase1_...`), so you can tell which backend stage a pass belongs to.
- **Codegen capture** — the final TIR-to-source lowering is recorded, and the generated C/CUDA/HIP code is dropped to disk for inspection or editing.
- **Edit-and-recompile workflow** — edit the generated codegen source on disk and rerun; your edits are injected back into compilation (with conflict detection).
- **Multi-run accumulation** — repeated compilations in the same process are tagged with `run2_`, `run3_`, … prefixes, so you can diff across runs.
- **Raw `.tir` dumps** — before/after IR for every pass is written to disk, keyed by phase and pass index.
- **Crash-safe incremental HTML** — the report is flushed after every pass, so partial results survive even if the process crashes.
- **Enhanced HTML report** — sidebar pass navigation, status dots, phase tabs, `j`/`k` keyboard navigation, `Shift+E` global expand, `F7` manual alignment, dark/light theme.

### Quick Start (Environment Variable)

The simplest way to enable IR Lower Trace is to set the `TL_LOWER_TRACE` environment variable before running your script:

```bash
# HTML report (default when set to 1/on/true/yes)
TL_LOWER_TRACE=1 python3 my_script.py

# Colored diff printed to the terminal only
TL_LOWER_TRACE=terminal python3 my_script.py

# Both terminal output and HTML report
TL_LOWER_TRACE=both python3 my_script.py

# Disabled (default — zero overhead, no patching)
python3 my_script.py
```

When HTML output is enabled, a stable symlink `<script_dir>/report.html` is maintained and points to the latest run's report. Open it directly in a browser:

```bash
# typical location
open tmp/lower_trace_dir/my_script/report.html
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TL_LOWER_TRACE` | Enable tracing. Values: `0`/`off`/`false`/`no` (off), `1`/`on`/`true`/`yes` (→ html), `terminal`, `html`, `both` | off |
| `TL_LOWER_TRACE_DIR` | Base output directory for all trace artifacts | `./tmp/lower_trace_dir` |

### Output Directory Structure

A single run produces the following layout under `TL_LOWER_TRACE_DIR`:

```text
<TL_LOWER_TRACE_DIR>/
└── <script_name>/                      # derived from sys.argv[0], e.g. "my_script"
    ├── report.html                # symlink → latest run's report
    ├── codegen.cpp                     # generated codegen source (editable, see below)
    ├── codegen.cpp.original            # baseline snapshot for edit/recompile workflow
    ├── codegen.cpp.latest              # actual codegen output of the most recent run
    └── .run_records/
        └── run_<YYYYMMDD_HHMMSS_ffffff>_<pid>/
            ├── report.html        # this run's full report
            ├── pipeline_c/             # one subdir per phase (example)
            │   ├── 00_BindTarget_before.tir
            │   ├── 00_BindTarget_after.tir
            │   ├── 01_Simplify_before.tir
            │   └── 01_Simplify_after.tir
            ├── phase2_optimize/        # another phase (illustrative)
            │   └── ...
            ├── codegen/
            │   ├── 42_codegen_before.tir
            │   └── 42_codegen_after.cpp
            └── unscoped/               # passes outside any pipeline window
                └── ...
```

Each phase gets its own subdirectory; passes are numbered globally (`00_`, `01_`, …) so ordering is unambiguous. Codegen records write the *after* artifact as `.cpp` (the generated source), while ordinary passes use `.tir` for both sides.

### Programmatic API

For fine-grained control, IR Lower Trace exposes two layers of API.

#### One-shot: `lower_trace()`

Diff a fixed chain of passes against an IR module without installing any global hook:

```python
from tilelang.tools import lower_trace as lt
from tilelang import tvm
import tilelang.transform as transform

# Diff a single pass
results = lt.lower_trace(func, transform.Simplify(), mode="terminal")

# Diff a named chain, write an HTML report
results = lt.lower_trace(
    func,
    [
        ("Annotate",   tvm.tirx.transform.AnnotateDeviceRegions()),
        ("Split",      tvm.tirx.transform.SplitHostDevice()),
        ("ThreadSync", transform.ThreadSync("shared")),
    ],
    mode="both",
    html_path="my_diff.html",
)
```

| Parameter | Description |
|-----------|-------------|
| `func_or_mod` | A `PrimFunc` or `IRModule` to run passes on |
| `passes` | A single pass, a list of passes, or a list of `(name, pass)` tuples |
| `mode` | `"terminal"`, `"html"`, or `"both"` (default: `"terminal"`) |
| `context` | Number of context lines in the unified diff (default: 3) |
| `html_path` | Output path for the HTML report (default: `"lower_trace_report.html"`) |

Returns a `list[dict]` with one entry per pass step, each containing `name`, `before_script`, `after_script`, `diff_lines`, `insertions`, `deletions`, and `changed`.

#### Global hook: `enable()` / `disable()` / `reset()`

To trace the *entire* compilation pipeline of a real kernel (what the environment variable does, but programmatically):

```python
from tilelang.tools import lower_trace as lt

# Enable tracing for the rest of the process.
lt.enable(mode="both")

# ... run tilelang.compile() / kernel compilation ...
```

All three parameters of `lt.enable()` are optional — `mode`, `trace_dir`, and `codegen_output` fall back to the `TL_LOWER_TRACE` / `TL_LOWER_TRACE_DIR` env vars (or sensible defaults) when omitted. See the parameter table below for details.

| Parameter | Description |
|-----------|-------------|
| `mode` | Force a trace mode: `"terminal"`, `"html"`, `"both"`, or `None` to disable. When omitted, falls back to the `TL_LOWER_TRACE` env var. |
| `trace_dir` | Base output directory. When omitted, falls back to `TL_LOWER_TRACE_DIR`, then `./tmp/lower_trace_dir`. |
| `codegen_output` | Path to save the generated codegen source (enables the edit-recompile workflow). When omitted, defaults to `<script_dir>/codegen.cpp`. Pass `None` explicitly to suppress. |

`enable()` is idempotent — calling it multiple times is safe.

##### When to use `reset()` and `disable()`

Both are **optional** and only needed in specific scenarios:

| Function | When to call | What it does |
|----------|--------------|--------------|
| `reset()` | Compiling **multiple kernels in the same process** and you want each kernel's report to start fresh (instead of accumulating into one combined report) | Clears collected records while keeping the hook active. Without it, records accumulate across compilations, tagged with `run2_`, `run3_`, … prefixes — which is desirable if you *want* to compare runs side by side. |
| `disable()` | You want to **disable tracing for subsequent compilations** within the same process (e.g. a long-running service that only traces the first kernel) | Restores the original `Pass.__call__`, `PassPipeline.lower`, and codegen FFIs, and clears all state. |

```python
from tilelang.tools import lower_trace as lt

lt.enable(mode="both")

# First kernel — traced.
kernel1 = tilelang.compile(func_a)

# Optional: clear records so kernel2 gets its own clean report.
# Omit this line if you prefer a combined multi-run report.
lt.reset()

# Second kernel — traced (into a fresh report if lt.reset() was called).
kernel2 = tilelang.compile(func_b)

# Optional: disable tracing for any further compilations.
lt.disable()
```

:::{note}
If neither `lt.reset()` nor `lt.disable()` is called, tracing stays active for the lifetime of the process and the final HTML report is generated automatically at exit. This is the simplest workflow and is sufficient for most one-off scripts.
:::

### HTML Report Features

The HTML report is a single self-contained file (no external assets) providing:

- **Sidebar** with per-pass navigation, status dots (● changed / ○ no-op / ✕ failed / ◆ codegen), and `+`/`−` line-count statistics. Collapsible and drag-resizable.
- **Phase tabs** to filter passes by pipeline phase.
- **Summary bar** with clickable filter badges (changed / failed / codegen).
- **Side-by-side diff** with GitHub-style coloring, word-level inline highlighting, and collapsible context (`↑↓ Expand` buttons reveal hidden equal lines).
- **Keyboard navigation** — `j`/`k` to move between passes, `Shift+E` to expand/collapse all, `F7` for Beyond-Compare-style manual alignment.
- **Dark/Light theme toggle** persisted via `localStorage`.
- **Copy buttons** for before/after IR of any pass.
- **Error boxes** — a failed pass shows its exception message alongside the IR *before* the crash.

```{figure} ../_static/img/lower_trace_html.png
:width: 600
:alt: Screenshot of the IR Lower Trace HTML report
:align: center

```

### Codegen Source Capture & Edit-Recompile Workflow

When tracing is enabled, the final codegen step (TIR → C/CUDA/HIP/…) is intercepted. The generated source is written to `<script_dir>/codegen.cpp` (or the path passed to `codegen_output=`), so you can inspect — and even edit — the code that will actually be compiled.

To support editing the generated code and re-running with your edits applied, IR Lower Trace maintains **three cooperating files**:

| File | Role |
|------|------|
| `codegen.cpp` | **Working copy** — user-editable. This is what gets compiled when you rerun. |
| `codegen.cpp.original` | **Baseline** — the codegen snapshot the working copy was last synced from. Written only on init or re-sync, never blindly overwritten. |
| `codegen.cpp.latest` | **Latest codegen output** — the actual output of the most recent run, overwritten every run for diff reference. |

On each run a three-way comparison (baseline / working copy / current codegen output) decides how to proceed:

| Situation | `codegen.cpp` vs `.original` | `.latest` vs `.original` | Action | Console tag |
|-----------|------------------------------|--------------------------|--------|-------------|
| No change | identical | identical | Compile with codegen output as-is | — |
| Codegen changed only | identical | **differs** | Regenerate `codegen.cpp` and `.original` from new codegen | `REGENERATED` |
| User edited only | **differs** | identical | Inject the working copy (`PATCHED`) | `PATCHED` |
| Both changed, working == latest | **differs** | **differs** (working matches latest) | Advance baseline; use working copy | `SYNCED` |
| Both changed, working != latest | **differs** | **differs** (working differs from latest) | **CONFLICT** — back up working copy → `.bak` (*conflict backup*) and old baseline → `.original.bak`, then regenerate from new codegen | `CONFLICT` |
| First run (no baseline) | — | — | Initialise `.original`, copy to `codegen.cpp` | (init) |
| `codegen.cpp` exists without baseline | — | — | Back up pre-existing `codegen.cpp` → `.bak` (*safety backup*), then initialise baseline | `INIT-BACKUP` |

> **Note on `.bak` files:** The backups created by `CONFLICT` and `INIT-BACKUP` serve different purposes. `INIT-BACKUP` preserves a pre-existing `codegen.cpp` of unknown origin before the trace tool takes it over. `CONFLICT` preserves the user's edits before a codegen change overwrites them. Recover `CONFLICT` edits with `diff codegen.cpp.original.bak codegen.cpp.bak`.

#### Typical Workflow

1. **Inspect** — Run once with `TL_LOWER_TRACE=1`. Open `codegen.cpp` to read the generated source.
2. **Edit** — Modify `codegen.cpp` (e.g. add a `printf`, tweak a loop). Do *not* touch `.original`.
3. **Rerun** — Run again. Because `codegen.cpp` differs from `.original` but codegen output is unchanged, you'll see `PATCHED from …/codegen.cpp` and your edited source is compiled.
4. **Iterate** — Keep editing and rerunning. Each run re-injects your working copy.
5. **If codegen itself changes** (e.g. you modified the TileLang program) — two outcomes:
   - If your edits happen to match the new codegen output → `SYNCED` (baseline advances, your edits preserved).
   - If both your edits and codegen changed and they differ → `CONFLICT`. Your working copy is backed up to `codegen.cpp.bak` and the old baseline to `codegen.cpp.original.bak`. Recover your edits with `diff codegen.cpp.original.bak codegen.cpp.bak`, then re-apply them against the freshly regenerated `codegen.cpp`.

:::{note}
**Backend requirements for edit-and-recompile.** The edit-and-recompile workflow requires a source-compiling execution backend — `nvrtc`, `cython`, or `cutedsl`. These backends use `*_without_compile` codegen FFIs that produce source-only modules, then compile the (edited) source string at runtime via NVRTC / Cython / CuTeDSL.

The default `tvm_ffi` backend pre-compiles device code to a binary (PTX/hsaco) from TIR during codegen. When the `tvm_ffi` backend is active and you edit `codegen.cpp`, you'll see a `NOTE` message indicating that your edits are recorded in the trace for diff viewing but were **not recompiled**. To use edit-and-recompile, switch to a source-compiling backend:

```python
# For CUDA targets:
tilelang.compile(..., execution_backend="nvrtc")

# For HIP targets:
tilelang.compile(..., execution_backend="cython")
```
:::

:::{note}
The `codegen_output` path defaults to `<script_dir>/codegen.cpp` when tracing is enabled. To disable codegen-to-disk entirely, pass `codegen_output=None` to `enable()`.
:::

### How It Works

IR Lower Trace installs three layers of transparent hooks (all via `monkey-patch`, restored by `disable()`):

1. **`tvm.ir.transform.Pass.__call__`** — every pass invocation is intercepted to capture `str(mod)` before and after, compute `+`/`−` line counts, and append a `LowerRecord`. Passes that run outside any pipeline window are tagged with the `unscoped` phase.
2. **`PassPipeline.lower`** (new architecture) or **phase functions** (legacy architecture) — sets the current phase context so passes invoked within a pipeline run are grouped under a label like `pipeline_c`. Legacy phase functions are discovered via AST scanning (`_discover_passes`) and bytecode inspection.
3. **Codegen FFI** (`target.build.tilelang_cuda`, `…_hip`, `…_c`, `…_llvm`, etc.) — captures the final TIR → source lowering and drives the three-file edit-recompile workflow described above.

Pass records are appended **at runtime** (not pre-registered), so conditional passes that are skipped at runtime — e.g. `LetInline` when `should_force_let_inline()` is `False` — simply do not appear, leaving no phantom/skipped slots. The HTML report is flushed **incrementally** after every pass (O(n) total cost), so partial results survive even a crash or `SIGKILL`.

When the same process compiles multiple kernels, each `PassPipeline.lower` invocation increments a run counter and tags phases with a `run2_`, `run3_`, … prefix; all records accumulate into a single report so you can compare runs side by side.

### Tips

- **Use `terminal` mode for quick checks** — the colored diff prints as passes run, so you can see changes in real time.
- **Use `html` mode for thorough analysis** — navigate across many passes, expand hidden context, and copy IR snippets.
- **Combine with `TL_LOWER_TRACE_DIR`** to direct reports to a specific location, e.g. when running in CI or comparing across runs.
- **The hook captures all passes** in the lowering pipeline, including those triggered internally by `tilelang.compile()`. This makes it useful for understanding the full compilation flow.
- **If you previously used `TILELANG_PASS_DIFF`**, switch to `TL_LOWER_TRACE` — it is a strict superset and is the tool that receives future improvements.

## Pass Diff: Observing IR Changes Across Passes

:::{admonition} Superseded — use IR Lower Trace
:class: warning

This tool has been superseded by **IR Lower Trace** (`TL_LOWER_TRACE`) documented above. New users should use IR Lower Trace directly — it provides phase context, codegen capture, multi-run accumulation, and an enhanced HTML report. `TILELANG_PASS_DIFF` is retained only for backward compatibility.
:::

TileLang programs are lowered through a sequence of compiler *passes*, each of which may transform the IR. Understanding exactly what each pass changes is essential for debugging incorrect transformations, unexpected optimizations, or missing passes.

TileLang provides a built-in **Pass Diff** tool that automatically captures the IR before and after every pass and generates a human-readable diff report. It works transparently — no code changes are required.

### Quick Start (Environment Variable)

The simplest way to enable pass diff is to set the `TILELANG_PASS_DIFF` environment variable before running your script:

```bash
# Colored diff printed to the terminal
TILELANG_PASS_DIFF=terminal python3 my_script.py

# Generate an HTML report
TILELANG_PASS_DIFF=html python3 my_script.py

# Both terminal output and HTML report
TILELANG_PASS_DIFF=both python3 my_script.py

# Disabled (default — zero overhead)
python3 my_script.py
```

The HTML report is saved to the directory specified by the `TILELANG_PASS_DIFF_OUTPUT` environment variable (default: `tmp/pass_diff_output`). Each run produces a timestamped file, e.g. `pass_diff_20260611_205421.html`.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TILELANG_PASS_DIFF` | Enable pass diff. Values: `0` (off), `terminal`, `html`, `both` | `0` |
| `TILELANG_PASS_DIFF_OUTPUT` | Output directory for HTML reports | `tmp/pass_diff_output` |

### HTML Report Features

The HTML report provides a rich diff viewer with:

- **Side-by-side before/after view** for each pass, with color-coded insertions and deletions
- **Collapsible pass sections** — click a pass header to expand or collapse its diff
- **Expand context** — when a diff hunk hides unchanged lines, a `⋯ Show N lines above/below` control lets you reveal them
- **Dark/Light theme toggle** — the default dark theme can be switched via the toolbar button; the preference is persisted across sessions
- **Copy buttons** — copy the before or after IR of any pass to the clipboard
- **Summary statistics** — total passes, changed passes, insertions, and deletions shown in the toolbar

```{figure} ../_static/img/pass_diff_html.png
:width: 600
:alt: Screenshot of the Pass Diff HTML report
:align: center

```

### Programmatic API

For more fine-grained control, you can use the `pass_diff` function directly in your code:

```python
from tilelang.utils.pass_diff import pass_diff
from tilelang import tvm

# Diff a single pass
pass_diff(func, tilelang.transform.ThreadSync("shared"))

# Diff a chain of named passes
pass_diff(func, [
    ("AnnotateDeviceRegions", tvm.tirx.transform.AnnotateDeviceRegions()),
    ("SplitHostDevice",       tvm.tirx.transform.SplitHostDevice()),
    ("ThreadSync",            tilelang.transform.ThreadSync("shared")),
], mode="html")
```

| Parameter | Description |
|-----------|-------------|
| `func` | A `PrimFunc` or `IRModule` to run passes on |
| `passes` | A single pass or a list of `(name, pass)` tuples |
| `mode` | `"terminal"`, `"html"`, or `"both"` (default: `"terminal"`) |
| `context` | Number of context lines in the unified diff (default: 3) |
| `html_path` | Output path for HTML report (default: `"pass_diff_report.html"`) |

### How It Works

When enabled, the hook monkey-patches `tvm.ir.transform.Pass.__call__` at import time. Every pass invocation is intercepted to capture the IR before and after, compute a unified diff, and emit the result in the chosen format. When disabled (the default), no patching occurs and there is zero overhead.

### Tips

- **Use `terminal` mode** for quick checks — the colored diff is printed as passes run, so you can see changes in real time.
- **Use `html` mode** for thorough analysis — the report lets you navigate across many passes, expand hidden context, and copy IR snippets.
- **Combine with `TILELANG_PASS_DIFF_OUTPUT`** to direct reports to a specific location, e.g. when running in CI or comparing across runs.
- **The hook captures all passes** in the lowering pipeline, including those triggered internally by `tilelang.compile()`. This makes it useful for understanding the full compilation flow.

## AutoDD: Automatic Delta Debugging

When dealing with complex TileLang programs that produce errors, manually isolating the bug can be tedious. **AutoDD** (Automatic Delta Debugging) is a built-in tool that automatically simplifies your program to the minimal code needed to reproduce a specific error.

### What is Delta Debugging?

Delta Debugging is an automated debugging technique that:
1. Takes a program that triggers a bug
2. Systematically removes code fragments
3. Checks if the simplified program still triggers the same bug
4. Produces the minimal code that reproduces the bug

AutoDD uses a Probability Distribution Driven Delta Debugging (PDD) algorithm for efficient minimization.

### Why Use AutoDD?

- **Large codebases**: Real projects often have hundreds of lines of configuration, helper functions, and logging
- **Hard-to-locate errors**: Error messages may point to TVM/CUDA internals rather than your TileLang code
- **Time-saving**: Manually deleting code to isolate bugs is very time-consuming

AutoDD can reduce a 200+ line program to just 30 lines, directly exposing the root cause.

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

Suppose you have a complex TileLang program with a GEMM shape mismatch bug:

```python
# buggy_matmul.py (200+ lines)
@tilelang.jit
def buggy_matmul(M, N, K, block_M, block_N, block_K, ...):
    @T.prim_func
    def matmul_kernel(...):
        with T.Kernel(...) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug: should be (block_K, block_N)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # ... lots of other code ...
            T.gemm(A_shared, B_shared, C_local)  # Error here
    return matmul_kernel
```

Run AutoDD to minimize:

```bash
python -m tilelang.autodd buggy_matmul.py --err-msg "Dimension mismatch" -o minimized.py -j 4
```

AutoDD will produce a minimal reproduction:

```python
# minimized.py (~30 lines)
import tilelang.language as T

def buggy_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, *args, **kwargs):
    @T.prim_func
    def matmul_kernel():
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug exposed!
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.gemm(A_shared, B_shared, C_local)
```

### How AutoDD Works

AutoDD uses AST (Abstract Syntax Tree) analysis with multiple rewrite rules:

1. **Fast Reducers**: Remove statements, simplify if/for constructs
2. **Canonicalizers**: Expand with statements, add `*args, **kwargs` for compatibility
3. **Simplifiers**: Replace expressions with constants, simplify function calls
4. **Slow Reducers**: Remove arbitrary expressions, reduce integer constants

### Tips

- **Error message matching**: Use a unique substring from the error output
- **Timeout**: Increase `--timeout` for programs with long compilation times
- **Parallel jobs**: Use `-j 4` or higher to speed up minimization
- **Backend**: Try `--backend subproc` if `runner` is unstable

### Complete Example

A complete example is available in `examples/autodd/`:
- `tilelang_buggy.py`: A complex program with a bug (~200 lines)
- `tilelang_minimized_expected.py`: Expected output after AutoDD (~30 lines)
- `README.md`: Detailed documentation

## Conclusion

By carefully examining intermediate representations (IR) before final code generation—and by leveraging runtime printing through `T.print`—one can quickly diagnose where index calculations, copy logic, or other kernel operations deviate from the intended behavior. The **IR Lower Trace** tool (`TL_LOWER_TRACE`) complements this by providing automatic, pass-by-pass visibility into every IR transformation — including the final codegen step — making it easy to pinpoint exactly which pass introduces an unexpected change. (The older **Pass Diff** tool is retained for backward compatibility but is superseded by IR Lower Trace.) This three-pronged approach (inspecting IR transformations, observing pass-level diffs, and using runtime prints) is often sufficient for resolving generation and correctness issues in TileLang programs.

For complex programs where manual debugging is tedious, **AutoDD** provides automated delta debugging to quickly isolate the minimal code that reproduces a bug.

For advanced performance tuning (e.g., analyzing memory bandwidth or occupancy), more specialized profiling tools such as **Nsight Compute**, **rocProf**, or vendor-specific profilers may be required. Those aspects will be covered in future documents.
