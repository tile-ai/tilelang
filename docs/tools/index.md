# Tools

TileLang includes utilities for inspecting kernels, visualizing layouts,
reducing failing programs, observing compiler transformations, and collecting
GPU timeline events. These tools are separate from the kernel language: import
or invoke only the tool needed for the current workflow.

| Task | Tool | Entry point |
| --- | --- | --- |
| Estimate compute and memory cost from TIR | [Performance Analyzer](analyzer.md) | `tilelang.tools.Analyzer` |
| Inspect thread and data mappings | [Layout Visualization](layout_visualization.md) | `tilelang.tools.plot_layout` |
| Reduce a failing program to a smaller reproducer | [AutoDD](autodd.md) | `python -m tilelang.autodd` |
| Track IR changes across the full lowering pipeline (recommended) | [IR Lower Trace](lower_trace.md) | `TL_LOWER_TRACE` |
| Compare IR before and after compiler passes (legacy) | [Pass Diff](pass_diff.md) | `TILELANG_PASS_DIFF` |
| Record CUDA kernel markers, ranges, and payloads | [IKET](iket.md) | `tilelang.tools.cuda.iket` |

## Choosing a Tool

- Use **Analyzer** before benchmarking when you need a rough roofline-style
  estimate from a `PrimFunc` or `IRModule`.
- Use **Layout Visualization** to inspect how logical indices map to threads and
  local indices.
- Use **AutoDD** after obtaining a stable failure signature and before filing a
  large reproducer.
- Use **IR Lower Trace** to observe every pass and the final codegen step of a
  full lowering pipeline. It supersedes **Pass Diff** and is the tool that
  receives future improvements.
- Use **Pass Diff** only when you need the legacy `TILELANG_PASS_DIFF` workflow.
- Use **IKET** for CUDA timeline instrumentation when the external IKET runtime
  is available.

Kernel-level debugging primitives such as `T.print` remain part of the TileLang
language. See {doc}`../tutorials/debug_tools_for_tilelang` for an end-to-end
debugging workflow that combines language primitives with these tools.
