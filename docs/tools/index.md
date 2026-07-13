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
| Compare IR before and after compiler passes | [Pass Diff](pass_diff.md) | `TILELANG_PASS_DIFF` |
| Record CUDA kernel markers, ranges, and payloads | [IKET](iket.md) | `tilelang.tools.cuda.iket` |

## Choosing a Tool

- Use **Analyzer** before benchmarking when you need a rough roofline-style
  estimate from a `PrimFunc` or `IRModule`.
- Use **Layout Visualization** to inspect how logical indices map to threads and
  local indices.
- Use **AutoDD** after obtaining a stable failure signature and before filing a
  large reproducer.
- Use **Pass Diff** when a lowering pass introduces an unexpected IR change.
- Use **IKET** for CUDA timeline instrumentation when the external IKET runtime
  is available.

Kernel-level debugging primitives such as `T.print` remain part of the TileLang
language. See {doc}`../tutorials/debug_tools_for_tilelang` for an end-to-end
debugging workflow that combines language primitives with these tools.
