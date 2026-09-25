# Source-Span Visualization

`tilelang.tools.tirx_span_viz` prints the official TVMScript of a `PrimFunc`
or `IRModule` with each statement's `file:line:column` source location appended
as a trailing `# ...` comment, so lowered IR can be traced back to the Python
line that produced it.

The official TVMScript printer never reads the `span` field, which is why
`print(func)` does not show locations. This tool walks the IR with
`tvm.tirx.stmt_functor.post_order_visit`, reads each statement's span via
`tilelang.ir.get_stmt_span`, and hands the locations to the printer's
`obj_to_annotate` support, which renders them as trailing comments.

## Usage

```python
from tilelang.tools.tirx_span_viz import dump_tirx_with_span, render_tirx_with_span

dump_tirx_with_span(func)  # print a PrimFunc / IRModule with spans
text = render_tirx_with_span(func)  # or get the string back
```

Both functions accept a `PrimFunc` or an `IRModule`, so they can be called at
any stage — a freshly parsed `PrimFunc`, or an `IRModule` after any pass in
`tilelang/cuda/pipeline.py`. Statements synthesized by passes without a source
span are left un-annotated.

## Example

A runnable example lives in
[`examples/tirx_span_viz/example_tirx_span_viz.py`](https://github.com/tile-ai/tilelang/tree/main/examples/tirx_span_viz):
it builds a `@tilelang.jit` GEMM, retrieves the parsed TIR with `get_tir(...)`,
and prints the span-annotated TVMScript. No GPU is required.

## API

- `render_tirx_with_span(func_or_mod) -> str` — return the span-annotated
  TVMScript string.
- `dump_tirx_with_span(func_or_mod, file=None) -> None` — print the
  span-annotated TVMScript, optionally to `file`.
