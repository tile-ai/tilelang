# Source-Span Visualization

`example_tirx_span_viz.py` demonstrates `tilelang.tools.tirx_span_viz`, which
prints the official TVMScript of a `PrimFunc` or `IRModule` with each
statement's `file:line:column` source location appended as a trailing
`# ...` comment. It accepts either form, so it can be called after any pass in
`tilelang/cuda/pipeline.py`.

## Run the example

```bash
python examples/tirx_span_viz/example_tirx_span_viz.py
```

The example needs no GPU: it builds a `@tilelang.jit` GEMM, retrieves its
parsed TIR with `get_tir(...)`, and prints the span-annotated TVMScript.

## Programmatic use

```python
from tilelang.tools.tirx_span_viz import dump_tirx_with_span, render_tirx_with_span

dump_tirx_with_span(func)  # print a PrimFunc / IRModule with spans
text = render_tirx_with_span(func)  # or get the string back
```
