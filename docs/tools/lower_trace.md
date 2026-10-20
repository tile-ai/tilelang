# IR Lower Trace

IR Lower Trace is provided by the optional
[tilelang-devkit](https://github.com/easy-tilelang/tilelang-devkit) package.
It captures compiler pass IR and generated source, with terminal and HTML diffs.

Install it into the same Python environment as TileLang:

```bash
pip install tilelang-devkit
TL_LOWER_TRACE=both python my_script.py
```

TileLang discovers installed tools through the `tilelang.instrumentation_tools`
entry-point group. The devkit owns the `TL_LOWER_TRACE` and
`TL_LOWER_TRACE_DIR` settings; no tracing implementation is bundled in TileLang.

For a selected pass chain:

```python
import tilelang_devkit.lower_trace as lt

results = lt.lower_trace(func, my_pass, mode="terminal")
```

The former `tilelang.tools.lower_trace` import has moved to
`tilelang_devkit.lower_trace`. See the
[full manual](https://github.com/easy-tilelang/tilelang-devkit/blob/main/docs/lower_trace.md)
for output paths, the Python API, and the source edit-and-recompile workflow.
