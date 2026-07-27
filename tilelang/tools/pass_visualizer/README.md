# Pass Visualizer

An interactive, pass-by-pass **structure-tree** visualizer for TileLang kernels.

This is a debugging complement to [`tilelang.utils.pass_diff`](../pass_diff.py).
Where `pass_diff` shows a line-level diff of the **TVMScript text**, this tool
renders the IR as an **`SBlock` structure tree** — the block nesting plus
`reads` / `writes` / `alloc_buffers` / `annotations` fields — and expands every
tile op by field name. It emits a single self-contained, interactive HTML file
that steps through each CUDA lowering pass.

Use it when debugging **structural** passes (layout inference, warp
specialization, pipelining), where what matters is how the IR's block structure
and operator semantics change, not just which text lines moved.

## How it differs from `pass_diff`

| Aspect | `pass_diff` | `pass_visualizer` |
|--------|-------------|-------------------|
| Compared object | TVMScript text lines | `SBlock` structure tree |
| Operator display | Raw one-liner, positional args | Expanded **by field name** (`M=64`, `K=32`, `policy=0`) |
| Highlighting | Generic `+` / `-` | Per-class: tile op / sync primitive / lowered hardware intrinsic |
| Trigger | Environment-variable hook, captures the real full pipeline | Explicit CLI, runs the focused lowering prologue |

## Usage

```bash
python -m tilelang.tools.pass_visualizer.viewer \
    tilelang/tools/pass_visualizer/examples/gemm_relu.py \
    --set M=1024 --set N=1024 --set K=1024 \
    --set block_M=128 --set block_N=128 --set block_K=32 \
    --out gemm_relu_passes.html
```

This writes `gemm_relu_passes.html` (the interactive browser) and a sibling
`gemm_relu_passes.txt` (a greppable text dump of the same per-pass trees).

| Argument | Description |
|----------|-------------|
| `path` | Python file containing a `@tilelang.jit` kernel (positional) |
| `--factory` | Name of the kernel to analyze (default: first discovered) |
| `--target` | Compilation target (default: `auto`) |
| `--set K=V` | Argument forwarded to the kernel factory (repeatable) |
| `--out` | Output HTML path (default: `<kernel>_passes.html` next to the source) |

## What the HTML shows

- **Left pane**: the ordered pass list, each tagged `changed` / `no-op` with an
  added/removed line count. Click a pass — or use the ↑/↓ keys — to step through
  the pipeline.
- **Right pane**: the structure tree for the selected pass, with lines **added**
  by that pass highlighted green and lines it **removed** shown ghosted red.
- **Operator highlighting**: tile ops (`T.gemm`, `T.copy`, …), synchronization
  primitives, and lowered hardware intrinsics (`ptx_mma`, `tma_load`, …) are
  each colored distinctly, so you can follow a `T.copy` as it lowers into
  TMA/PTX intrinsics.

## Programmatic API

```python
from tilelang.tools.pass_visualizer.viewer import build_pass_data, emit_html

name, stages = build_pass_data(
    "path/to/kernel.py", factory=None, target="auto",
    kwargs={"M": 1024, "N": 1024, "K": 1024,
            "block_M": 128, "block_N": 128, "block_K": 32},
    source=open("path/to/kernel.py").read(),
)
html = emit_html(name, stages)
```

## Files

- `viewer.py` — CLI entry point; per-pass capture, diffing, and HTML/text emission.
- `core.py` — kernel loading, the CUDA lowering pass pipeline, and the structure-tree renderer.
- `examples/gemm_relu.py` — a small fused GEMM + bias + ReLU kernel used as demo input.
