# Layout Visualization Examples

These examples visualize TileLang layouts and fragments with
`tilelang.tools.plot_layout`. See the
[Layout Visualization guide](../../docs/tools/layout_visualization.md) for
installation, API details, compiler-driven layout visualization, and current
limitations.

| File | Coverage |
| --- | --- |
| `layout_transform.py` | Layout composition and transforms |
| `layout_swizzle.py` | Shared-memory swizzle mappings |
| `fragment_mma_load_a.py` | NVIDIA MMA fragment mapping |
| `fragment_mfma_load_a.py` | AMD MFMA fragment mapping |

Install the visualization dependency and run an example from the repository
root:

```bash
pip install "tilelang[vis]"
python examples/plot_layout/fragment_mma_load_a.py
```

Plots are written under `./tmp` unless the example supplies another output
directory. A representative fragment plot is available at
[`images/base_layout.png`](images/base_layout.png).
