# Layout Visualization

TileLang provides two related ways to inspect data layouts:

- `tilelang.tools.plot_layout` plots a `T.Layout` or `T.Fragment` that you
  already have.
- Layout inference visualization prints fragment layouts discovered by the
  compiler and can save a plot for each inferred two-dimensional fragment.

Use the first workflow while designing a mapping and the second when checking
the mapping selected for buffers in a compiled kernel.

## Installation

PNG, PDF, and SVG output requires Matplotlib. Install TileLang's visualization
extra:

```bash
pip install "tilelang[vis]"
```

Text-only compiler output does not create a Matplotlib figure.

## Plot a Layout Directly

This example plots a 4 by 4 transpose mapping:

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))

plot_layout(
    transpose,
    save_directory="./tmp",
    name="transpose_4x4",
    formats="png",
)
```

The call writes `./tmp/transpose_4x4.png` and prints the saved path. In the
default input view, each grid cell represents an input position and its label
is the flattened output position. Set `view="output"` to put output positions
on the grid and label them with their source coordinates.

For a `T.Fragment`, cells are colored by thread and labeled with both the
thread ID (`T`) and thread-local register index (`L`):

```python
plot_layout(fragment, name="mma_load", formats="pdf")
```

## `plot_layout` API

```python
plot_layout(
    layout,
    save_directory="./tmp",
    name="layout",
    colormap=None,
    verbose=False,
    formats="pdf",
    view="input",
    grid_shape=None,
)
```

| Parameter | Meaning |
| --- | --- |
| `layout` | A `T.Layout` or `T.Fragment`. Other objects raise `TypeError`. |
| `save_directory` | Directory created for output files. Defaults to `./tmp`. |
| `name` | Base filename without an extension. |
| `colormap` | Matplotlib colormap name. The defaults are `Spectral` for `T.Layout` and `RdPu` for `T.Fragment`. |
| `verbose` | Print each mapping while building the plot. |
| `formats` | A string: `pdf`, `png`, `svg`, `all`, or a comma-separated combination such as `png,svg`. |
| `view` | For `T.Layout`, use `input` or `output`. It is not used for `T.Fragment`. |
| `grid_shape` | For the input view of a `T.Layout`, override the display grid with `(rows, columns)`. Its product must equal the number of input elements. |

`plot_layout` returns `None`. With `formats="all"`, it writes PDF, PNG, and SVG
files. A higher-dimensional `T.Layout` is flattened to a two-dimensional grid:
all dimensions except the last form the row coordinate, and the last dimension
forms the column coordinate.

## Visualize Inferred Layouts

Enable the compiler pass through `pass_configs` on a JIT kernel:

```python
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
    }
)
def kernel(A, block_M, block_N):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), T.float16)
    B = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        fragment = T.alloc_fragment((block_M, block_N), T.float16)
        T.copy(A[by * block_M, bx * block_N], fragment)
        T.copy(fragment, B[by * block_M, bx * block_N])

    return B
```

Compile or invoke the kernel normally. For each inferred `T.Fragment` found in
a block's `layout_map` annotation, the pass prints output like:

```text
C_local inferred layout:
  Shape: [32, 32] -> [8]
  Thread: <thread-index expression>
  Index:  [<local-index expression>]
  Replicate:  1
```

Image files are written under `./tmp` with names derived from the buffer, such
as `C_local_layout.svg`.

`TL_LAYOUT_VISUALIZATION_ENABLE` defaults to `False`. When it is enabled,
`TL_LAYOUT_VISUALIZATION_FORMATS` accepts `txt`, `png`, `pdf`, `svg`, `all`, or
a comma-separated combination. Omitting the formats setting selects text-only
output. The compiler prints the textual mapping whenever visualization is
enabled; `txt` does not add an image format.

## Limitations

- Direct `T.Fragment` plotting requires a two-dimensional input shape and
  single-valued thread and local-index mappings.
- Compiler-generated images are limited to inferred fragments with
  two-dimensional input shapes. Other inferred fragment shapes are still
  printed, but image generation is skipped with a warning.
- Plots enumerate layout elements in Python. Large shapes can produce large
  figures and take substantial time and memory.
- The output-space view infers its shape from mapped coordinates. Mappings that
  are non-bijective or sparse may overwrite a cell or leave empty cells, so the
  plot should be treated as a diagnostic view rather than layout validation.

## Examples

- [Layout transforms and swizzles](https://github.com/tile-ai/tilelang/tree/main/examples/plot_layout)
- [CUDA MMA fragment layouts](https://github.com/tile-ai/tilelang/blob/main/examples/plot_layout/fragment_mma_load_a.py)
- [AMD MFMA fragment layouts](https://github.com/tile-ai/tilelang/blob/main/examples/plot_layout/fragment_mfma_load_a.py)
- [Compiler layout inference visualization](https://github.com/tile-ai/tilelang/blob/main/examples/visual_layout_inference/visual_layout_inference.py)
