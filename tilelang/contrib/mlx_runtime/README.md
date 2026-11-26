# TileLang–MLX native bridge

This optional extension connects TileLang-generated Metal kernels to MLX's
native primitive and execution APIs. It is built separately from TileLang;
the Python entry point remains `tilelang.contrib.mlx`.

From the TileLang repository root on Apple Silicon running macOS 26 or later,
with Xcode Command Line Tools installed:

```bash
python -m pip install ./tilelang/contrib/mlx_runtime
python examples/metal/mlx_cache_update.py
```

The distribution is named `tilelang-mlx-runtime` and exports `_tilelang_mlx`.
It requires MLX 0.32.4 and builds with nanobind 3.1.0 because the native APIs
are version-specific.

See [the MLX integration guide](../../../examples/metal/README.mlx.md) for the API,
supported layouts, state ownership rules, and tests.
