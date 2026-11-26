"""In-place cache-update example for Metal issue #3222.

Run: python examples/metal/mlx_cache_update.py
Requires the optional tilelang-mlx-runtime package and an Apple Silicon GPU.
"""

import mlx.core as mx
import numpy as np

from tilelang.contrib.mlx import compile as compile_mlx
from tilelang.metal import language as T


@T.prim_func
def cache_update(
    old: T.Tensor((64,), T.float32),
    delta: T.Tensor((16,), T.float32),
):
    with T.Kernel(1, threads=32):
        for i in T.Parallel(64):
            if i >= 16 and i < 32:
                old[i] += delta[i - 16]


def main():
    update = compile_mlx(cache_update, inplace_idx=0)
    cache = mx.arange(64).astype(mx.float32)
    original = cache
    delta = mx.ones((16,), dtype=mx.float32)
    # MLX evaluates the dependency chain lazily. Rebind the returned alias to order writes.
    cache = update(cache, delta)
    cache = update(cache, delta)
    mx.eval(cache)
    expected = np.arange(64, dtype=np.float32)
    expected[16:32] += 2
    np.testing.assert_array_equal(np.array(cache), expected)
    np.testing.assert_array_equal(np.array(original), expected)
    print("PASS MLX in-place cache update; original allocation updated")


if __name__ == "__main__":
    main()
