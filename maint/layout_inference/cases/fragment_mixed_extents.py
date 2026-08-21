"""Ensure a shared fragment layout covers the largest parallel-loop extent."""

import tilelang.language as T


def _mixed_extents(small_extent, full_extent, threads):
    @T.prim_func
    def main(C: T.Tensor((full_extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            fragment = T.alloc_fragment((full_extent,), T.float32)
            T.clear(fragment)
            for i in T.Parallel(small_extent):
                fragment[i] = 5.0
            for i in T.Parallel(full_extent):
                C[i] = fragment[i] + 1.0

    return main


VARIANTS = {
    "issue2957_100_to_256_t128": lambda: _mixed_extents(100, 256, 128),
}


def check(variant, model, result):
    del variant, model
    assert result["buffers"]["fragment"]["input_shape"] == [256]
    for key, layout in result["loops"].items():
        extent = int(key.split("[")[1].split("]")[0])
        assert layout["input_shape"][0] >= extent
