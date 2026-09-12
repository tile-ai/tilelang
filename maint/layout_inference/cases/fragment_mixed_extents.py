"""Ensure a shared fragment layout covers the largest parallel-loop extent."""

import tilelang.language as T


def _mixed_extent_read(small_extent, full_extent, threads):
    @T.prim_func
    def main(C: T.Tensor((full_extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((full_extent,), T.float32)
            T.clear(frag)
            for i in T.Parallel(small_extent):
                frag[i] = 5.0
            for i in T.Parallel(full_extent):
                C[i] = frag[i] + 1.0

    return main


def _mixed_extent_write(small_extent, full_extent, threads):
    @T.prim_func
    def main(C: T.Tensor((full_extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((full_extent,), T.float32)
            T.clear(frag)
            for i in T.Parallel(small_extent):
                frag[i] = 7.0
            for i in T.Parallel(full_extent):
                frag[i] = 5.0
            for i in T.Parallel(full_extent):
                C[i] = frag[i]

    return main


def _equal_extents(full_extent, threads):
    @T.prim_func
    def main(C: T.Tensor((full_extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            frag = T.alloc_fragment((full_extent,), T.float32)
            T.clear(frag)
            for i in T.Parallel(full_extent):
                frag[i] = 5.0
            for i in T.Parallel(full_extent):
                C[i] = frag[i] + 1.0

    return main


def _independent_fragments(small_extent, full_extent, threads):
    @T.prim_func
    def main(C: T.Tensor((small_extent + full_extent,), T.float32)):
        with T.Kernel(1, threads=threads):
            small_frag = T.alloc_fragment((small_extent,), T.float32)
            full_frag = T.alloc_fragment((full_extent,), T.float32)
            T.clear(small_frag)
            T.clear(full_frag)
            for i in T.Parallel(small_extent):
                small_frag[i] = 7.0
            for i in T.Parallel(full_extent):
                full_frag[i] = 5.0
            for i in T.Parallel(small_extent):
                C[i] = small_frag[i]
            for i in T.Parallel(full_extent):
                C[small_extent + i] = full_frag[i]

    return main


VARIANTS = {
    "issue2957_100_to_256_t128": lambda: _mixed_extent_read(100, 256, 128),
    "issue2957_write_100_to_256_t128": lambda: _mixed_extent_write(100, 256, 128),
    "equal_extents_256_t128": lambda: _equal_extents(256, 128),
    "independent_fragments_100_and_256_t128": lambda: _independent_fragments(100, 256, 128),
}

EXPECTED_BUFFER_SHAPES = {
    "issue2957_100_to_256_t128": {"frag": [256]},
    "issue2957_write_100_to_256_t128": {"frag": [256]},
    "equal_extents_256_t128": {"frag": [256]},
    "independent_fragments_100_and_256_t128": {
        "small_frag": [100],
        "full_frag": [256],
    },
}


def check(variant, model, result):
    del model
    buffer_shapes = {name: layout["input_shape"] for name, layout in result["buffers"].items()}
    assert buffer_shapes == EXPECTED_BUFFER_SHAPES[variant]
    for key, layout in result["loops"].items():
        extent = int(key.split("[")[1].split("]")[0])
        assert layout["input_shape"][0] >= extent
