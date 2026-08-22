"""A Python integer coalesced-width hint must control loop partitioning.

The 132-element loop naturally minimizes padding with a scalar partition over
64 threads.  An explicit width of four instead requests a four-lane partition,
so the two variants must produce different parallel-loop layouts.
"""

import tilelang.language as T


def _elementwise(coalesced_width=None, *, use_annotations=False):
    length = 132

    @T.prim_func
    def main(A: T.Tensor((length,), T.float32), B: T.Tensor((length,), T.float32)):
        with T.Kernel(1, threads=64):
            if use_annotations:
                for i in T.Parallel(length, annotations={"coalesced_width": coalesced_width}):
                    B[i] = A[i]
            elif coalesced_width is not None:
                for i in T.Parallel(length, coalesced_width=coalesced_width):
                    B[i] = A[i]
            else:
                for i in T.Parallel(length):
                    B[i] = A[i]

    return main


VARIANTS = {
    "default": lambda: _elementwise(),
    "keyword_int": lambda: _elementwise(4),
    "annotation_int": lambda: _elementwise(4, use_annotations=True),
}


def check(variant, model, result):
    assert len(result["loops"]) == 1
    layout = next(iter(result["loops"].values()))
    expected_output_shape = [3] if variant == "default" else [4]
    assert layout["output_shape"] == expected_output_shape, f"{variant} must infer output shape {expected_output_shape}, got: {layout}"
