"""Shared Metal reduce test utilities.

The core kernel constructor is derived from the upstream public constructor
``_make_allreduce_dim0_scale_kernel`` in
``testing/python/language/test_tilelang_language_reduce.py``. Upstream keeps
that helper private to its own test file and does not parameterize the
Metal-only concerns (explicit threadgroup extent, fp16/bf16 destinations,
and ``clear=False`` duplicate-buffer updates), so the Metal tests share this
copy here instead of duplicating it in every test module.
"""

import tilelang.language as T


def make_allreduce_dim0_scale_kernel(
    reduce_fn, logical_width, scale, threads=None, dtype="float32", clear=True
):
    """Allreduce-on-dim-0 kernel constructor (upstream-derived).

    Source: upstream ``_make_allreduce_dim0_scale_kernel`` in
    ``testing/python/language/test_tilelang_language_reduce.py``, extended for
    the Metal backend with:

    - ``threads``: defaults to ``logical_width * scale`` (N == nt). Supplying
      an explicit value decouples the threadgroup extent from nt so
      misaligned threadgroups are reachable through the same public
      constructor.
    - ``dtype``: covers the fp32/fp16/bf16 paths.
    - ``clear=False``: exercises the duplicate-buffer update path (Phase 3)
      that accumulate-into-dst reductions take on Metal.
    """
    if threads is None:
        threads = logical_width * scale

    @T.prim_func
    def kernel(
        A: T.Tensor((logical_width, scale), dtype),
        B: T.Tensor((scale,), dtype),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((logical_width, scale), dtype)
            dst = T.alloc_fragment((scale,), dtype)
            T.copy(A, src)
            reduce_fn(src, dst, dim=0, clear=clear)
            T.copy(dst, B)

    return kernel
