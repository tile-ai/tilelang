"""Regression tests for issue #4134.

This module enables ``from __future__ import annotations`` (PEP 563), so every
annotation below is a *string* at decoration time. The eager frontend's
``get_type_hints`` used to evaluate non-dtype string annotations through
``typing._eval_type``, whose ``_type_check`` rejects TileLang annotations such
as ``T.Tensor((128,), "float32")`` because they evaluate to ``tirx.Buffer``
objects rather than Python types (on CPython != 3.11). Decorating a kernel with
``@T.prim_func`` then aborted with
``TypeError: Forward references must evaluate to types. Got buffer.``
"""

from __future__ import annotations

import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def test_prim_func_tensor_annotations_under_future_annotations():
    @T.prim_func
    def main(
        x: T.Tensor((128,), "float32"),
        y: T.Tensor((128,), "float32"),
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                y[i] = x[i]

    assert isinstance(main, tvm.tirx.PrimFunc)
    assert len(main.params) == 2
    stores = []
    tvm.tirx.stmt_functor.post_order_visit(
        main.body,
        lambda node: stores.append(node) if isinstance(node, tvm.tirx.BufferStore) else None,
    )
    assert len(stores) == 1


def test_prim_func_scalar_dtype_annotations_under_future_annotations():
    @T.prim_func
    def main(t_1: T.float32, t_2: T.bool):
        pass

    assert isinstance(main, tvm.tirx.PrimFunc)
    assert len(main.params) == 2


def test_macro_tensor_annotation_under_future_annotations():
    @T.macro
    def add(x: T.Tensor((128,), "float32")):
        return x + 1

    assert add is not None


def test_macro_explicitly_quoted_ref_annotation_under_future_annotations():
    # With future annotations, `x: "T.Ref"` stores the source text '"T.Ref"'
    # (quotes included). get_type_hints must resolve the nested string all the
    # way to the Ref class so macro_arg passes the argument by reference;
    # otherwise `x += 1` would silently rebind a local copy instead of
    # storing into the caller's buffer.
    @T.macro
    def add_one(x: "T.Ref"):
        x += 1

    @T.prim_func
    def main(A: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            add_one(A[0])

    assert isinstance(main, tvm.tirx.PrimFunc)
    stores = []
    tvm.tirx.stmt_functor.post_order_visit(
        main.body,
        lambda node: stores.append(node) if isinstance(node, tvm.tirx.BufferStore) else None,
    )
    # The augmented assignment must reach the caller's buffer A.
    assert len(stores) == 1
    assert stores[0].buffer.name == "A"


if __name__ == "__main__":
    tilelang.testing.main()
