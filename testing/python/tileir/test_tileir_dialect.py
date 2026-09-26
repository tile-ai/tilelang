"""TileIR hint ownership and frontend tracing contracts."""

import inspect

import pytest

import tilelang
from tilelang import language as Tcuda
from tilelang.language import common
from tilelang.language.kernel import is_kernel_launch_factory
from tilelang.tileir import language as T
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def test_tileir_dialect_owns_its_hints():
    assert T.__tilelang_dialect__ == "tileir"
    assert set(T.__all__) == set(common.__all__)
    assert is_kernel_launch_factory(T.Kernel)
    for name in common.__all__:
        if name not in {"Kernel", "copy"}:
            assert getattr(T, name) is getattr(common, name)
    for name in ("num_ctas", "occupancy", "num_worker_warps", "tileir_hints"):
        assert name in inspect.signature(T.Kernel).parameters
        assert name not in inspect.signature(common.Kernel).parameters
        assert name not in inspect.signature(Tcuda.Kernel).parameters
    assert "latency" in inspect.signature(T.copy).parameters
    assert "latency" not in inspect.signature(common.copy).parameters
    assert "latency" not in inspect.signature(Tcuda.copy).parameters


@pytest.mark.parametrize("tileir_hints", [None, {}])
def test_tileir_eager_launch_and_copy_annotations(tileir_hints):
    @tilelang.jit
    def hinted(A, B):
        A: T.Tensor((128,), "float32")
        B: T.Tensor((128,), "float32")
        with T.Kernel(1, threads=128, num_ctas=2, occupancy=4, num_worker_warps=4, tileir_hints=tileir_hints) as bx:
            S = T.alloc_shared((128,), "float32")
            T.copy(A[bx * 128], S, latency=2, disable_tma=True, annotations={"tileir.latency": 6})
            T.copy(S, B[bx * 128])

    func = hinted.get_tir(None, None)
    launch_annotations = {}
    copy_annotations = {}

    def visit(node):
        if isinstance(node, tirx.SBlock):
            launch_annotations.update({str(k): v for k, v in node.annotations.items()})
        if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.copy":
            copy_annotations.update({str(k): v for k, v in node.annotations.items()})

    post_order_visit(func.body, visit)
    assert int(launch_annotations["tileir.num_ctas"]) == 2
    assert int(launch_annotations["tileir.occupancy"]) == 4
    assert int(launch_annotations["tileir.num_worker_warps"]) == 4
    assert "tileir.hints" not in launch_annotations
    assert int(copy_annotations["tileir.latency"]) == 6
    assert bool(copy_annotations["disable_tma"])
