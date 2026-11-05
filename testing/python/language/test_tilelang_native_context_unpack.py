"""Context targets retain the source DSL's scalar and tuple binding forms."""

from contextlib import contextmanager

import pytest
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


@pytest.mark.parametrize("form", ["scalar", "tuple", "list"])
def test_single_kernel_dimension_accepts_each_existing_binding_form(form):
    kernel_scope = T.Kernel

    @T.prim_func
    def kernel(A: T.Tensor((2,), T.int32)):
        if form == "tuple":
            with kernel_scope(2, threads=1) as (pid,):
                A[pid] = 1
        elif form == "list":
            with kernel_scope(2, threads=1) as [pid]:
                A[pid] = 1
        else:
            with kernel_scope(2, threads=1) as pid:
                A[pid] = 1

    assert tirx.analysis.check_well_formed(kernel)
    assert not tirx.analysis.undefined_vars(kernel)
    launch = next(region for region in collect_nodes(kernel, ir.RegionStmt) if region.op.name == "tl.launch_kernel")
    (store,) = collect_nodes(kernel, ir.TensorStore)
    assert store.indices[0].same_as(launch.body_params[0])


def test_context_items_bind_in_order_and_exit_the_python_context_once():
    events = []

    @contextmanager
    def choose_extent():
        events.append("enter")
        try:
            yield (2,)
        finally:
            events.append("exit")

    @T.prim_func
    def kernel(A: T.Tensor((2,), T.int32)):
        with choose_extent() as (extent,), T.Kernel(extent, threads=1) as (pid,):
            A[pid] = extent

    assert events == ["enter", "exit"]
    assert tirx.analysis.check_well_formed(kernel)
    assert not tirx.analysis.undefined_vars(kernel)
    (store,) = collect_nodes(kernel, ir.TensorStore)
    assert int(store.value) == 2


def test_make_tensor_from_integer_pointer_has_explicit_native_pointer_type():
    @T.prim_func
    def kernel(pointers: T.Tensor((1,), T.ptr), output: T.Tensor((16, 16), T.float16)):
        with T.Kernel(1, threads=128):
            source = T.make_tensor(pointers[0], (16, 16), T.float16)
            T.copy(source, output)

    assert tirx.analysis.check_well_formed(kernel)
    assert not tirx.analysis.undefined_vars(kernel)
    (recast,) = collect_calls(kernel, op="tirx.reinterpret")
    assert isinstance(recast.ty, ir.PtrType)
    assert str(recast.ty.element_type) == "float16"
    assert str(recast.ty.scope) == "global"
