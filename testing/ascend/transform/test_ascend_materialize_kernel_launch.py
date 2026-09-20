"""Ascend's launch lowering through MaterializeKernelLaunch.

Upstream's target-neutral ``T.Kernel`` records a 1-D grid, three
``tl.launch_thread_idx`` placeholders and the launch annotations. What those
mean is decided per backend by ``MaterializeKernelLaunch``. Ascend is the one
backend that needs the two decisions decoupled:

  * the NPU launch is a real 1-D core grid, so the ``blockIdx.x`` loop must
    become a ``thread_extent`` AttrStmt (like CUDA), and
  * there is no threadIdx at kernel scope, so the placeholders must be dropped
    (like CPU). Thread domains only exist inside ``T.SimtVF``, which emits its
    own scopes below the launch nest.

``T.MixedKernel`` adds a second launch dimension, ``cthread``, which the Ascend
codegen reads back off the ``thread_extent`` AttrStmt; it is declared through
the pass's ``launch_dim_tags`` rather than hard-coded in the pass.
"""

import pytest
import tilelang
import tilelang.ascend.language as TA
from tilelang import tvm
from testing.ascend._ir import nodes as _collect


def _materialize(func, target: str = "ascend", **kwargs):
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target(target))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch(**kwargs)(mod)
    return mod[func.attrs["global_symbol"]]


def _ascend_materialize(func):
    """Exactly what tilelang/ascend/pipeline.py asks for."""
    return _materialize(
        func,
        "ascend",
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )


def _launch_placeholders(func):
    return [
        stmt
        for stmt in _collect(func, tvm.tirx.Bind)
        if isinstance(stmt.value, tvm.tirx.Call) and str(stmt.value.op.name) == "tl.launch_thread_idx"
    ]


def _thread_extents(func):
    """{thread_tag: extent} of every thread_extent AttrStmt in `func`."""
    extents = {}
    for attr in _collect(func, tvm.tirx.AttrStmt):
        if attr.attr_key == "thread_extent":
            extents[str(attr.node.thread_tag)] = int(attr.value)
    return extents


def test_ascend_kernel_has_no_threads():
    """The launch annotations are explicit keyword parameters, so an unsupported
    one is rejected by Python rather than by a runtime hardware probe."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'threads'"):

        @TA.prim_func
        def main(A: TA.Tensor((16,), "float32")):
            with TA.Kernel(1, threads=128):
                A[0] = 0


def test_ascend_kernel_rejects_cluster_dims():
    with pytest.raises(TypeError, match="unexpected keyword argument 'cluster_dims'"):

        @TA.prim_func
        def main(A: TA.Tensor((16,), "float32")):
            with TA.Kernel(1, cluster_dims=2):
                A[0] = 0


def test_ascend_kernel_accepts_prelude():
    @TA.prim_func
    def main(A: TA.Tensor((16,), "float32")):
        with TA.Kernel(1, prelude="// ascend"):
            A[0] = 0

    assert str(_collect(main, tvm.tirx.SBlock)[0].annotations["pragma_import_c"]) == "// ascend"


def test_ascend_kernel_rejects_multidimensional_grid():
    with pytest.raises(ValueError, match="1-D core grid"):

        @TA.prim_func
        def main(A: TA.Tensor((16,), "float32")):
            with TA.Kernel(2, 2):
                A[0] = 0


def _ascend_launch_func():
    @TA.prim_func
    def main(A: TA.Tensor((16,), "float32")):
        with TA.Kernel(2) as bx:
            A[bx] = 1.0

    return main


def test_ascend_materializes_grid_and_drops_thread_placeholders():
    materialized = _ascend_materialize(_ascend_launch_func())
    assert _thread_extents(materialized) == {"blockIdx.x": 2}
    assert _launch_placeholders(materialized) == []


def test_ascend_rejects_thread_index_use_outside_simtvf():
    @TA.prim_func
    def main(A: TA.Tensor((16,), "float32")):
        with TA.Kernel(2) as bx:
            A[bx + TA.get_thread_binding()] = 1.0

    with pytest.raises(Exception, match="no SIMT threads"):
        _ascend_materialize(main)


def test_ascend_rejects_cluster_dims_annotation_from_other_dialects():
    """A launch traced through another dialect must not smuggle an unsupported
    annotation past the Ascend pipeline."""
    import tilelang.cuda.language as TC

    @TC.prim_func
    def main(A: TC.Tensor((16,), "float32")):
        with TC.Kernel(2, threads=32, cluster_dims=2) as bx:
            A[bx] = 1.0

    with pytest.raises(Exception, match="cluster_dims"):
        _ascend_materialize(main)


def test_ascend_accepts_cthread_as_a_launch_dimension():
    """T.MixedKernel's sub-block id must survive as a thread_extent AttrStmt:
    the Ascend codegen reads it back off the cthread tag."""

    @TA.prim_func
    def main(A: TA.Tensor((16,), "float32")):
        with TA.MixedKernel(2, sids=2) as (bx, sid):
            A[bx + sid] = 1.0

    materialized = _ascend_materialize(main)
    assert _thread_extents(materialized) == {"blockIdx.x": 2, "cthread": 2}
    assert _launch_placeholders(materialized) == []


def test_cthread_is_opt_in_per_backend():
    """launch_dim_tags is a backend declaration: a pipeline that does not declare
    cthread leaves that loop for later passes instead of turning it into a
    launch thread_extent."""

    @TA.prim_func
    def main(A: TA.Tensor((16,), "float32")):
        with TA.MixedKernel(2, sids=2) as (bx, sid):
            A[bx + sid] = 1.0

    materialized = _materialize(main, "ascend", lower_grid_binding=True, lower_thread_binding=False, default_threads=None)
    extents = _thread_extents(materialized)
    assert "cthread" not in extents
    assert extents["blockIdx.x"] == 2


if __name__ == "__main__":
    tilelang.testing.main()
