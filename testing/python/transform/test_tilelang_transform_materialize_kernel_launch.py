"""Tests for the target-neutral T.Kernel encoding and MaterializeKernelLaunch.

T.Kernel is traced before the Target is known, so it only records the grid
loops, thread-index placeholders and launch annotations; the backend pipeline
decides what the thread placeholders mean when it runs MaterializeKernelLaunch.
"""

import importlib
import inspect

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _collect(root, kind):
    found = []

    def _visit(node):
        if isinstance(node, kind):
            found.append(node)

    post_order_visit(root.body if hasattr(root, "body") else root, _visit)
    return found


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


def _root_block(func):
    blocks = [b for b in _collect(func, tvm.tirx.SBlock) if b.name_hint == "tilelang_root"]
    assert len(blocks) == 1
    return blocks[0]


def _materialize(func, target: str, **kwargs):
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target(target))(mod)
    mod = tl.transform.MaterializeKernelLaunch(**kwargs)(mod)
    return mod[func.attrs["global_symbol"]]


def _parallel_kernel(threads=None):
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(2, threads=threads) as bx:
            for i in T.Parallel(128):
                B[bx * 128 + i] = A[bx * 128 + i] + 1.0

    return main


def _thread_indexed_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(2, threads=128) as bx:
            tx = T.get_thread_binding()
            B[bx * 128 + tx] = A[bx * 128 + tx] + 1.0

    return main


def test_traced_launch_records_grid_and_thread_placeholders():
    func = _parallel_kernel()

    grid = [f for f in _collect(func, tvm.tirx.For) if f.kind == tvm.tirx.ForKind.THREAD_BINDING]
    assert [str(f.thread_binding.thread_tag) for f in grid] == ["blockIdx.x"]

    placeholders = _launch_placeholders(func)
    assert [p.var.name for p in placeholders] == ["tx", "ty", "tz"]
    assert [int(p.value.args[0]) for p in placeholders] == [0, 1, 2]

    # No threads= means no SIMT hint is recorded; the backend picks.
    assert "tl.launch_threads" not in _root_block(func).annotations
    assert _thread_extents(func) == {}


def test_traced_launch_records_requested_threads_as_annotation():
    func = _parallel_kernel(threads=(64, 2))
    threads = _root_block(func).annotations["tl.launch_threads"]
    assert [int(x) for x in threads] == [64, 2, 1]
    # Still only placeholders: the frontend does not bind threadIdx itself.
    assert _thread_extents(func) == {}


def test_kernel_launch_annotations_are_recorded_on_the_root_block():
    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.Kernel(1, threads=64, prelude="// hi", cluster_dims=2):
            A[0] = 0

    annotations = _root_block(main).annotations
    assert [int(x) for x in annotations["tl.launch_threads"]] == [64, 1, 1]
    assert [int(x) for x in annotations["cluster_dims"]] == [2, 1, 1]
    assert str(annotations["pragma_import_c"]) == "// hi"


def test_kernel_rejects_unknown_launch_annotation():
    """A dialect's Kernel declares its launch annotations as explicit keyword
    parameters, so a misspelled or foreign key fails at trace time."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'thread'"):

        @T.prim_func
        def typo(A: T.Tensor((16,), "int32")):
            with T.Kernel(1, thread=128):
                A[0] = 0

    with pytest.raises(TypeError, match="unexpected keyword argument 'core_type'"):

        @T.prim_func
        def foreign(A: T.Tensor((16,), "int32")):
            with T.Kernel(1, core_type="aiv"):
                A[0] = 0


def _launch_annotations(kernel) -> set[str]:
    return {name for name, p in inspect.signature(kernel).parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}


def test_each_dialect_declares_its_own_launch_annotations():
    expected = {
        "tilelang.language.common": set(),
        "tilelang.cuda.language": {"threads", "prelude", "cluster_dims"},
        "tilelang.rocm.language": {"threads", "prelude"},
        "tilelang.metal.language": {"threads", "prelude"},
        "tilelang.webgpu.language": {"threads"},
        "tilelang.cpu.language": {"prelude"},
    }
    for module, keys in expected.items():
        dialect = importlib.import_module(module)
        assert _launch_annotations(dialect.Kernel) == keys, module
        assert dialect.Kernel.__module__.startswith(module.removesuffix(".common")), module
    # The default facade is the CUDA dialect.
    assert T.Kernel is importlib.import_module("tilelang.cuda.language").Kernel


def test_cpu_dialect_kernel_has_no_threads():
    from tilelang.cpu import language as Tcpu

    with pytest.raises(TypeError, match="unexpected keyword argument 'threads'"):

        @Tcpu.prim_func
        def main(A: Tcpu.Tensor((16,), "int32")):
            with Tcpu.Kernel(1, threads=128):
                A[0] = 0

    @Tcpu.prim_func
    def ok(A: Tcpu.Tensor((16,), "int32")):
        with Tcpu.Kernel(1, prelude="// cpu"):
            A[0] = 0

    assert str(_root_block(ok).annotations["pragma_import_c"]) == "// cpu"


def test_simt_binds_requested_threads():
    func = _materialize(_parallel_kernel(threads=64), "cuda")
    assert _thread_extents(func) == {"blockIdx.x": 2, "threadIdx.x": 64, "threadIdx.y": 1, "threadIdx.z": 1}
    assert _launch_placeholders(func) == []


def test_simt_uses_backend_default_when_threads_omitted():
    func = _materialize(_parallel_kernel(), "cuda", default_threads=256)
    assert _thread_extents(func)["threadIdx.x"] == 256

    func = _materialize(_parallel_kernel(), "cuda")
    assert _thread_extents(func)["threadIdx.x"] == tl.transform.DEFAULT_SIMT_THREADS


def test_simt_requires_threads_when_backend_has_no_default():
    with pytest.raises(Exception, match="did not specify threads="):
        _materialize(_parallel_kernel(), "cuda", default_threads=None)


def test_simt_preserves_thread_var_identity():
    """The Var handed out by T.get_thread_binding() at trace time must be the
    Var bound by the threadIdx.x thread_extent after materialization."""
    func = _thread_indexed_kernel()
    (placeholder,) = [p for p in _launch_placeholders(func) if p.var.name == "tx"]

    lowered = _materialize(func, "cuda")
    (attr,) = [a for a in _collect(lowered, tvm.tirx.AttrStmt) if str(a.node.thread_tag) == "threadIdx.x"]
    assert attr.node.var.same_as(placeholder.var)
    body_vars = [v for v in _collect(lowered, tvm.tirx.Var) if v.name == "tx"]
    assert body_vars and all(v.same_as(placeholder.var) for v in body_vars)


def test_non_simt_drops_thread_placeholders():
    func = _materialize(_parallel_kernel(threads=128), "c", lower_thread_binding=False)
    assert _thread_extents(func) == {}
    assert _launch_placeholders(func) == []
    grid = [f for f in _collect(func, tvm.tirx.For) if f.loop_var.name == "bx"]
    assert len(grid) == 1 and grid[0].kind == tvm.tirx.ForKind.SERIAL and int(grid[0].extent) == 2
    assert not any(v.name in ("tx", "ty", "tz") for v in _collect(func, tvm.tirx.Var))


def test_non_simt_rejects_thread_index_use():
    with pytest.raises(Exception, match="references thread index `tx`"):
        _materialize(_thread_indexed_kernel(), "c", lower_thread_binding=False)


def test_get_thread_extent_requires_threads_at_trace_time():
    with pytest.raises(ValueError, match="not known at trace time"):

        @T.prim_func
        def main(A: T.Tensor((16,), "int32")):
            with T.Kernel(1):
                A[0] = T.get_thread_extent()


def test_get_thread_extent_with_threads_at_trace_time():
    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.Kernel(1, threads=(32, 4)):
            A[0] = T.get_thread_extent(0) * T.get_thread_extent(1)

    (store,) = _collect(main, tvm.tirx.BufferStore)
    assert int(store.value) == 128


def _cluster_kernel():
    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.ClusterKernel(8, 4, threads=128, cluster_dims=2) as (bx, by):
            A[0] = 0

    return main


def test_cluster_dims_is_a_launch_annotation():
    dims = _root_block(_cluster_kernel()).annotations["cluster_dims"]
    assert [int(d) for d in dims] == [2, 1, 1]


def test_cluster_id_is_program_space_arithmetic():
    """Cluster identity is derived from the program index and cluster_dims at
    trace time, so it needs no target-specific intrinsic."""
    captured = {}

    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.ClusterKernel(8, 4, threads=128, cluster_dims=2) as (bx, by):
            captured["bx"], captured["by"] = bx, by
            captured["ids"] = T.get_cluster_ids()
            captured["dims"] = T.get_cluster_dims()
            captured["size"] = T.get_cluster_size()
            captured["extents"] = T.get_cluster_extents()
            A[0] = T.get_cluster_id(0)

    cx, cy = captured["ids"]
    assert isinstance(cx, tvm.tirx.FloorDiv) and cx.a.same_as(captured["bx"]) and int(cx.b) == 2
    # A unit cluster axis is the program index itself.
    assert cy.same_as(captured["by"])
    assert captured["dims"] == [2, 1, 1]
    assert captured["size"] == 2
    assert captured["extents"] == [4, 4, 1]
    (store,) = _collect(main, tvm.tirx.BufferStore)
    assert isinstance(store.value, tvm.tirx.FloorDiv)


def test_cluster_id_without_clusters_is_the_program_index():
    captured = {}

    @T.prim_func
    def main(A: T.Tensor((16,), "int32")):
        with T.Kernel(8) as bx:
            captured["bx"] = bx
            captured["id"] = T.get_cluster_id()
            captured["dims"] = T.get_cluster_dims()
            captured["extents"] = T.get_cluster_extents()
            A[0] = 0

    assert captured["id"].same_as(captured["bx"])
    assert captured["dims"] == [1, 1, 1]
    # Axes beyond the launched grid have a single cluster.
    assert captured["extents"] == [8, 1, 1]


def test_cluster_dims_accepted_by_default_and_rejected_when_unsupported():
    func = _materialize(_cluster_kernel(), "cuda")
    assert "cluster_dims" in _root_block(func).annotations

    with pytest.raises(Exception, match="`cluster_dims` is not supported on target `c`"):
        _materialize(_cluster_kernel(), "c", lower_thread_binding=False, unsupported_annotations=["cluster_dims"])

    # A launch without the annotation is unaffected by the rejection list.
    _materialize(_parallel_kernel(), "c", lower_thread_binding=False, unsupported_annotations=["cluster_dims"])


if __name__ == "__main__":
    tilelang.testing.main()
