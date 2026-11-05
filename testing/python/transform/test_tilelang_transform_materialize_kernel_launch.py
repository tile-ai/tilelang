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
import tilelang
import tilelang.testing
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir
from tilelang.testing.ir import collect_nodes


def _collect(root, kind):
    return collect_nodes(root, kind)


def _kernel_regions(func):
    return [node for node in _collect(func, ir.RegionStmt) if node.op.name == "tl.launch_kernel"]


def _launch_thread_vars(func):
    # A native Kernel region defines the grid coordinates followed by the
    # three logical thread identities; it does not emit placeholder Calls.
    return [variable for region in _kernel_regions(func) for variable in region.body_params[len(region.args[1].fields) :]]


def _thread_extents(func):
    return {
        str(region.args[0].value): int(region.args[1]) for region in _collect(func, ir.RegionStmt) if region.op.name == "tirx.launch_thread"
    }


def _launch_hints(func):
    regions = _kernel_regions(func)
    if regions:
        assert len(regions) == 1
        region = regions[0]
        return tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[4], region.attrs.get("hint_schema", ["map"]))
    metadata = []
    for region in _collect(func, ir.RegionStmt):
        if region.op.name == "tl.attr":
            fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
            if fields["key"] == "tl.kernel":
                metadata.append(fields["value"])
    assert len(metadata) == 1
    return metadata[0]


def _materialize(func, target: str, **kwargs):
    mod = tvm_tirx.IRModule.from_expr(func)
    mod = tvm_tirx.tirx.transform.BindTarget(tvm_tirx.target.Target(target))(mod)
    mod = tl.transform.MaterializeKernelLaunch(**kwargs)(mod)
    return mod[func.attrs["global_symbol"]]


def _parallel_kernel(threads=None):
    @T.prim_func
    def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
        with T.Kernel(2, threads=threads) as bx:
            for i in T.Parallel(128):
                B[bx * 128 + i] = A[bx * 128 + i] + 1.0

    return main


def _thread_indexed_kernel():
    @T.prim_func
    def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
        with T.Kernel(2, threads=128) as bx:
            tx = T.get_thread_binding()
            B[bx * 128 + tx] = A[bx * 128 + tx] + 1.0

    return main


@pytest.mark.parametrize(
    ("arguments", "keywords", "expected"),
    [
        ((), {}, (True, True, [128, 1, 1], None, None)),
        ((False,), {}, (False, False, [128, 1, 1], None, None)),
        ((True, (32, 2)), {}, (True, True, [32, 2, 1], None, None)),
        ((False, None, ("cluster_dims",)), {}, (False, False, None, ["cluster_dims"], None)),
        (
            (False, None, ("cluster_dims",)),
            {"lower_grid_binding": True, "launch_dim_tags": ("cthread",)},
            (True, False, None, ["cluster_dims"], ["cthread"]),
        ),
        (
            (),
            {
                "lower_grid_binding": True,
                "lower_thread_binding": False,
                "default_threads": None,
                "unsupported_annotations": ["cluster_dims"],
                "launch_dim_tags": ["cthread"],
            },
            (True, False, None, ["cluster_dims"], ["cthread"]),
        ),
    ],
)
def test_materialize_launch_argument_compatibility(monkeypatch, arguments, keywords, expected):
    monkeypatch.setattr(tl.transform._ffi_api, "MaterializeKernelLaunch", lambda *ffi_arguments: ffi_arguments)
    assert tl.transform.MaterializeKernelLaunch(*arguments, **keywords) == expected


@pytest.mark.parametrize("parameter_name", ["lower_grid_binding", "launch_dim_tags"])
def test_materialize_launch_new_options_are_keyword_only(parameter_name):
    signature = inspect.signature(tl.transform.MaterializeKernelLaunch)
    assert signature.parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY


def test_materialize_launch_preserves_existing_grid_annotations():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        for bx in T.thread_binding(2, thread="blockIdx.x", annotations={"backend.grid_axis": 7, "backend.parallel_grid": True}):
            A[bx] = 1

    materialized = _materialize(main, "c", lower_thread_binding=False)
    loop = _collect(materialized, ir.For)[0]
    # Direct loop metadata is a lexical operand region until finalization.
    carriers = [region for region in _collect(materialized, ir.RegionStmt) if region.op.name == "tl.attr"]
    fields = [tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"]) for region in carriers]
    (annotations,) = [entry["value"] for entry in fields if entry["key"] == "tl.loop.annotations"]
    assert int(annotations["backend.grid_axis"]) == 7
    assert bool(annotations["backend.parallel_grid"])
    assert loop.kind == ir.ForKind.DEFAULT and int(loop.end - loop.begin) == 2


def test_materialize_launch_does_not_read_cpu_parallel_config():
    source = _parallel_kernel()
    with tvm_tirx.transform.PassContext(config={"tl.cpu_parallel": True}):
        enabled = _materialize(source, "c", lower_thread_binding=False)
    with tvm_tirx.transform.PassContext(config={"tl.cpu_parallel": False}):
        disabled = _materialize(source, "c", lower_thread_binding=False)
    # Native grid provenance is independent of the later CPU parallel policy.
    # Compare the complete results so a config-dependent change cannot hide.
    tvm_tirx.ir.assert_structural_equal(enabled, disabled)


def test_traced_launch_records_grid_and_thread_placeholders():
    func = _parallel_kernel()

    (region,) = _kernel_regions(func)
    assert list(region.attrs["grid_tags"]) == ["blockIdx.x"]
    assert [int(value) for value in region.args[0].fields] == [0]
    assert [int(value) for value in region.args[1].fields] == [2]
    assert [int(value) for value in region.args[2].fields] == [1]
    assert len(region.body_params) == 4
    assert region.body_params[0].ty.dtype == "int32"
    assert [var.name for var in _launch_thread_vars(func)] == ["tx", "ty", "tz"]

    # No threads= means no SIMT hint is recorded; the backend picks.
    assert not region.attrs["threads_was_specified"]
    assert not region.args[3].fields
    assert _thread_extents(func) == {}


def test_traced_launch_records_requested_threads_as_annotation():
    func = _parallel_kernel(threads=(64, 2))
    (region,) = _kernel_regions(func)
    assert region.attrs["threads_was_specified"]
    threads = region.args[3].fields
    assert [int(x) for x in threads] == [64, 2, 1]
    # Still only placeholders: the frontend does not bind threadIdx itself.
    assert _thread_extents(func) == {}


def test_kernel_launch_annotations_are_recorded_on_the_root_block():
    @T.prim_func
    def main(A: T.Tensor((16,), T.int32)):
        with T.Kernel(1, threads=64, prelude="// hi", cluster_dims=2):
            A[0] = 0

    annotations = _launch_hints(main)
    assert [int(x) for x in _kernel_regions(main)[0].args[3].fields] == [64, 1, 1]
    assert [int(x) for x in annotations["cluster_dims"]] == [2, 1, 1]
    assert str(annotations["pragma_import_c"]) == "// hi"


def test_kernel_rejects_unknown_launch_annotation():
    """A dialect's Kernel declares its launch annotations as explicit keyword
    parameters, so a misspelled or foreign key fails at trace time."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'thread'"):

        @T.prim_func
        def typo(A: T.Tensor((16,), T.int32)):
            with T.Kernel(1, thread=128):
                A[0] = 0

    with pytest.raises(TypeError, match="unexpected keyword argument 'core_type'"):

        @T.prim_func
        def foreign(A: T.Tensor((16,), T.int32)):
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
        # cpu_num_threads is a CPU-only launch annotation; the default facade
        # (the CUDA dialect) rejects it at trace time.
        "tilelang.cpu.language": {"prelude", "cpu_num_threads"},
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
        def main(A: Tcpu.Tensor((16,), Tcpu.int32)):
            with Tcpu.Kernel(1, threads=128):
                A[0] = 0

    @Tcpu.prim_func
    def ok(A: Tcpu.Tensor((16,), Tcpu.int32)):
        with Tcpu.Kernel(1, prelude="// cpu"):
            A[0] = 0

    assert str(_launch_hints(ok)["pragma_import_c"]) == "// cpu"


def test_simt_binds_requested_threads():
    func = _materialize(_parallel_kernel(threads=64), "cuda")
    assert _thread_extents(func) == {"blockIdx.x": 2, "threadIdx.x": 64, "threadIdx.y": 1, "threadIdx.z": 1}
    assert _launch_thread_vars(func) == []


def test_simt_uses_backend_default_when_threads_omitted():
    func = _materialize(_parallel_kernel(), "cuda", default_threads=256)
    assert _thread_extents(func)["threadIdx.x"] == 256

    func = _materialize(_parallel_kernel(), "cuda")
    assert _thread_extents(func)["threadIdx.x"] == tl.transform.DEFAULT_SIMT_THREADS


def test_simt_requires_threads_when_backend_has_no_default():
    with pytest.raises(Exception, match="needs threads= or a backend default"):
        _materialize(_parallel_kernel(), "cuda", default_threads=None)


def test_simt_preserves_thread_var_identity():
    """The Var handed out by T.get_thread_binding() at trace time must be the
    Var bound by the threadIdx.x thread_extent after materialization."""
    func = _thread_indexed_kernel()
    (thread_var,) = [p for p in _launch_thread_vars(func) if p.name == "tx"]

    lowered = _materialize(func, "cuda")
    (launch,) = [a for a in _collect(lowered, ir.RegionStmt) if a.op.name == "tirx.launch_thread" and a.args[0].value == "threadIdx.x"]
    assert launch.body_params[0].same_as(thread_var)
    body_vars = [v for v in _collect(lowered, ir.Var) if v.name == "tx"]
    assert body_vars and all(v.same_as(thread_var) for v in body_vars)


def test_non_simt_drops_thread_placeholders():
    source = _parallel_kernel(threads=128)
    grid_var = _kernel_regions(source)[0].body_params[0]
    func = _materialize(source, "c", lower_thread_binding=False)
    assert _thread_extents(func) == {}
    assert _launch_thread_vars(func) == []
    grid = [f for f in _collect(func, ir.For) if f.var.same_as(grid_var)]
    assert len(grid) == 1 and grid[0].kind == ir.ForKind.DEFAULT and int(grid[0].end - grid[0].begin) == 2
    assert not any(v.name in ("tx", "ty", "tz") for v in _collect(func, ir.Var))


def test_non_simt_rejects_thread_index_use():
    with pytest.raises(Exception, match="references thread index `tx`"):
        _materialize(_thread_indexed_kernel(), "c", lower_thread_binding=False)


def test_get_thread_extent_requires_threads_at_trace_time():
    with pytest.raises(ValueError, match="not known at trace time"):

        @T.prim_func
        def main(A: T.Tensor((16,), T.int32)):
            with T.Kernel(1):
                A[0] = T.get_thread_extent()


def test_get_thread_extent_with_threads_at_trace_time():
    @T.prim_func
    def main(A: T.Tensor((16,), T.int32)):
        with T.Kernel(1, threads=(32, 4)):
            A[0] = T.get_thread_extent(0) * T.get_thread_extent(1)

    (store,) = _collect(main, ir.TensorStore)
    assert int(store.value) == 128


class _TraceFailure(Exception):
    pass


def test_failed_trace_unwinds_launch_frames():
    """An exception inside T.Kernel must leave no stale launch frame behind,
    otherwise the next trace sees the previous kernel's KernelLaunchFrame."""
    with pytest.raises(_TraceFailure):

        @T.prim_func
        def failing(A: T.Tensor((16,), T.int32)):
            with T.Kernel(1, threads=128):
                raise _TraceFailure()

    assert T.KernelLaunchFrame.Current() is None

    @tilelang.jit
    def failing_jit(A):
        A: T.Tensor[[16], T.int32]
        with T.Kernel(1, threads=128):
            raise _TraceFailure()

    import torch

    with pytest.raises(_TraceFailure):
        failing_jit.get_tir(torch.zeros(16, dtype=torch.int32))
    assert T.KernelLaunchFrame.Current() is None


def _cluster_kernel():
    @T.prim_func
    def main(A: T.Tensor((16,), T.int32)):
        with T.ClusterKernel(8, 4, threads=128, cluster_dims=2) as (bx, by):
            A[0] = 0

    return main


def test_cluster_dims_is_a_launch_annotation():
    dims = _launch_hints(_cluster_kernel())["cluster_dims"]
    assert [int(d) for d in dims] == [2, 1, 1]


def test_cluster_id_is_program_space_arithmetic():
    """Cluster identity is derived from the program index and cluster_dims at
    trace time, so it needs no target-specific intrinsic."""
    captured = {}

    @T.prim_func
    def main(A: T.Tensor((16,), T.int32)):
        with T.ClusterKernel(8, 4, threads=128, cluster_dims=2) as (bx, by):
            captured["bx"], captured["by"] = bx, by
            captured["ids"] = T.get_cluster_ids()
            captured["dims"] = T.get_cluster_dims()
            captured["size"] = T.get_cluster_size()
            captured["extents"] = T.get_cluster_extents()
            A[0] = T.get_cluster_id(0)

    cx, cy = captured["ids"]
    assert isinstance(cx, ir.prim.FloorDiv) and cx.a.same_as(captured["bx"]) and int(cx.b) == 2
    # A unit cluster axis is the program index itself.
    assert cy.same_as(captured["by"])
    assert captured["dims"] == [2, 1, 1]
    assert captured["size"] == 2
    assert captured["extents"] == [4, 4, 1]
    (store,) = _collect(main, ir.TensorStore)
    assert isinstance(store.value, ir.prim.FloorDiv)


def test_cluster_id_without_clusters_is_the_program_index():
    captured = {}

    @T.prim_func
    def main(A: T.Tensor((16,), T.int32)):
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
    assert "cluster_dims" in _launch_hints(func)

    with pytest.raises(Exception, match="`cluster_dims` is not supported on target `c`"):
        _materialize(_cluster_kernel(), "c", lower_thread_binding=False, unsupported_annotations=["cluster_dims"])

    # A launch without the annotation is unaffected by the rejection list.
    _materialize(_parallel_kernel(), "c", lower_thread_binding=False, unsupported_annotations=["cluster_dims"])


if __name__ == "__main__":
    tilelang.testing.main()
