"""Core region, scalar and storage contracts of AutoSimtVF."""

import pytest
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _prepare(func):
    mod = tirx.transform.BindTarget(tvm.target.Target("ascend"))(tvm.IRModule.from_expr(func))
    return tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)


def _apply(func):
    return transform.AutoSimtVF()(_prepare(func))["main"]


def _vfs(func):
    return [block for block in nodes(func, tirx.SBlock) if block.name_hint == "SIMT_VF"]


def _copies(func):
    return [call for call in nodes(func, tirx.Call) if getattr(call.op, "name", "") in ("tl.tileop.copy", "tl.tileop.ascend_copy")]


def _lower(func):
    with tvm.target.Target("ascend") as target:
        assert tilelang.lower(func, target=target, enable_device_compile=False).kernel_source.strip()


@pytest.mark.parametrize("count", [0, 3])
def test_serial_copy_reduce_forms_one_region(count):
    @T.prim_func
    def main(A: T.Tensor((4, 8), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1):
            acc = T.alloc_fragment((4, 8), "float32")
            reduced = T.alloc_fragment((4,), "float32")
            T.copy(A, acc)
            for _ in T.serial(count):
                for i, j in T.Parallel(4, 8):
                    acc[i, j] += A[i, j]
            T.reduce_sum(acc, reduced, dim=1)
            T.copy(reduced, B)

    original = tvm.ir.save_json(main)
    result = _apply(main)
    (vf,) = _vfs(result)
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(vf, tirx.For))
    assert any(getattr(call.op, "name", "") == "tl.tileop.reduce" for call in nodes(vf, tirx.Call))
    assert all(buffer.scope() == "local.fragment" for buffer in vf.alloc_buffers)
    threads = [
        int(attr.value) for attr in nodes(vf, tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == "threadIdx.x"
    ]
    assert threads == [512]
    assert tvm.ir.save_json(main) == original
    repeated = transform.AutoSimtVF()(tvm.IRModule.from_expr(result))["main"]
    tvm.ir.assert_structural_equal(result, repeated)


def test_scalar_and_mte_stay_outside_region():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((8,), "float32")
            scale = T.alloc_var("float32")
            T.copy(A, shared)
            scale = 0
            for k in T.serial(3):
                scale += A[k]
            for i in T.Parallel(8):
                B[i] = shared[i] * scale

    with tvm.transform.PassContext(config={"tl.ascend_auto_simt_report_scalar": True}):
        result = _apply(main)
    (vf,) = _vfs(result)
    assert not _copies(vf)
    assert not any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(vf, tirx.For))
    assert not any(store.buffer.name == "scale" for store in nodes(vf, tirx.BufferStore))
    assert all(str(row["strategy"]) == "split" for row in result.attrs["tl.auto_simt_scalar_decisions"])


def test_conditional_cross_region_state_rewrites_views_and_reloads():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32"), flag: T.int32):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            alias = T.view(frag)
            shared = T.alloc_shared((8,), "float32")
            T.fill(frag, 1)
            T.copy(A, shared)
            if flag > 0:
                for i in T.Parallel(8):
                    frag[i] = alias[i] + shared[i]
            T.copy(A, shared)
            for i in T.Parallel(8):
                B[i] = frag[i]

    result = _apply(main)
    assert len(_vfs(result)) == 3
    copies = _copies(_vfs(result)[1])
    assert any("simtvf_state" in str(call.args[0].args[0].buffer.name) for call in copies)
    allocations = [buffer for block in nodes(result, tirx.SBlock) for buffer in block.alloc_buffers]
    aliases = [load.buffer for load in nodes(result, tirx.BufferLoad) if load.buffer.name == "alias"]
    assert aliases
    assert all(any(alias.data.same_as(buffer.data) for buffer in allocations) for alias in aliases)
    _lower(main)


def test_logical_store_guard_keeps_private_fragment_updates():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            source = T.alloc_fragment((8,), "float32")
            acc = T.alloc_fragment((8,), "float32")
            T.copy(A, source)
            T.clear(acc)
            for i in T.Parallel(8):
                acc[i] += source[i]
                B[i] += source[i]

    result = _apply(main)
    guards = [attr for attr in nodes(result, tirx.AttrStmt) if attr.attr_key == "tl.parallel_multiplicity"]
    assert len(guards) == 1
    assert isinstance(guards[0].body, tirx.BufferStore)
    assert guards[0].body.buffer.name == "B"
    assert any(store.buffer.name == "acc" for store in nodes(_vfs(result)[0], tirx.BufferStore))


def test_explicit_vf_is_opaque():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            with T.SimtVF(threads=64):
                for i in T.Parallel(8):
                    B[i] = A[i] + 1
            for i in T.Parallel(8):
                B[i] += 2

    prepared = _prepare(main)
    original = _vfs(prepared["main"])[0]
    result = transform.AutoSimtVF()(prepared)["main"]
    assert len(_vfs(result)) == 2
    tvm.ir.assert_structural_equal(_vfs(result)[0], original)


def test_task_scope_is_preserved_around_generated_region():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1), T.Task():
            for i in T.Parallel(8):
                B[i] = A[i] + 1

    result = _apply(main)
    tasks = [attr for attr in nodes(result, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]
    assert len(tasks) == 1
    assert len(_vfs(tasks[0])) == 1


@pytest.mark.parametrize("rows,region_count", [(64, 1), (65, 0)])
def test_shared_fill_requires_divisible_thread_geometry(rows, region_count):
    @T.prim_func
    def main(B: T.Tensor((rows, 128), "float32")):
        with T.Kernel(1):
            shared = T.alloc_shared((rows, 128), "float32")
            T.fill(shared, 0)
            T.copy(shared, B)

    assert len(_vfs(_apply(main))) == region_count
    _lower(main)


def test_partial_fragment_copy_is_rejected():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((8,), "float32")
            T.copy(A[:4], frag[:4])
            for i in T.Parallel(8):
                B[i] = frag[i]

    with pytest.raises(tvm.error.InternalError, match="partial fragment copy"):
        _apply(main)


def test_subcore_dispatch_stays_outside_vf():
    @T.prim_func
    def main(B: T.Tensor((2, 8), "float32")):
        with T.MixedKernel(1) as (_, sid):
            shared = T.alloc_shared((8,), "float32")
            for i in T.Parallel(8):
                B[sid, i] = 1
            if sid == 0:
                T.fill(shared, 0)

    result = _apply(main)
    assert len(_vfs(result)) == 2
    assert not any(nodes(vf, tirx.IfThenElse) for vf in _vfs(result))
