"""Buffer reuse must preserve the original synchronization access contracts."""

import pytest

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.engine.lower import lower


def _source(program, disable_reuse=False):
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        return lower(program, target="ascend").kernel_source


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_pointer_store_past_view_keeps_scalar_read_dependency(disable_reuse, strided):
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            if strided:
                view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            else:
                view = tvm.tirx.decl_buffer((96,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.float32(1), "float32"))
                out[0] = view[88 if strided else 96]

    source = _source(main, disable_reuse)
    assert "asc_sync_notify(PIPE_V, PIPE_S," in source
    assert "asc_sync_wait(PIPE_V, PIPE_S," in source


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("kind", ["dense_view", "strided", "padded_copy"])
@pytest.mark.parametrize("hint", ["region", "storage", "region_overrides_storage"])
def test_expanded_footprint_preserves_explicit_conflict(kind, disable_reuse, hint):
    @T.prim_func
    def main(A: T.Tensor((30,), "float32"), C: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            consumer = T.alloc_shared((64,), "float32")
            view = T.StridedTensor((128,), (2 if kind == "strided" else 1,), "float32", data=allocation.data, scope="shared.dyn")
            # Initialize independently of the producer's pipe so this does not
            # provide the ordering asserted below.
            consumer[0] = T.float32(2)
            if kind != "padded_copy":
                if hint == "region_overrides_storage":
                    T.assume_no_conflict(view, consumer, cross=False)
                if hint == "storage":
                    T.assume_conflict(view, consumer, cross=False)
                else:
                    T.assume_conflict(view[:64], consumer[:64], cross=False)
                with T.SimdVF():
                    T.simd.vsts(view[0], T.simd.vdup(T.float32(1), "float32"))
            else:
                if hint == "region_overrides_storage":
                    T.assume_no_conflict(allocation, consumer, cross=False)
                if hint == "storage":
                    T.assume_conflict(allocation, consumer, cross=False)
                else:
                    T.assume_conflict(allocation[:30], consumer[:64], cross=False)
                T.copy(A, allocation[:30], pad_value=0.0)
            T.copy(consumer, C)

    source = _source(main, disable_reuse)
    producer_pipe = "MTE2" if kind == "padded_copy" else "V"
    assert f"asc_sync_notify(PIPE_{producer_pipe}, PIPE_MTE3," in source
    assert f"asc_sync_wait(PIPE_{producer_pipe}, PIPE_MTE3," in source


@pytest.mark.parametrize("hint", ["none", "logical_region", "storage"])
def test_no_conflict_keeps_original_logical_region_matching(hint):
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            if hint == "logical_region":
                T.assume_no_conflict(view[64:128], view[88:89], cross=False)
            elif hint == "storage":
                # An explicit storage-wide assertion keeps its existing scope.
                T.assume_no_conflict(view, cross=False)
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.float32(1), "float32"))
                out[0] = view[88]

    source = _source(main)
    # Physical opacity only blocks reuse; it does not change hint matching.
    assert ("asc_sync_notify(PIPE_V, PIPE_S," in source) == (hint == "none")


def test_unknown_pointer_read_does_not_create_read_read_dependency():
    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            result = T.alloc_shared((64,), "float32")
            view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                view[88] = T.float32(1)
                with T.SimdVF():
                    T.simd.vsts(result[0], T.simd.vld(view[64]))
                out[0] = view[88]

    source = _source(main)
    assert "asc_sync_notify(PIPE_S, PIPE_V," in source
    assert "asc_sync_notify(PIPE_V, PIPE_S," not in source


@pytest.mark.parametrize("versions", [1, 2])
@pytest.mark.parametrize("cross", [False, True])
def test_padded_source_hint_survives_loop_scheduling(versions, cross):
    @T.prim_func
    def main(A: T.Tensor((4, 30), "float32"), C: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            producer = T.alloc_shared((32,), "float32")
            consumer = T.alloc_shared((64,), "float32")
            consumer[0] = T.float32(1)
            for i in T.Pipelined(4, num_stages=versions):
                T.assume_conflict(producer[:30], consumer[:64], cross=cross)
                T.copy(A[i, :], producer[:30], pad_value=0.0)
                T.copy(consumer, C[i, :])

    source = _source(main)
    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_MTE2, PIPE_MTE3," in source


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("control", ["condition", "loop_extent"])
def test_physical_pointer_write_orders_control_expression_reads(control, disable_reuse):
    @T.prim_func
    def main(out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "int32")
            view = tvm.tirx.decl_buffer((96,), "int32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.int32(1), "int32"))
                if control == "condition":
                    if view[96] > 0:
                        out[0] = 1
                else:
                    for _i in T.serial(view[96]):
                        out[0] = 1

    source = _source(main, disable_reuse)
    assert "asc_sync_notify(PIPE_V, PIPE_S," in source
    assert "asc_sync_wait(PIPE_V, PIPE_S," in source
