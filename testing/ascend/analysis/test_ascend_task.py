import pytest
import tilelang.ascend.language as T
import tilelang.testing

from tilelang import tvm
from tilelang.engine.lower import lower


TILE = 256
FLAG = 3


def test_task_rejects_cross_pipe_body():
    @T.prim_func
    def main(A: T.Tensor((TILE, TILE), "bfloat16")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(a_l1, a_l0)

    with pytest.raises(tvm.error.InternalError, match="exactly one Ascend hardware pipe"):
        lower(main, target="ascend")


def test_task_rejects_conflicting_core_affinity():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_ub = T.alloc_shared((TILE, TILE), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(B, b_ub)

    with pytest.raises(tvm.error.InternalError, match="both AIC and AIV HBM paths"):
        lower(main, target="ascend")


def test_task_rejects_mixed_copy_and_fill_pipes():
    # The copy is MTE2 and an out-of-VF fill is PIPE_S, so one task cannot hold
    # both: the fill runs on the scalar unit, not on the DMA path.
    @T.prim_func
    def main(A: T.Tensor((TILE,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((TILE,), "float32")
            with T.Task():
                T.copy(A, temp)
                T.fill(temp, 0)

    with pytest.raises(tvm.error.InternalError, match="exactly one Ascend hardware pipe"):
        lower(main, target="ascend")


def test_out_of_vf_fill_issues_on_scalar_pipe():
    # A fill outside a VF block is element-wise scalar work, so it must be
    # ordered against its consumer as a PIPE_S task.
    @T.prim_func
    def main(A: T.Tensor((TILE,), "float32"), n: T.int32):
        with T.Kernel(1):
            temp = T.alloc_shared((TILE,), "float32")
            T.fill(temp[0:n], 0)
            T.copy(temp, A)

    # tilelang.lower expects the caller to hold the target scope; the
    # vectorize planner consults Target.current().
    with tvm.target.Target("ascend"):
        source = lower(main, target="ascend").kernel_source
    assert "asc_sync_notify(PIPE_S, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_S, PIPE_MTE3," in source


def test_statically_shaped_out_of_vf_fill_is_still_scalar():
    @T.prim_func
    def main(A: T.Tensor((TILE,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((TILE,), "float32")
            T.fill(temp, 0)
            T.copy(temp, A)

    # tilelang.lower expects the caller to hold the target scope; the
    # vectorize planner consults Target.current().
    with tvm.target.Target("ascend"):
        source = lower(main, target="ascend").kernel_source
    assert "asc_sync_notify(PIPE_S, PIPE_MTE3," in source
    assert "asc_sync_wait(PIPE_S, PIPE_MTE3," in source


def test_task_recognizes_non_dma_copy_as_vector_pipe():
    @T.prim_func
    def main(A: T.Tensor((TILE,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((TILE,), "float32")
            destination = T.alloc_shared((TILE,), "float32")
            with T.Task():
                T.copy(A, source)
                T.copy(source, destination)

    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="exactly one Ascend hardware pipe"):
        lower(main, target="ascend")


def test_task_allows_standalone_fixpipe_dual_copy():
    tile = 128

    @T.prim_func
    def main():
        with T.Kernel(1):
            accum = T.alloc_l0c((tile, tile), "float32")
            temp = T.alloc_shared((tile // 2, tile), "float32")
            with T.Task():
                T.dual_copy(accum, temp)

    source = lower(main, target="ascend").kernel_source
    assert "asc_get_sub_block_id()" not in source


def test_task_allows_standalone_cross_core_sync():
    @T.prim_func
    def main():
        with T.Kernel(1), T.Task():
            T.ascend_sync_inter_wait("PIPE_V", FLAG)

    source = lower(main, target="ascend").kernel_source
    assert f"asc_sync_inter_wait(PIPE_V, {FLAG});" in source


def test_task_rejects_multiple_cross_core_syncs():
    @T.prim_func
    def main():
        with T.Kernel(1), T.Task():
            T.ascend_sync_inter_arrive("PIPE_V", FLAG)
            T.ascend_sync_inter_wait("PIPE_V", FLAG)

    with pytest.raises(tvm.error.InternalError, match="must be the task's only statement"):
        lower(main, target="ascend")


def test_task_does_not_expose_core_mask_argument():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        T.Task(core_mask=1)


if __name__ == "__main__":
    tilelang.testing.main()
