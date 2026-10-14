import tilelang.language as T
import tilelang.testing


def test_persistent_tile_scheduler_scalar_state_access():
    @T.prim_func
    def persistent(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, name="sched")
            sched.init(block_id)
            while sched.valid():
                A[sched.m_idx] = sched.n_idx + sched.current_iter
                sched.next_tile()

    script = persistent.script()
    assert "sched_m_idx" in script
    assert "sched_n_idx" in script
    assert "sched_current_iter" in script
    assert "while" in script


def test_persistent_tile_scheduler_stateless_coord_access():
    @T.prim_func
    def stateless(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, stateful=False)
            bx, by = sched.coord(block_id)
            A[bx] = by

    script = stateless.script()
    assert "alloc_var" not in script
    assert "A[bx_1] = by" in script


if __name__ == "__main__":
    tilelang.testing.main()
