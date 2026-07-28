"""Before/after IR tests for ``Schedule.cache_read_at``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.region"),
        load,
        access_mask,
        *extents,
    )


def test_cache_read_at_allocates_cache_copies_tile_and_rewrites_reads():
    @T.prim_func
    def before(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        for i_0, i_1 in T.grid(4, 4):
            with T.sblock("B"):
                v_i = T.axis.spatial(16, i_0 * 4 + i_1)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] + T.float32(1.0)

    @T.prim_func(check_well_formed=False)
    def after(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        T.func_attr({"global_symbol": "before"})
        for i_0 in range(4):
            with T.sblock(""):
                T.reads()
                T.writes()
                A_local_fragment = T.sblock_alloc_buffer((4,), scope="local.fragment")
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.copy"),
                        _region(A[i_0 * 4], 1, 4),
                        _region(A_local_fragment[0], 2, 4),
                    )
                )
                for i_1 in range(4):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(16, i_0 * 4 + i_1)
                        T.reads(A_local_fragment[v_i - i_0 * 4])
                        T.writes(B[v_i])
                        B[v_i] = A_local_fragment[v_i - i_0 * 4] + T.float32(1.0)

    sch = Schedule(before)
    block = sch.get_sblock("B")
    outer = sch.get_loops(block)[0]
    sch.cache_read_at(outer, block, 0, "local.fragment")

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


def test_cache_read_at_supports_shared_dyn_cache_tiles():
    @T.prim_func
    def before(
        A: T.Buffer((8, 8), "float32"),
        B: T.Buffer((8, 8), "float32"),
    ):
        for i_0, j_0, i_1, j_1 in T.grid(2, 2, 4, 4):
            with T.sblock("B"):
                v_i = T.axis.spatial(8, i_0 * 4 + i_1)
                v_j = T.axis.spatial(8, j_0 * 4 + j_1)
                T.reads(A[v_i, v_j])
                T.writes(B[v_i, v_j])
                B[v_i, v_j] = A[v_i, v_j] + T.float32(1.0)

    @T.prim_func(check_well_formed=False)
    def after(
        A: T.Buffer((8, 8), "float32"),
        B: T.Buffer((8, 8), "float32"),
    ):
        T.func_attr({"global_symbol": "before"})
        for i_0, j_0 in T.grid(2, 2):
            with T.sblock(""):
                T.reads()
                T.writes()
                A_shared_dyn = T.sblock_alloc_buffer((4, 4), scope="shared.dyn")
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.copy"),
                        _region(A[i_0 * 4, j_0 * 4], 1, 4, 4),
                        _region(A_shared_dyn[0, 0], 2, 4, 4),
                    )
                )
                for i_1, j_1 in T.grid(4, 4):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(8, i_0 * 4 + i_1)
                        v_j = T.axis.spatial(8, j_0 * 4 + j_1)
                        T.reads(A_shared_dyn[v_i - i_0 * 4, v_j - j_0 * 4])
                        T.writes(B[v_i, v_j])
                        B[v_i, v_j] = A_shared_dyn[v_i - i_0 * 4, v_j - j_0 * 4] + T.float32(1.0)

    sch = Schedule(before)
    block = sch.get_sblock("B")
    j_outer = sch.get_loops(block)[1]
    sch.cache_read_at(j_outer, block, 0, "shared.dyn")

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
