"""Before/after IR tests for ``Schedule.fill_at``."""

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin("handle", tirx.op.Op.get("tl.tileop.region"), load, access_mask, *extents)


def test_fill_at_inserts_fill_before_target_loop_body():
    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        for i_0, i_1 in T.grid(4, 4):
            with T.sblock("B"):
                v_i = T.axis.spatial(16, i_0 * 4 + i_1)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] + T.float32(1.0)

    @T.prim_func
    def after(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        T.func_attr({"global_symbol": "before"})
        for i_0 in range(4):
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tirx.op.Op.get("tl.tileop.fill"),
                    _region(B[i_0 * 4], 2, 4),
                    T.float32(3.0),
                )
            )
            for i_1 in range(4):
                with T.sblock("B"):
                    v_i = T.axis.spatial(16, i_0 * 4 + i_1)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = A[v_i] + T.float32(1.0)

    sch = Schedule(before)
    block = sch.get_sblock("B")
    outer = sch.get_loops(block)[0]
    sch.fill_at(outer, block, 0, 3.0)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


def test_fill_at_rejects_loop_from_sibling_block():
    @T.prim_func
    def before(B: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        for i in range(4):
            with T.sblock("B"):
                v_i = T.axis.spatial(4, i)
                T.writes(B[v_i])
                B[v_i] = T.float32(1.0)
        for j in range(4):
            with T.sblock("C"):
                v_j = T.axis.spatial(4, j)
                T.writes(C[v_j])
                C[v_j] = T.float32(2.0)

    sch = Schedule(before)
    block_b = sch.get_sblock("B")
    sibling_loop = sch.get_loops(sch.get_sblock("C"))[0]

    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fill_at(sibling_loop, block_b, 0)


if __name__ == "__main__":
    tilelang.testing.main()
