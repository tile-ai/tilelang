"""Before/after IR tests for ``Schedule.copy_at``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin("handle", tirx.op.Op.get("tl.tileop.region"), load, access_mask, *extents)


def test_copy_at_replaces_copy_block_compute_nest_with_tile_copy():
    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), C: T.Buffer((16,), "float32")):
        B = T.sblock_alloc_buffer((16,), "float32")
        for i in range(16):
            with T.sblock("B"):
                v_i = T.axis.spatial(16, i)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i]
            with T.sblock("C"):
                v_i = T.axis.spatial(16, i)
                T.reads(B[v_i])
                T.writes(C[v_i])
                C[v_i] = B[v_i] + T.float32(1.0)

    @T.prim_func
    def after(A: T.Buffer((16,), "float32"), C: T.Buffer((16,), "float32")):
        T.func_attr({"global_symbol": "before"})
        B = T.sblock_alloc_buffer((16,), "float32")
        for i in range(16):
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tirx.op.Op.get("tl.tileop.copy"),
                    _region(A[i], 1, 1),
                    _region(B[i], 2, 1),
                )
            )
            with T.sblock("C"):
                v_i = T.axis.spatial(16, i)
                T.reads(B[v_i])
                T.writes(C[v_i])
                C[v_i] = B[v_i] + T.float32(1.0)

    sch = Schedule(before)
    block = sch.get_sblock("B")
    sch.copy_at(sch.get_loops(block)[0], block)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
