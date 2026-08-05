"""Before/after IR tests for ``Schedule.reduce_at``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin("handle", tirx.op.Op.get("tl.tileop.region"), load, access_mask, *extents)


def test_reduce_at_replaces_reduction_loop_with_tile_reduce():
    @T.prim_func
    def before(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
        for i, k in T.grid(4, 8):
            with T.sblock("B"):
                v_i, v_k = T.axis.remap("SR", [i, k])
                T.reads(A[v_i, v_k])
                T.writes(B[v_i])
                with T.init():
                    B[v_i] = T.float32(0.0)
                B[v_i] = B[v_i] + A[v_i, v_k]

    @T.prim_func
    def after(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "before"})
        with T.sblock("root"):
            T.reads()
            T.writes()
            for i in range(4):
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.reduce"),
                        _region(A[i, 0], 1, 1, 8),
                        _region(B[i], 2, 1),
                        "sum",
                        1,
                        T.bool(True),
                    )
                )

    sch = Schedule(before)
    block = sch.get_sblock("B")
    spatial_loop, _ = sch.get_loops(block)
    sch.reduce_at(spatial_loop, block, 0, 0, "sum", 1)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
