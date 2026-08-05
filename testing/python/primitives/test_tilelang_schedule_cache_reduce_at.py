"""Before/after IR tests for ``Schedule.cache_reduce_at``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin("handle", tirx.op.Op.get("tl.tileop.region"), load, access_mask, *extents)


def test_cache_reduce_at_allocates_initializes_and_writes_back_accumulator():
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

    @T.prim_func(check_well_formed=False)
    def after(A: T.Buffer((4, 8), "float32"), B: T.Buffer((4,), "float32")):
        T.func_attr({"global_symbol": "before"})
        for i in range(4):
            with T.sblock(""):
                T.reads()
                T.writes()
                B_local_fragment = T.sblock_alloc_buffer((1,), scope="local.fragment")
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.fill"),
                        _region(B_local_fragment[0], 2, 1),
                        T.float32(0.0),
                    )
                )
                for k in range(8):
                    with T.sblock("B"):
                        v_i, v_k = T.axis.remap("SR", [i, k])
                        T.reads(A[v_i, v_k])
                        T.writes(B_local_fragment[v_i - i])
                        with T.init():
                            B_local_fragment[v_i - i] = T.float32(0.0)
                        B_local_fragment[v_i - i] = B_local_fragment[v_i - i] + A[v_i, v_k]
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.copy"),
                        _region(B_local_fragment[0], 1, 1),
                        _region(B[i], 2, 1),
                    )
                )

    sch = Schedule(before)
    block = sch.get_sblock("B")
    spatial_loop, _ = sch.get_loops(block)
    sch.cache_reduce_at(spatial_loop, block, 0, "local.fragment", 0.0)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
