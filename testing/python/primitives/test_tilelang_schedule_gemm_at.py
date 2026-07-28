"""Before/after IR tests for ``Schedule.gemm_at``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm import tirx
from tvm.script import tirx as T


def _region(load, access_mask, *extents):
    return T.call_intrin("handle", tirx.op.Op.get("tl.tileop.region"), load, access_mask, *extents)


def test_gemm_at_replaces_cached_matmul_compute_nest_with_tile_gemm():
    @T.prim_func
    def before(
        A: T.Buffer((8, 8), "float16"),
        B: T.Buffer((8, 8), "float16"),
        C: T.Buffer((8, 8), "float32"),
    ):
        for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(2, 2, 2, 4, 4, 4):
            with T.sblock("C"):
                v_i = T.axis.spatial(8, i_0 * 4 + i_1)
                v_j = T.axis.spatial(8, j_0 * 4 + j_1)
                v_k = T.axis.reduce(8, k_0 * 4 + k_1)
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(C[v_i, v_j])
                with T.init():
                    C[v_i, v_j] = T.float32(0.0)
                C[v_i, v_j] = C[v_i, v_j] + T.Cast("float32", A[v_i, v_k]) * T.Cast("float32", B[v_k, v_j])

    @T.prim_func
    def after(
        A: T.Buffer((8, 8), "float16"),
        B: T.Buffer((8, 8), "float16"),
        C: T.Buffer((8, 8), "float32"),
    ):
        T.func_attr({"global_symbol": "before"})
        for i_0, j_0, k_0 in T.grid(2, 2, 2):
            with T.sblock(""):
                T.reads()
                T.writes()
                B_local_fragment = T.sblock_alloc_buffer(
                    (4, 4),
                    "float16",
                    scope="local.fragment",
                )
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tirx.op.Op.get("tl.tileop.copy"),
                        _region(B[k_0 * 4, j_0 * 4], 1, 4, 4),
                        _region(B_local_fragment[0, 0], 2, 4, 4),
                    )
                )
                with T.sblock(""):
                    T.reads()
                    T.writes()
                    A_local_fragment = T.sblock_alloc_buffer(
                        (4, 4),
                        "float16",
                        scope="local.fragment",
                    )
                    T.evaluate(
                        T.call_intrin(
                            "handle",
                            tirx.op.Op.get("tl.tileop.copy"),
                            _region(A[i_0 * 4, k_0 * 4], 1, 4, 4),
                            _region(A_local_fragment[0, 0], 2, 4, 4),
                        )
                    )
                    with T.sblock(""):
                        T.reads()
                        T.writes()
                        C_local_fragment = T.sblock_alloc_buffer(
                            (4, 4),
                            scope="local.fragment",
                        )
                        T.evaluate(
                            T.call_intrin(
                                "handle",
                                tirx.op.Op.get("tl.tileop.fill"),
                                _region(C_local_fragment[0, 0], 2, 4, 4),
                                T.float32(0.0),
                            )
                        )
                        T.evaluate(
                            T.call_intrin(
                                "handle",
                                tirx.op.Op.get("tl.tileop.gemm"),
                                _region(A_local_fragment[0, 0], 1, 4, 4),
                                _region(B_local_fragment[0, 0], 1, 4, 4),
                                _region(C_local_fragment[0, 0], 3, 4, 4),
                                T.bool(False),
                                T.bool(False),
                                4,
                                4,
                                4,
                                0,
                                T.bool(False),
                                4,
                                4,
                                0,
                                0,
                                1,
                                0,
                                T.uint32(0),
                                0,
                                0,
                            )
                        )

    sch = Schedule(before)
    block = sch.get_sblock("C")
    _, _, reduce_outer, _, _, _ = sch.get_loops(block)
    sch.cache_write_at(reduce_outer, block, 0, "local.fragment", write_back=False)
    block = sch.get_sblock("C")
    sch.fill_at(reduce_outer, block, 0, 0.0)
    block = sch.get_sblock("C")
    sch.cache_read_at(reduce_outer, block, 0, "local.fragment")
    block = sch.get_sblock("C")
    sch.cache_read_at(reduce_outer, block, 1, "local.fragment")
    block = sch.get_sblock("C")
    sch.gemm_at(reduce_outer, block)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
