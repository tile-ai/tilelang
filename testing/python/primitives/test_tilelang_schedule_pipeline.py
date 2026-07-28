"""Before/after IR tests for ``Schedule.pipeline``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.schedule import Schedule
from tvm.script import tirx as T


def test_pipeline_adds_num_stages_annotation_to_target_loop():
    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        for i in range(16):
            with T.sblock("B"):
                v_i = T.axis.spatial(16, i)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] + T.float32(1.0)

    @T.prim_func
    def after(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        T.func_attr({"global_symbol": "before"})
        for i in T.serial(16, annotations={"num_stages": T.int32(4)}):
            with T.sblock("B"):
                v_i = T.axis.spatial(16, i)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] + T.float32(1.0)

    sch = Schedule(before)
    block = sch.get_sblock("B")
    sch.pipeline(sch.get_loops(block)[0], 4)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
