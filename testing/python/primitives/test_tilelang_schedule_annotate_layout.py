"""Before/after IR tests for ``Schedule.annotate_layout``."""

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.layout import Layout
from tilelang.schedule import Schedule
from tvm.script import tirx as T

_LAYOUT = Layout((4,), lambda i: i)


def test_annotate_layout_adds_layout_map_for_named_alloc_buffer():
    @T.prim_func(check_well_formed=False)
    def before(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        with T.sblock("root"):
            T.reads()
            T.writes()
            A_local_fragment = T.sblock_alloc_buffer((4,), scope="local.fragment")
            T.evaluate(A_local_fragment.data)

    @T.prim_func(check_well_formed=False)
    def after(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        T.func_attr({"global_symbol": "before"})
        with T.sblock("root"):
            T.reads()
            T.writes()
            A_local_fragment = T.handle("float32", "local.fragment")
            T.sblock_attr({"layout_map": {A_local_fragment: _LAYOUT}})
            A_local_fragment_1 = T.sblock_alloc_buffer(
                (4,),
                data=A_local_fragment,
                scope="local.fragment",
            )
            T.evaluate(A_local_fragment_1.data)

    sch = Schedule(before)
    sch.annotate_layout(sch.get_sblock("root"), "A_local_fragment", _LAYOUT)

    tvm.ir.assert_structural_equal(sch.mod["main"], after)


if __name__ == "__main__":
    tilelang.testing.main()
