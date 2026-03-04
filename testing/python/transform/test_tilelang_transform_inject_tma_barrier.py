from tilelang import tvm
import tilelang as tl
import tilelang.language as T
from tvm import tir


def test_arrive_expect_tx_in_elect_updates_barrier_thread_count():
    @T.prim_func
    def before():
        T.call_intrin(
            "handle",
            tir.op.Op.get("tl.create_list_of_mbarrier"),
            128,
            128,
            128,
            512,
            512,
            512,
            1,
        )
        for nbn_i in range(6):
            if T.shuffle_elect(128):
                T.evaluate(
                    tir.Call(
                        "handle",
                        "tir.ptx_arrive_barrier_expect_tx",
                        [T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), nbn_i % 3), 16384],
                    )
                )
            T.evaluate(
                tir.Call(
                    "handle",
                    "tir.ptx_arrive_barrier",
                    [T.call_intrin("handle", tir.op.Op.get("tl.get_mbarrier"), nbn_i % 3 + 3)],
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectTmaBarrier()(mod)

    # Barriers 0..2 are arrived only by elected thread, 3..5 by all threads.
    assert "T.create_list_of_mbarrier(1, 1, 1, 640, 640, 640, 1)" in mod.script()


if __name__ == "__main__":
    tl.testing.main()
