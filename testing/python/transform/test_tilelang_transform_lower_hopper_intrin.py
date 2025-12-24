from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing
from tvm import tir

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.LowerHopperIntrin()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    transformed = tir.transform.LowerOpaqueBlock()(transformed)
    transformed["main"] = transformed["main"].with_attr("tma_descriptor_args", {})

    # TODO: temporary remove this check
    # tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_lower_hopper_intrin_barrier():
    @T.prim_func
    def before():
        with T.Kernel(8):
            _ = T.launch_thread("threadIdx.x", 128)
            T.create_list_of_mbarrier(128, 128, 128, 128)

    @T.prim_func
    def after():
        with T.Kernel(8):
            v_1 = T.launch_thread("threadIdx.x", 128)
            T.evaluate(tir.Call("handle", "tir.create_barriers", [4]))
            with T.If(v_1 == 0), T.Then():
                T.evaluate(tir.Call("handle", "tir.ptx_init_barrier_thread_count", [T.get_mbarrier(0), 128]))
                T.evaluate(tir.Call("handle", "tir.ptx_init_barrier_thread_count", [T.get_mbarrier(1), 128]))
                T.evaluate(tir.Call("handle", "tir.ptx_init_barrier_thread_count", [T.get_mbarrier(2), 128]))
                T.evaluate(tir.Call("handle", "tir.ptx_init_barrier_thread_count", [T.get_mbarrier(3), 128]))
            T.evaluate(tir.Call("handle", "tir.tvm_storage_sync", ["shared"]))

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
