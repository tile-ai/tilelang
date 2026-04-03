# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T


auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.ReuseLocalDescriptorAllocations()(mod)

    expected = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    expected = tvm.tir.transform.BindTarget(auto_target)(expected)

    tvm.ir.assert_structural_equal(mod["main"], expected["main"], True)


def test_reuse_local_descriptor_allocations():
    @T.prim_func
    def before():
        T.func_attr({"tir.noalias": True})
        with T.attr(0, "test.region", 0):
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_a_buf = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_buf = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            T.initialize_wgmma_descriptor(desc_a_buf[0], T.uint64(0), 1, 1, 64)
            T.initialize_wgmma_descriptor(desc_b_buf[0], T.uint64(0), 1, 1, 64)
            T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))
        with T.attr(0, "test.region", 1):
            desc_a_1 = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b_1 = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_a_buf_1 = T.Buffer((1,), "uint64", data=desc_a_1, scope="local.descriptor.wgmma")
            desc_b_buf_1 = T.Buffer((1,), "uint64", data=desc_b_1, scope="local.descriptor.wgmma")
            T.initialize_wgmma_descriptor(desc_a_buf_1[0], T.uint64(1), 1, 1, 64)
            T.initialize_wgmma_descriptor(desc_b_buf_1[0], T.uint64(1), 1, 1, 64)
            T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a_1, desc_b_1))

    @T.prim_func
    def after():
        T.func_attr({"tir.noalias": True})
        desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
        desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
        with T.attr(0, "test.region", 0):
            desc_a_buf = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_buf = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            T.initialize_wgmma_descriptor(desc_a_buf[0], T.uint64(0), 1, 1, 64)
            T.initialize_wgmma_descriptor(desc_b_buf[0], T.uint64(0), 1, 1, 64)
            T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))
        with T.attr(0, "test.region", 1):
            desc_a_buf_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
            desc_b_buf_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
            T.initialize_wgmma_descriptor(desc_a_buf_1[0], T.uint64(1), 1, 1, 64)
            T.initialize_wgmma_descriptor(desc_b_buf_1[0], T.uint64(1), 1, 1, 64)
            T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))

    _check(before, after)


def test_reuse_local_descriptor_allocations_stays_inside_launch_thread():
    @T.prim_func
    def before():
        T.func_attr({"tir.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.attr(0, "test.region", 0):
                desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
                desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
                desc_a_buf = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
                desc_b_buf = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(desc_a_buf[0], T.uint64(0), 1, 1, 64)
                T.initialize_wgmma_descriptor(desc_b_buf[0], T.uint64(0), 1, 1, 64)
                T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))
            with T.attr(0, "test.region", 1):
                desc_a_1 = T.allocate([1], "uint64", "local.descriptor.wgmma")
                desc_b_1 = T.allocate([1], "uint64", "local.descriptor.wgmma")
                desc_a_buf_1 = T.Buffer((1,), "uint64", data=desc_a_1, scope="local.descriptor.wgmma")
                desc_b_buf_1 = T.Buffer((1,), "uint64", data=desc_b_1, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(desc_a_buf_1[0], T.uint64(1), 1, 1, 64)
                T.initialize_wgmma_descriptor(desc_b_buf_1[0], T.uint64(1), 1, 1, 64)
                T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a_1, desc_b_1))

    @T.prim_func
    def after():
        T.func_attr({"tir.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            desc_a = T.allocate([1], "uint64", "local.descriptor.wgmma")
            desc_b = T.allocate([1], "uint64", "local.descriptor.wgmma")
            with T.attr(0, "test.region", 0):
                desc_a_buf = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
                desc_b_buf = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(desc_a_buf[0], T.uint64(0), 1, 1, 64)
                T.initialize_wgmma_descriptor(desc_b_buf[0], T.uint64(0), 1, 1, 64)
                T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))
            with T.attr(0, "test.region", 1):
                desc_a_buf_1 = T.Buffer((1,), "uint64", data=desc_a, scope="local.descriptor.wgmma")
                desc_b_buf_1 = T.Buffer((1,), "uint64", data=desc_b, scope="local.descriptor.wgmma")
                T.initialize_wgmma_descriptor(desc_a_buf_1[0], T.uint64(1), 1, 1, 64)
                T.initialize_wgmma_descriptor(desc_b_buf_1[0], T.uint64(1), 1, 1, 64)
                T.evaluate(T.call_extern("handle", "use_desc_pair", desc_a, desc_b))

    _check(before, after)
