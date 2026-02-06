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
    mod = tl.transform.InjectFenceProxy()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    transformed = tir.transform.LowerOpaqueBlock()(transformed)

    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_lower_fence_proxy():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2 : i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.tl_gemm"),
                "tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>",
                T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
            )

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1, 8, 256), T.float16, scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), T.float16, scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2 : i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            T.fence_proxy_async()
            T.call_intrin(
                "handle",
                tir.op.Op.get("tl.tl_gemm"),
                "tl::gemm_ss<128, 128, 32, 4, 1, 0, 0, 0, 32, 128, 0, 0, true>",
                T.tvm_access_ptr(T.type_annotation(T.float16), A_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float16), B_shared.data, 0, 2048, 1),
                T.tvm_access_ptr(T.type_annotation(T.float32), C_local.data, 0, 32, 3),
            )

    _check(before, after)


def test_async_to_generic_no_double_fence():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            T.ptx_cp_async("uint8", A_shared.data, 0, B_shared.data, 0, 16)
            T.fence_proxy_async()
            T.call_extern("handle", "generic_op")

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    def _count_fences(stmt):
        count = 0

        def visit(node):
            nonlocal count
            if isinstance(node, tir.Evaluate):
                call = node.value
                if isinstance(call, tir.Call):
                    op = call.op
                    name = getattr(op, "name", None)
                    if name == "tl.fence_proxy_async":
                        count += 1

        tir.stmt_functor.post_order_visit(stmt, visit)
        return count

    assert _count_fences(mod["main"].body) == 1


def test_proxy_hint_override():
    @T.prim_func
    def before():
        with T.Kernel(8):
            T.evaluate(T.call_extern("handle", "custom_async"))
            with T.attr("proxy_scope", "tl.proxy_hint", "neutral"):
                T.evaluate(T.call_extern("handle", "custom_generic"))
            T.evaluate(T.call_extern("handle", "custom_async_tail"))

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    def _has_fence(stmt):
        result = False

        def visit(node):
            nonlocal result
            if isinstance(node, tir.Evaluate):
                call = node.value
                if isinstance(call, tir.Call):
                    op = call.op
                    name = getattr(op, "name", None)
                    if name == "tl.fence_proxy_async":
                        result = True

        tir.stmt_functor.post_order_visit(stmt, visit)
        return result

    assert not _has_fence(mod["main"].body)


def test_tma_store_sync_injection():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_global = T.decl_buffer((128,), T.float16, scope="global")
            T.evaluate(T.call_intrin("handle", tir.op.Op.get("tl.tma_store"), A_global.data))

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)

    arrives = 0
    waits = 0

    def visit(node):
        nonlocal arrives, waits
        if isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call):
                name = getattr(call.op, "name", None)
                if name == "tl.tma_store_arrive":
                    arrives += 1
                elif name in ("tl.tma_store_wait", "tl.tma_store_wait<0>"):
                    waits += 1

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert arrives == 1
    assert waits == 1


def test_wgmma_marked_async():
    @T.prim_func
    def before():
        with T.Kernel(1):
            A_shared = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            A_shared[0] = T.float16(0)
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.InjectFenceProxy()(mod)
    order = []

    def visit(node):
        if isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir.Call):
                order.append(getattr(call.op, "name", ""))

    tir.stmt_functor.post_order_visit(mod["main"].body, visit)

    assert "tl.ptx_wgmma_ss" in order
    assert "tl.fence_proxy_async" in order
    assert order.index("tl.fence_proxy_async") < order.index("tl.ptx_wgmma_ss")


if __name__ == "__main__":
    tilelang.testing.main()
