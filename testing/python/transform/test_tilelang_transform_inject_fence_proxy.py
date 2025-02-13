# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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
            A_shared = T.decl_buffer((1, 8, 256), "float16", scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), "float16", scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            T.call_extern("handle", "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                          T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 2048, 1),
                          T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 2048, 1),
                          T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 32, 3))

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1, 8, 256), "float16", scope="shared.dyn")
            B_shared = T.decl_buffer((1, 4, 512), "float16", scope="shared.dyn")
            C_local = T.decl_buffer((32,), scope="local")
            for i in T.unroll(16):
                C_local[i * 2:i * 2 + 2] = T.Broadcast(T.float32(0), 2)
            T.FenceProxyAsyncOp()
            T.call_extern("handle", "tl::gemm_ss<64, 64, 32, 4, 1, 0, 0>",
                          T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 2048, 1),
                          T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 2048, 1),
                          T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 32, 3))

    _check(before, after)


if __name__ == "__main__":
    test_lower_fence_proxy()
