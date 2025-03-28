# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from tilelang import tvm as tvm
import tilelang as tl
from tilelang.utils.target import determine_target
import tilelang.language as T
import tilelang.testing

auto_target = tvm.target.Target(determine_target("auto"))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(auto_target)(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.Simplify()(mod)
    transformed = tvm.IRModule.from_expr(transformed.with_attr("global_symbol", "main"))
    transformed = tvm.tir.transform.BindTarget(auto_target)(transformed)
    tvm.ir.assert_structural_equal(mod["main"], transformed["main"], True)


def test_simple_pipeline():

    @T.prim_func
    def before(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    @T.prim_func
    def after(A: T.Tensor((1024, 32), "float32"), B: T.Tensor((32, 1024), "float32"), C: T.Tensor(
        (1024, 1024), "float32")):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "float32")
            B_shared = T.alloc_shared((32, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")

            T.clear(C_local)

            for ko in T.serial(
                    32,
                    annotations={
                        "software_pipeline_async_stages": [0],
                        "software_pipeline_order": [0, 1, 2],
                        "software_pipeline_stage": [3, 3, 3]
                    }):
                T.copy(A[by * 128, ko * 32], A_shared)
                T.copy(B[ko * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])

    _check(before, after)


if __name__ == "__main__":
    tilelang.testing.main()
