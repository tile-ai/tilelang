from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    print(mod["main"])
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"),
                                   True)


def test_trival_pipeline():

    @T.prim_func
    def before(A: T.Tensor((16, 1), "float32"), C: T.Tensor((16, 1), "float32")):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                    0,
                    1,
                    annotations={
                        "software_pipeline_stage": [0, 1],
                        "software_pipeline_order": [0, 1]
                    }):
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(C[tx, i])
                    B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(B[tx, 0])
                        B[tx, 0] = A[tx, i] * T.float32(2)
                    with T.block():
                        T.reads(B[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = B[tx, 0] + T.float32(1)

    @T.prim_func
    def expected(A: T.Tensor((16, 1), "float32"), C: T.Tensor((16, 1), "float32")) -> None:
        for tx in T.thread_binding(16, thread="threadIdx.x"):
            with T.block(""):
                T.reads(A[tx, 0])
                T.writes(C[tx, 0])
                B = T.alloc_buffer((2, 16, 1), scope="shared")
                with T.block(""):
                    T.reads(A[tx, 0])
                    T.writes(B[0, tx, 0])
                    B[0, tx, 0] = A[tx, 0] * T.float32(2.0)
                with T.block(""):
                    T.reads()
                    T.writes()
                    T.evaluate(0)
                with T.block(""):
                    T.reads(B[0, tx, 0])
                    T.writes(C[tx, 0])
                    C[tx, 0] = B[0, tx, 0] + T.float32(1.0)

    _check(before, expected)


if __name__ == "__main__":
    tilelang.testing.main()
