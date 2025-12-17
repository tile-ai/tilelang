from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LetInline()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), True)


def test_let_binding():
    @T.prim_func
    def before(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.block("compute"):
                    factor = T.float32(2.0)
                    value = A[i, j] * factor
                    B[i, j] = value

    @T.prim_func
    def expected(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.block("compute"):
                    B[i, j] = A[i, j] * T.float32(2.0)

    _check(before, expected)


def test_parallel_scope():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.block("parallel"):
                value = T.float32(1.0)
                A[i] = value

    @T.prim_func
    def expected(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.block("parallel"):
                A[i] = T.float32(1.0)

    _check(before, expected)


if __name__ == "__main__":
    tilelang.testing.main()
