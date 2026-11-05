import tvm_tirx
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import pytest


def _check(original, transformed):
    func = original
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.LetInline()(mod)
    expected = transformed.with_attr("global_symbol", "main")
    tvm_tirx.tirx.analysis.check_well_formed(mod["main"])
    tvm_tirx.tirx.analysis.check_well_formed(expected)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_let_binding():
    @T.prim_func
    def before(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    factor = T.float32(2.0)
                    value = A[i, j] * factor
                    B[i, j] = value

    @T.prim_func
    def expected(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    # A native Bind that contains a load preserves its read
                    # point. Only the pure constant factor is inlined.
                    value = A[i, j] * T.float32(2.0)
                    B[i, j] = value

    _check(before, expected)


def test_parallel_scope():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                value = T.float32(1.0)
                A[i] = value

    @T.prim_func
    def expected(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                A[i] = T.float32(1.0)

    _check(before, expected)


@pytest.mark.parametrize("write_first", [False, True])
def test_tensor_read_bind_keeps_its_evaluation_point(write_first):
    @T.prim_func
    def main(A: T.Tensor((1,), T.float32), B: T.Tensor((2,), T.float32)):
        if write_first:
            A[0] = T.float32(7.0)
        value = A[0]
        A[0] = T.float32(5.0)
        B[0] = value
        B[1] = A[0]

    # Moving the Bind into B[0]'s store would read 5.0 in both cases. Its
    # original position must survive even when another store precedes it.
    _check(main, main)


if __name__ == "__main__":
    tilelang.testing.main()
