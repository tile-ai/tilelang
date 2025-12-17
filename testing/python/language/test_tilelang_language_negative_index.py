from tilelang import tvm
import tilelang as tl
import tilelang.testing
import tilelang.language as T


@T.prim_func
def negative_index_before(A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    B[0] = A[T.int32(-1)]


@T.prim_func
def negative_index_expected(A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    B[0] = A[T.int32(15)]


@T.prim_func
def negative_index_loop_before(A: T.Buffer((16,), T.float32), B: T.Buffer((4,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(4):
        B[i] = A[-i - 1]


@T.prim_func
def negative_index_loop_expected(A: T.Buffer((16,), T.float32), B: T.Buffer((4,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(4):
        B[i] = A[15 - i]


@T.prim_func
def negative_index_symbolic_before(shift: T.int32, A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(16):
        B[i] = A[shift + i]


def test_legalize_negative_index_scalar():
    mod = tvm.IRModule({"main": negative_index_before})
    transformed = tl.transform.LegalizeNegativeIndex()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, negative_index_expected.body)


def test_legalize_negative_index_affine_expr():
    mod = tvm.IRModule({"main": negative_index_loop_before})
    transformed = tl.transform.LegalizeNegativeIndex()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, negative_index_loop_expected.body)


def test_legalize_negative_index_symbolic_passthrough():
    mod = tvm.IRModule({"main": negative_index_symbolic_before})
    transformed = tl.transform.LegalizeNegativeIndex()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, negative_index_symbolic_before.body)


if __name__ == "__main__":
    tilelang.testing.main()
