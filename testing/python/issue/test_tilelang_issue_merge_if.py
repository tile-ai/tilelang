import numpy as np

import tilelang
from tilelang import tvm as tvm
from tvm.ir import IRModule
import tilelang.testing
import tilelang.language as T


def merge_if_test():
    @T.prim_func
    def main():
        A = T.alloc_fragment((1,), T.float16)
        B = T.alloc_fragment((1,), T.float16)
        C = T.alloc_fragment((1,), T.float16)
        D = T.alloc_fragment((1,), T.float16)
        if A[0] == 0:
            A[0] = 0
        if B[0] == 0:
            B[0] = 0
        if C[0] == 0:
            C[0] = 0
        if D[0] == 0:
            D[0] = 0

    return main


def test_merge_if():
    func = merge_if_test()
    original_module = IRModule.from_expr(func)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    tvm.ir.assert_structural_equal(original_module["main"], transformed["main"], True)


def test_merge_if_preserves_re_evaluation_of_buffer_condition():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        if A[0] > 0:
            A[0] = 0
        if A[0] > 0:
            A[1] = 1

    original_module = IRModule.from_expr(main)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    executable = tvm.compile(transformed["main"], target="c").jit(options=["-std=c++17"])

    values = tvm.runtime.tensor(np.array([1, 7], dtype="int32"))
    executable(values)

    np.testing.assert_array_equal(values.numpy(), np.array([0, 7], dtype="int32"))


if __name__ == "__main__":
    tilelang.testing.main()
