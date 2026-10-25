import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang._compat.stmt_functor import post_order_visit
from tvm_tirx import ir


def test_issue_1734():
    """Test that loop-invariant if statements are hoisted out of loops."""

    @tilelang.jit()
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor[(2, 512), T.float32],
            B: T.Tensor[(2, 512), T.float32],
            C: T.Tensor[(2,), T.float32],
        ):
            with T.Kernel(1, threads=128):
                A_local = T.alloc_fragment((2, 512), T.float32)
                B_local = T.alloc_fragment((2, 512), T.float32)
                C_local = T.alloc_fragment((2,), T.float32)

                # Each thread owns 8 contiguous elements of one row, so i is
                # constant per thread and the guard below is loop-invariant,
                # while a residual serial loop survives vectorization.
                row_chunks = T.Fragment((2, 512), forward_fn=lambda i, j: (i * 64 + j // 8, j % 8))
                T.annotate_layout({A_local: row_chunks, B_local: row_chunks})

                T.copy(A, A_local)
                T.copy(C, C_local)

                for i, j in T.Parallel(2, 512):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]

                T.copy(B_local, B)

        return main

    mod = kernel.compile()
    guarded_loops = []

    def inspect_guard(node):
        if not isinstance(node, ir.If):
            return
        loads = []
        post_order_visit(node.condition, lambda value: loads.append(value) if isinstance(value, ir.TensorLoad) else None)
        if any("C_local" in load.source.name for load in loads):
            post_order_visit(node.then_case, lambda value: guarded_loops.append(value) if isinstance(value, ir.For) else None)

    for function in mod.artifact.device_mod.functions.values():
        post_order_visit(function, inspect_guard)
    assert guarded_loops, "Loop-invariant if should enclose the residual loop"


if __name__ == "__main__":
    tilelang.testing.main()
