import tilelang
import tilelang.testing
from tilelang import language as T


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
    source = mod.get_kernel_source()
    # Verify that the if statement is hoisted outside the for loop: the
    # guarded loop must sit inside the if block (a "for (" before the
    # first "}" after the if).
    if_pos = source.find("if (")
    assert if_pos != -1, "Guard should survive lowering"
    after_if = source[if_pos:]
    for_pos = after_if.find("for (")
    assert for_pos != -1 and for_pos < after_if.find("}"), "Loop-invariant if should be hoisted outside the loop"


if __name__ == "__main__":
    tilelang.testing.main()
