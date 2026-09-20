"""PTO codegen must print Let-bearing For extents (post-#382 shape >= 0)."""

import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def dynamic_persistent_copy_kernel():
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n, 64), T.float32), B: T.Tensor((n, 64), T.float32)):
        with T.Kernel(1) as pid:
            for idx in T.Persistent([n], 1, pid, group_size=32, num_stages=2):
                ub = T.alloc_shared((64,), T.float32)
                T.copy(A[idx, 0], ub)
                T.copy(ub, B[idx, 0])

    return main


def test_pto_dynamic_persistent_lowers_lets():
    source = lower(dynamic_persistent_copy_kernel(), target="pto").kernel_source
    # Regression for #382: missing Let visitor used to throw
    # ``Do not have a default for tirx.Let`` while printing For extents.
    assert "tirx.Let" not in source, source
    assert "pto.for_" in source or " in range(" in source, source


if __name__ == "__main__":
    tilelang.testing.main()
