import tilelang.testing
from tilelang import language as T


def test_issue_1237_dynamic_copy_extent_builds():
    # Repro from debug/1113_issues/copy_dyn.py, adapted as a unit test.
    # The goal is to ensure T.copy correctly handles dynamic extents
    # (e.g., src slice length vs. static dst buffer size) during prim_func building.

    length = T.symbolic("len", dtype=T.int32)

    @T.prim_func
    def sample_kernel(global_tensor: T.Tensor[(length,), T.int32]):  # noqa: F821
        with T.Kernel(1, threads=32):
            buffer_shared = T.alloc_shared((1024,), dtype=T.int32)
            T.copy(global_tensor[0:length], buffer_shared)

    # Building the prim_func is sufficient to exercise the bug path; no need to JIT/execute.
    _ = sample_kernel


if __name__ == "__main__":
    tilelang.testing.main()
