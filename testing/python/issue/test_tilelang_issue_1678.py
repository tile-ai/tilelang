# ruff: noqa
import tilelang
import tilelang.testing
import tilelang.language as T


def test_issue_1678():
    @tilelang.jit
    def qwq():
        @T.prim_func
        def qwq_kernel():
            with T.Kernel(4096, 1, threads=1) as (pid_y, pid_x):
                i = T.alloc_var("int32")
                i = 1
                tmp_row = T.alloc_local((4,), "float32")
                amax_local = T.alloc_var("float32")
                j = 0
                amax_local = T.max(amax_local, tmp_row[j])

        return qwq_kernel

    kernel = qwq()


if __name__ == "__main__":
    tilelang.testing.main()
