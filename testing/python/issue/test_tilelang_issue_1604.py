import tilelang
import tilelang.testing
import tilelang.language as T
import re


@tilelang.jit
def qwq():
    dtype = "float32"

    @T.prim_func
    def main(out: T.Tensor[(512,), dtype]):
        with T.Kernel(1, threads=512):
            A = T.alloc_shared((32,), dtype)
            B = T.alloc_shared((32,), dtype)

            tid = T.get_thread_binding()
            if tid < 32:
                A[tid] = tid
                B[tid] = tid

            out[tid] = A[tid % 32]

    return main


def test_issue_1604():
    kernel = qwq()
    print(kernel.get_kernel_source())
    target = "__syncthreads"
    pattern = r"if [^{]*{[^}]*\b" + re.escape(target) + r"\b[^}]*}"
    assert len(re.findall(pattern, kernel.get_kernel_source())) == 0


if __name__ == "__main__":
    tilelang.testing.main()
