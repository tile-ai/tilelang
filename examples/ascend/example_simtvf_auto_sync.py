import tilelang
import tilelang.ascend.language as T


def sync_kernel(N):
    @T.prim_func
    def main(
        A: T.Buffer((N,), "int32"),
    ):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((N,), "int32")
            with T.SimtVF(threads=N):
                tx = T.get_thread_binding()
                temp[tx] = tx ^ 1023
                A[tx] = temp[tx ^ 1023]

    return main


if __name__ == "__main__":
    kernel = tilelang.compile(sync_kernel(1024))
    source = kernel.get_kernel_source()
    assert "asc_syncthreads" in source
    print("AutoSync Success")
