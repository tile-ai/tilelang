# type: ignore
import tilelang
import tilelang.testing
import tilelang.language as T


def test_device_assert_no_trigger():
    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid == tid)

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


if __name__ == "__main__":
    tilelang.testing.main()
