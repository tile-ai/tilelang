# type: ignore
import tilelang
import tilelang.testing
import tilelang.language as T


# TODO(dyq) It intentionally triggers a device-side assert so we can't include this in CI
# Please run manually when you want to verify that device_assert actually traps on GPU.
def _manual_device_assert_triggered():
    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid > 0, "Assertion Trigger !")

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_device_assert_no_trigger():
    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid == tid)

    jit_kernel = tilelang.compile(program, target="cuda")
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


if __name__ == "__main__":
    _manual_device_assert_triggered()
