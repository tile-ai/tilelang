"""Regression for analyzer bindings across parallel-loop rewrites."""

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_nested_serial_bounds_after_thread_partition():
    @T.prim_func
    def main(bounds: T.Tensor((32, 2), T.int32), out: T.Tensor((32, 4, 4), T.int32)):
        with T.Kernel(1, threads=32):
            for p in T.Parallel(32):
                for y in T.serial(bounds[p, 0]):
                    for x in T.serial(bounds[p, 1]):
                        out[p, y, x] = x

    # Partitioning changes bounds[p, ...] without changing the serial loop Vars.
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        tilelang.lower(main, target=target, enable_host_codegen=False, enable_device_compile=False)


if __name__ == "__main__":
    tilelang.testing.main()
