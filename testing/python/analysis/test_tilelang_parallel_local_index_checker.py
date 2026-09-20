import pytest

import tilelang
import tilelang.language as T


@tilelang.jit
def invalid_local_store():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i in T.Parallel(32):
                local[i] = T.float32(1)

    return main


@tilelang.jit
def invalid_local_load():
    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i in T.serial(32):
                local[i] = T.float32(i)
            for i in T.Parallel(32):
                out[i] = local[i]

    return main


@tilelang.jit
def invalid_nested_parallel_index():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i, j in T.Parallel(8, 4):
                local[i * 4 + j] = T.float32(1)

    return main


@tilelang.jit
def valid_constant_local_index():
    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            scale_local = T.alloc_local((1,), T.float32)
            scale_local[0] = T.float32(2)
            for i in T.Parallel(32):
                out[i] = scale_local[0]

    return main


@tilelang.jit
def valid_inner_vectorized_local_index():
    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            scratch = T.alloc_local((4,), T.float32)
            for i in T.Parallel(32):
                for lane in T.vectorized(4):
                    scratch[lane] = T.float32(lane)
                out[i] = scratch[0]

    return main


def test_parallel_local_index_rejected():
    message = "Local buffer.*is indexed by T.Parallel loop variable"
    with pytest.raises(ValueError, match=message):
        invalid_local_store()
    with pytest.raises(ValueError, match=message):
        invalid_local_load()
    with pytest.raises(ValueError, match=message):
        invalid_nested_parallel_index()


def test_parallel_independent_local_index_allowed():
    valid_constant_local_index()
    valid_inner_vectorized_local_index()


if __name__ == "__main__":
    tilelang.testing.main()
