import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower
import pytest


def test_copy_matching_dtypes_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            T.copy(A[:256], temp)

    lower(kernel, target="ascend")


def test_copy_dtype_mismatch_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            T.copy(A[:256], temp)

    with pytest.raises(ValueError, match="DMA copies cannot perform type casting"):
        lower(kernel, target="ascend")


def test_copy_dtype_mismatch_inside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            with T.SimtVF(threads=128):
                T.copy(A[:256], temp)

    lower(kernel, target="ascend")


def test_parallel_outside_vf():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float32")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            for i in T.Parallel(256):
                temp[i] = A[i]

    with pytest.raises(ValueError, match="Parallel loops outside VF blocks"):
        lower(kernel, target="ascend")


def test_simd_vf_load_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    temp[i] = A[i]

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simd_vf_store_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    A[i] = temp[i]

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simd_vf_copy_global():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimdVF():
                for i in T.Parallel(256):
                    T.copy(A[i], temp[i])

    with pytest.raises(ValueError, match="SIMD_VF blocks cannot access global memory"):
        lower(kernel, target="ascend")


def test_simt_vf_access_global_allowed():
    @T.prim_func
    def kernel(A: T.Buffer((256,), "float16")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float16")
            with T.SimtVF(threads=128):
                for i in T.Parallel(256):
                    temp[i] = A[i]

    lower(kernel, target="ascend")


def test_outer_local_var_region_write_inside_vf_rejected():
    @T.prim_func
    def kernel(O: T.Buffer((1,), "int32")):
        with T.Kernel(1):
            value = T.alloc_var("int32")
            with T.SimtVF(threads=32):
                T.fill(value, 1)
            O[0] = value

    with pytest.raises(ValueError, match="pointer-type capture"):
        lower(kernel, target="ascend")


def test_simdvf_shared_allocation_escape_rejected():
    with pytest.raises(RuntimeError, match=r"Immutable variable `temp` is used outside its defining region"):

        @T.prim_func
        def kernel(O: T.Buffer((64,), "float32")):
            with T.Kernel(1):
                with T.SimdVF():
                    temp = T.alloc_shared((64,), "float32")
                    for i in T.Parallel(64):
                        temp[i] = T.float32(1)
                T.copy(temp, O)


def test_simtvf_shared_allocation_escape_rejected():
    with pytest.raises(RuntimeError, match=r"Immutable variable `temp` is used outside its defining region"):

        @T.prim_func
        def kernel(O: T.Buffer((64,), "float32")):
            with T.Kernel(1):
                with T.SimtVF(threads=32):
                    temp = T.alloc_shared((64,), "float32")
                    for i in T.Parallel(64):
                        temp[i] = T.float32(1)
                T.copy(temp, O)


def test_simtvf_local_allocation_escape_rejected():
    with pytest.raises(RuntimeError, match=r"Immutable variable `temp` is used outside its defining region"):

        @T.prim_func
        def kernel(O: T.Buffer((64,), "float32")):
            with T.Kernel(1):
                with T.SimtVF(threads=32):
                    temp = T.alloc_local((64,), "float32")
                    for i in T.Parallel(64):
                        temp[i] = T.float32(1)
                O[0] = temp[0]


def test_nested_buffer_load_checks_vf_allocation_escape():
    with pytest.raises(RuntimeError, match=r"Immutable variable `index` is used outside its defining region"):

        @T.prim_func
        def kernel(A: T.Buffer((64,), "int32"), O: T.Buffer((64,), "int32")):
            with T.Kernel(1):
                with T.SimtVF(threads=32):
                    index = T.alloc_shared((1,), "int32")
                    index[0] = 0
                O[0] = A[index[0]]


def test_outer_shared_allocation_used_inside_and_outside_vf_allowed():
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                for i in T.Parallel(64):
                    temp[i] = T.float32(1)
            T.copy(temp, O)

    lower(kernel, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
