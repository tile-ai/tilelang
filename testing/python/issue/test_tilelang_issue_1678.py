import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit
def _alloc_var_mixed_dtype_kernel():
    @T.prim_func
    def kernel():
        with T.Kernel(1, 1, threads=1) as (_, _):
            i = T.alloc_var(T.int32)
            i = 1
            tmp_row = T.alloc_local((4,), T.float32)
            amax_local = T.alloc_var(T.float32)
            j = i
            amax_local = T.max(amax_local, tmp_row[j])

    return kernel


@tilelang.testing.requires_cuda
def test_alloc_var_mixed_dtype_codegen():
    kernel = _alloc_var_mixed_dtype_kernel()
    source = kernel.get_kernel_source()
    assert "int i" in source
    assert "float amax_local" in source


if __name__ == "__main__":
    tilelang.testing.main()
