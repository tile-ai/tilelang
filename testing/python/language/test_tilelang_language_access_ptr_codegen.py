import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_access_ptr_cp_async_codegen():
    """Smoke-test codegen for T.access_ptr -> tl.access_ptr -> tvm_access_ptr -> cp.async."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.uint8),
        B: T.Tensor((64,), T.uint8),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.uint8)
            T.ptx_cp_async(
                T.access_ptr(S[8], "w", 16),
                T.access_ptr(A[16], "r", 16),
                16,
            )
            # Keep the shared buffer live so the pointers remain in generated code.
            B[0] = S[8]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== access_ptr cp.async codegen ===")
    print(src)
    assert "cp_async_gs<16>" in src, "Expected cp_async_gs<16> in generated CUDA source"


if __name__ == "__main__":
    tilelang.testing.main()
