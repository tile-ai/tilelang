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


@tilelang.testing.requires_cuda
def test_vectorized_cp_async_bytes_codegen():
    """Check vectorized ptx_cp_async byte folding (elem_bytes * lanes)."""

    @T.prim_func
    def main(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.float16)
            for i in T.vectorized(4):
                T.ptx_cp_async(
                    T.access_ptr(S[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    2,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[0] = S[0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== vectorized cp.async codegen ===")
    print(src)
    assert "cp_async_gs<8>" in src, "Expected vectorized cp.async bytes to fold into cp_async_gs<8>"
    assert "cp_async_gs<2>" not in src, "Did not expect scalar cp.async bytes in generated CUDA source"


@tilelang.testing.requires_cuda
def test_copy_global_to_shared_lowers_to_cp_async():
    """Check T.copy can choose CPAsync instruction for global->shared copy."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S, disable_tma=True)
            T.copy(S, B[0:4])

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== copy -> cp.async codegen ===")
    print(src)
    assert "cp_async_gs<8>" in src, "Expected T.copy(global->shared) to lower to cp_async_gs<8>"
    assert "tl::cp_async_commit" in src, "Expected CPAsync lowering to emit commit"
    assert "tl::cp_async_wait<0>" in src, "Expected CPAsync lowering to emit wait"


if __name__ == "__main__":
    tilelang.testing.main()
