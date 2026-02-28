import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
from tilelang import tvm


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


@tilelang.testing.requires_cuda
def test_async_copy_tileop_lowers_to_cp_async():
    """Check T.async_copy always uses CPAsync path and does not auto-wait."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((4,), T.float16)
            T.async_copy(A[0:4], S)
            T.copy(S, B[0:4])

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== async_copy -> cp.async codegen ===")
    print(src)
    assert "cp_async_gs<8>" in src, "Expected T.async_copy to lower to cp_async_gs<8>"
    assert "tl::cp_async_commit" in src, "Expected async_copy lowering to emit commit"
    assert "tl::cp_async_wait<0>" not in src, "Did not expect async_copy lowering to auto-emit wait"


@tilelang.testing.requires_cuda
def test_async_copy_tileop_rejects_invalid_cp_async_scope():
    """Check T.async_copy rejects non global->shared patterns."""

    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S0 = T.alloc_shared((4,), T.float16)
            S1 = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S0)
            # shared->shared cannot use cp.async and should fail for async_copy.
            T.async_copy(S0, S1)
            T.copy(S1, B[0:4])

    with pytest.raises(
        tvm.error.InternalError,
        match="T\\.async_copy only supports global->shared/shared\\.dyn copies",
    ):
        tilelang.compile(main, out_idx=[1], target="cuda")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_copy_global_to_shared_oob_lowers_to_predicated_cp_async():
    """Check T.copy can lower OOB-guarded global->shared copy to predicated cp.async."""

    M = 130
    K = 32
    block_m = 128
    block_k = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
            S = T.alloc_shared((block_m, block_k), T.float16)
            # OOB on the M dimension for the last block -> requires predicate,
            # but predicate is uniform across the vectorized K dimension.
            T.copy(A[pid_m * block_m : (pid_m + 1) * block_m, 0:block_k], S, disable_tma=True)
            T.copy(S, B[pid_m * block_m : (pid_m + 1) * block_m, 0:block_k])

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== OOB copy -> predicated cp.async codegen ===")
    print(src)
    assert "cp_async_gs_conditional<" in src, "Expected predicated cp.async (zero-fill) in generated CUDA source"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_oob_lowers_to_predicated_cp_async_without_wait():
    """Check T.async_copy supports OOB via predicated cp.async and does not auto-wait."""

    M = 130
    K = 32
    block_m = 128
    block_k = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m)) as pid_m:
            S = T.alloc_shared((block_m, block_k), T.float16)
            T.async_copy(A[pid_m * block_m : (pid_m + 1) * block_m, 0:block_k], S)
            # Don't read S here (no wait). Keep B live so kernel has an output.
            B[0, 0] = A[0, 0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print("=== OOB async_copy -> predicated cp.async codegen ===")
    print(src)
    assert "cp_async_gs_conditional<" in src, "Expected predicated cp.async (zero-fill) in generated CUDA source"
    assert "tl::cp_async_commit" in src, "Expected async_copy lowering to emit commit"
    assert "tl::cp_async_wait<0>" not in src, "Did not expect async_copy lowering to auto-emit wait"


if __name__ == "__main__":
    tilelang.testing.main()
