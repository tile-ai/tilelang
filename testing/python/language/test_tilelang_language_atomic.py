import re

import pytest
import tilelang.testing
import tilelang.layout
import tilelang.language as T
import torch
from tilelang import tvm


# ======================= Thread-level atomic add =======================


@tilelang.jit
def atomic_add_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_add


def run_atomic_add(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B + A.sum(dim=0)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_memory_order_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_with_memory_order(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_add(B[bx * block_M + i, by * block_N + j], A_shared[i, j], memory_order="relaxed")

    return atomic_with_memory_order


def run_atomic_memory_order(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_memory_order_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B + A.sum(dim=0)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_addx2_program(M, N, block_M, block_N, dtype=T.float16):
    @T.prim_func
    def atomic_addx2(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 2):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 2
                T.atomic_addx2(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx2


def run_atomic_addx2(M, N, block_M, block_N, dtype=T.float16):
    kernel = atomic_addx2_program(M, N, block_M, block_N, dtype=dtype)

    import torch

    A = torch.randn(M, N, dtype=torch.float32).cuda().to(getattr(torch, dtype))
    B = torch.zeros(M, N, dtype=torch.float32).cuda().to(getattr(torch, dtype))
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 1, 2):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_add_mixed_dtype_program(N, src_dtype, dst_dtype):
    @T.prim_func
    def atomic_add(Src: T.Tensor((N,), src_dtype), Out: T.Tensor((N,), dst_dtype)):
        with T.Kernel(threads=1):
            frag = T.alloc_fragment((N,), src_dtype)
            for i in T.Parallel(N):
                frag[i] = Src[i]
            for i in T.Parallel(N):
                T.atomic_add(Out[i], frag[i])

    return atomic_add


def run_atomic_add_mixed_dtype(N, src_dtype, dst_dtype):
    kernel = atomic_add_mixed_dtype_program(N, src_dtype, dst_dtype)
    assert "AtomicAddx2" in kernel.get_kernel_source()

    src = torch.arange(1, N + 1, dtype=getattr(torch, src_dtype)).cuda()
    out = torch.zeros(N, dtype=getattr(torch, dst_dtype)).cuda()
    kernel(src, out)
    torch.testing.assert_close(out, src.to(getattr(torch, dst_dtype)), atol=1e-2, rtol=1e-2)


@tilelang.jit
def atomic_addx2_mixed_dtype_program(M, N, block_M, block_N, src_dtype, dst_dtype):
    @T.prim_func
    def atomic_addx2(A: T.Tensor((M, N), src_dtype), B: T.Tensor((M, N), dst_dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 2):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 2
                T.atomic_addx2(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx2


def run_atomic_addx2_mixed_dtype(M, N, block_M, block_N, src_dtype, dst_dtype):
    kernel = atomic_addx2_mixed_dtype_program(M, N, block_M, block_N, src_dtype, dst_dtype)
    assert "AtomicAddx2" in kernel.get_kernel_source()

    A = torch.randn(M, N, dtype=getattr(torch, src_dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dst_dtype)).cuda()
    ref_B = A.to(getattr(torch, dst_dtype))
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-2, rtol=1e-2)


@tilelang.jit
def atomic_different_memory_orders_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_different_orders(
        A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype), D: T.Tensor((M, N), dtype)
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    val = A[idx_i, idx_j]
                    T.atomic_add(B[idx_i, idx_j], val, memory_order="release")
                    T.atomic_max(C[idx_i, idx_j], val, memory_order="relaxed")
                    T.atomic_min(D[idx_i, idx_j], val, memory_order="relaxed")

    return atomic_different_orders


def run_atomic_different_memory_orders(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_different_memory_orders_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    C = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    D = torch.full((M, N), float("inf"), dtype=getattr(torch, dtype)).cuda()

    kernel(A, B, C, D)

    torch.testing.assert_close(B, A, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(C, torch.maximum(torch.zeros_like(A), A))
    torch.testing.assert_close(D, torch.minimum(torch.full_like(A, float("inf")), A))


@tilelang.jit
def atomic_addx4_program(M, N, block_M, block_N):
    @T.prim_func
    def atomic_addx4(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N // 4):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j * 4
                T.atomic_addx4(B[idx_i, idx_j], A[idx_i, idx_j])

    return atomic_addx4


def run_atomic_addx4(M, N, block_M, block_N):
    kernel = atomic_addx4_program(M, N, block_M, block_N)
    import torch

    A = torch.randn(M, N, dtype=torch.float32).cuda()
    B = torch.zeros(M, N, dtype=torch.float32).cuda()
    ref_B = B.clone()

    for i in range(M):
        for j in range(0, N - 3, 4):
            ref_B[i, j] += A[i, j]
            ref_B[i, j + 1] += A[i, j + 1]
            ref_B[i, j + 2] += A[i, j + 2]
            ref_B[i, j + 3] += A[i, j + 3]

    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_addx4_sliced_dst_program(N, M, dtype=T.float32):
    @T.prim_func
    def atomic_addx4_sliced_dst(
        idx: T.Tensor((N,), "int32"),
        val: T.Tensor((N, 4), dtype),
        dst: T.Tensor((M, 4), dtype),
    ):
        with T.Kernel(1, threads=N):
            t = T.get_thread_binding()
            T.atomic_addx4(dst[idx[t], 0:4], val[t, 0:4])

    return atomic_addx4_sliced_dst


def run_atomic_addx4_sliced_dst_compile(N, M, dtype=T.float32):
    kernel = atomic_addx4_sliced_dst_program(N, M, dtype=dtype)
    assert "AtomicAddx4" in kernel.get_kernel_source()


@tilelang.jit
def atomic_addx4_16bit_program(dtype, offset, nthreads):
    @T.prim_func
    def atomic_addx4(A: T.Tensor((16,), dtype), B: T.Tensor((16,), dtype)):
        with T.Kernel(1, threads=nthreads):
            T.atomic_addx4(B[offset], A[offset])

    return atomic_addx4


def run_atomic_addx4_16bit(dtype, offset, nthreads):
    kernel = atomic_addx4_16bit_program(dtype, offset, nthreads)
    source = kernel.get_kernel_source()
    assert "AtomicAddx4" in source

    torch_dtype = getattr(torch, str(dtype))
    A = torch.zeros(16, dtype=torch_dtype, device="cuda")
    B_init = torch.zeros(16, dtype=torch_dtype, device="cuda")
    A[offset : offset + 4] = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
    B_init[offset : offset + 4] = torch.tensor([10, 20, 30, 40], dtype=torch_dtype, device="cuda")
    B = B_init.clone()

    ref_B = B_init.float()
    ref_B[offset : offset + 4] += nthreads * A[offset : offset + 4].float()

    kernel(A, B)
    torch.testing.assert_close(B.float(), ref_B, atol=0, rtol=0)


@tilelang.jit
def atomic_return_prev_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_with_return_prev(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), old_vals: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    old_vals[idx_i, idx_j] = T.atomic_add(B[idx_i, idx_j], A[idx_i, idx_j], return_prev=True)

    return atomic_with_return_prev


def run_atomic_return_prev(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_return_prev_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.ones(M, N, dtype=getattr(torch, dtype)).cuda() * 5.0
    B = torch.ones(M, N, dtype=getattr(torch, dtype)).cuda() * 2.0
    old_vals = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()

    initial_B = B.clone()
    kernel(A, B, old_vals)

    torch.testing.assert_close(old_vals, initial_B, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(B, initial_B + A, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tma_atomic_add_program(out, explicit_swizzle=False):
    out: T.Tensor[(16, 16), T.float32]

    with T.Kernel(
        1,
    ):
        out_shared = T.alloc_shared((16, 16), dtype=T.float32)
        if explicit_swizzle:
            T.annotate_layout({out_shared: tilelang.layout.make_swizzled_layout(out_shared)})
        T.fill(out_shared, 1)
        for _ in range(16):
            T.atomic_add(out, out_shared, use_tma=True)


@tilelang.jit
def tma_atomic_add_uint64_program(out):
    out: T.Tensor[(16, 16), T.uint64]

    with T.Kernel(1):
        out_shared = T.alloc_shared((16, 16), dtype=T.uint64)
        T.fill(out_shared, 1)
        for _ in range(4):
            T.atomic_add(out, out_shared, use_tma=True)


@tilelang.jit
def tma_atomic_add_32b_swizzle_program(out):
    out: T.Tensor[(16, 24), T.float32]

    with T.Kernel(1):
        out_shared = T.alloc_shared((16, 24), dtype=T.float32)
        T.fill(out_shared, 1)
        T.atomic_add(out, out_shared, use_tma=True)


def tma_atomic_add_compile_program(dtype, shape=(16, 16), explicit_layout=False):
    @T.prim_func
    def main(out: T.Tensor(shape, dtype)):
        with T.Kernel(1):
            out_shared = T.alloc_shared(shape, dtype=dtype)
            if explicit_layout:
                T.annotate_layout({out_shared: tilelang.layout.make_wgmma_swizzled_layout(out_shared)})
            T.atomic_add(out, out_shared, use_tma=True)

    return main


def lower_tma_atomic_add(dtype, arch="sm_90", shape=(16, 16), explicit_layout=False):
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        return tilelang.lower(
            tma_atomic_add_compile_program(dtype, shape=shape, explicit_layout=explicit_layout),
            target=target,
        )


def get_tma_atomic_add_descriptor_args(artifact):
    for func in artifact.host_mod.functions.values():
        if func.attrs is None or "tma_descriptor_args" not in func.attrs:
            continue
        descriptors = list(func.attrs["tma_descriptor_args"].values())
        assert len(descriptors) == 1
        return list(descriptors[0])
    raise AssertionError("TMA descriptor metadata not found")


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_atomic_add():
    out = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
    tma_atomic_add_program(out)
    torch.testing.assert_close(out, torch.ones((16, 16), dtype=torch.float32, device="cuda") * 16)

    kernel = tma_atomic_add_program.compile(out=T.Tensor[(16, 16), T.float32])
    assert "tma_store_add" in kernel.get_kernel_source()
    assert "desc" in kernel.get_kernel_source()  # Ensure using cp.reduce.async.bulk.tensor

    kernel_with_explicit_swizzle = tma_atomic_add_program.compile(out=T.Tensor[(16, 16), T.float32], explicit_swizzle=True)
    # Ensure auto swizzled layout is applied
    assert kernel.get_kernel_source() == kernel_with_explicit_swizzle.get_kernel_source()


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_atomic_add_uint64_runtime():
    out = torch.zeros((16, 16), dtype=torch.uint64, device="cuda")
    tma_atomic_add_uint64_program(out)
    torch.testing.assert_close(out, torch.full_like(out, 4), rtol=0, atol=0)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_atomic_add_wide_linear_layout_runtime():
    rows, global_width, tile_width, col_offset = 8, 520, 512, 8

    @tilelang.jit
    def kernel(out):
        out: T.Tensor[(rows, global_width), T.uint64]

        with T.Kernel(1):
            out_shared = T.alloc_shared((rows, tile_width), dtype=T.uint64)
            T.fill(out_shared, 1)
            T.atomic_add(
                out[:, col_offset : col_offset + tile_width],
                out_shared,
                use_tma=True,
            )

    compiled = kernel.compile(out=T.Tensor[(rows, global_width), T.uint64])
    source = compiled.get_kernel_source()
    assert re.search(
        r"for \(int \w+ = 0; \w+ < 2; \+\+\w+\) \{\n\s*tl::tma_store_add\(",
        source,
    )

    out = torch.zeros((rows, global_width), dtype=torch.uint64, device="cuda")
    compiled(out)
    torch.cuda.synchronize()
    assert torch.all(out[:, :col_offset] == 0)
    assert torch.all(out[:, col_offset:] == 1)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_atomic_add_versioned_smem_slice_runtime():
    """A version slice of a wide linear buffer must stay contiguous under the
    inferred layout, or the second TMA box silently reads the other version."""
    rows, width = 8, 512

    @tilelang.jit
    def kernel(out):
        out: T.Tensor[(rows, width), T.uint64]

        with T.Kernel(1):
            smem = T.alloc_shared((2, rows, width), dtype=T.uint64)
            T.fill(smem, 7)  # decoy in version 0
            for i, j in T.Parallel(rows, width):
                smem[1, i, j] = 1
            T.atomic_add(out, smem[1, :, :], use_tma=True)

    compiled = kernel.compile(out=T.Tensor[(rows, width), T.uint64])
    source = compiled.get_kernel_source()
    assert re.search(
        r"for \(int \w+ = 0; \w+ < 2; \+\+\w+\) \{\n\s*tl::tma_store_add\(",
        source,
    )

    out = torch.zeros((rows, width), dtype=torch.uint64, device="cuda")
    compiled(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_atomic_add_32b_swizzle_runtime():
    out = torch.zeros((16, 24), dtype=torch.float32, device="cuda")
    tma_atomic_add_32b_swizzle_program(out)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.int16, T.int64, T.float64, T.float32x2])
def test_tma_atomic_add_rejects_unsupported_dtype(dtype):
    with pytest.raises(
        tvm.error.InternalError,
        match=rf"TMA atomic add does not support dtype {dtype}.*supported scalar dtypes",
    ):
        lower_tma_atomic_add(dtype)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32, T.int32, T.uint32, T.uint64])
def test_tma_atomic_add_accepts_ptx_dtype(dtype):
    artifact = lower_tma_atomic_add(dtype)
    assert "tma_store_add" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_tma_atomic_add_uint64_uses_linear_layout():
    artifact = lower_tma_atomic_add(T.uint64)
    descriptor_args = get_tma_atomic_add_descriptor_args(artifact)
    # The final four descriptor arguments are interleave, swizzle, L2
    # promotion, and OOB fill. uint64 intentionally uses no swizzle because
    # its GEMM K-inner layout is not representable by TensorMap.
    assert descriptor_args[-3].value == 0


@tilelang.testing.requires_cuda
def test_tma_atomic_add_splits_32b_swizzle_box():
    artifact = lower_tma_atomic_add(T.float32, shape=(16, 24))
    descriptor_args = get_tma_atomic_add_descriptor_args(artifact)
    assert descriptor_args[-3].value == 1
    assert re.search(
        r"for \(int \w+ = 0; \w+ < 3; \+\+\w+\) \{\n\s*tl::tma_store_add",
        artifact.kernel_source,
    )


@tilelang.testing.requires_cuda
def test_tma_atomic_add_rejects_pre_hopper_target():
    with pytest.raises(
        tvm.error.InternalError,
        match=r"TMA atomic add requires a CUDA target with TMA support \(SM90\+\)",
    ):
        lower_tma_atomic_add(T.float32, arch="sm_80")


@tilelang.testing.requires_cuda
def test_tma_atomic_add_rejects_unencodable_explicit_layout():
    with pytest.raises(
        tvm.error.InternalError,
        match=r"TMA atomic add cannot encode the shared layout.*not representable by a TensorMap descriptor",
    ):
        lower_tma_atomic_add(T.uint64, explicit_layout=True)


def run_atomic_add_auto_vectorized(K, M, N, block_M, block_N, dtype=T.float32):
    tilelang.disable_cache()
    kernel = atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    assert "AtomicAddx4" in kernel.get_kernel_source()


@tilelang.jit
def atomic_add_auto_vectorized_unit_test(vec_size: int, dtype=T.float32):
    @T.prim_func
    def atomic_addx2(A: T.Tensor((vec_size,), dtype)):
        with T.Kernel(threads=1):
            A_local = T.alloc_fragment((vec_size,), dtype)
            for i in T.Parallel(vec_size):
                T.atomic_add(A[i], A_local[i])

    return atomic_addx2


def run_atomic_add_auto_vectorized_unit_test(vec_size: int, dtype=T.float32):
    kernel = atomic_add_auto_vectorized_unit_test(vec_size, dtype)
    assert f"AtomicAddx{vec_size}" in kernel.get_kernel_source()


@tilelang.jit
def atomic_add_vectorized_memory_order_program(N, threads, memory_order, dtype=T.float16):
    @T.prim_func
    def atomic_add(X: T.Tensor((N,), dtype), Out: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=threads):
            T.atomic_add(Out[0:N], X[0:N], memory_order=memory_order)

    return atomic_add


def run_atomic_add_vectorized_memory_order(N, threads, memory_order, memory_order_id, dtype=T.float16):
    # A vectorized atomic add must carry the requested memory order, just like
    # the scalar lowering of the same call does. Cache disabled because the key
    # does not cover the native library by default, so a stale pre-fix kernel
    # source would otherwise be served here.
    tilelang.disable_cache()
    try:
        kernel = atomic_add_vectorized_memory_order_program(N, threads, memory_order, dtype=dtype)

        wide_calls = re.findall(r"AtomicAddx\d\([^;]*\);", kernel.get_kernel_source())
        assert wide_calls, f"expected a vectorized atomic add, got:\n{kernel.get_kernel_source()}"
        for call in wide_calls:
            assert call.endswith(f", {memory_order_id});"), f"{memory_order} dropped from vectorized atomic add: {call}"

        # Each order takes a separately spelled asm string in the device helpers,
        # so run the kernel to check this one actually computes the sum.
        torch_dtype = getattr(torch, dtype)
        X = torch.randn(N, dtype=torch_dtype).cuda()
        Out = torch.zeros(N, dtype=torch_dtype).cuda()
        kernel(X, Out)
        torch.testing.assert_close(Out, X, atol=1e-2, rtol=1e-2)
    finally:
        tilelang.enable_cache()


@tilelang.jit
def atomic_add_complicated_parallel_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                value = A_shared[i, j]
                T.atomic_add(B[bx * block_M + i, by * block_N + j], value)

    return atomic_add


def run_atomic_add_complicated_parallel(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_add_complicated_parallel_program(K, M, N, block_M, block_N, dtype=dtype)
    assert "float4 value" in kernel.get_kernel_source()
    assert "AtomicAddx4" in kernel.get_kernel_source()


def test_atomic_memory_order():
    run_atomic_memory_order(4, 64, 64, 16, 16)


@tilelang.testing.requires_cuda
def test_atomic_addx2_half():
    run_atomic_addx2(32, 64, 8, 16, dtype=T.float16)


def test_atomic_addx2_float():
    run_atomic_addx2(32, 64, 8, 16, dtype=T.float32)


@tilelang.testing.requires_cuda
def test_atomic_add_mixed_dtype_fp16():
    run_atomic_add_mixed_dtype(8, T.float32, T.float16)
    run_atomic_addx2_mixed_dtype(32, 64, 8, 16, T.float32, T.float16)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
def test_atomic_add_mixed_dtype_bf16():
    run_atomic_add_mixed_dtype(8, T.float32, T.bfloat16)
    run_atomic_addx2_mixed_dtype(32, 64, 8, 16, T.float32, T.bfloat16)


@tilelang.testing.requires_cuda
def test_atomic_different_memory_orders():
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float32)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.float16)
    run_atomic_different_memory_orders(32, 32, 8, 8, dtype=T.bfloat16)


# ids match cuda::memory_order. Each one is a separately spelled asm string in
# the device helpers, so all three ordered spellings need exercising.
_ORDERED_MEMORY_ORDERS = [("acquire", 2), ("release", 3), ("acq_rel", 4)]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("memory_order, memory_order_id", _ORDERED_MEMORY_ORDERS)
def test_atomic_add_vectorized_preserves_memory_order(memory_order, memory_order_id):
    run_atomic_add_vectorized_memory_order(64, 32, memory_order, memory_order_id, dtype=T.float16)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("memory_order, memory_order_id", _ORDERED_MEMORY_ORDERS)
def test_atomic_add_vectorized_preserves_memory_order_bf16(memory_order, memory_order_id):
    # bf16 takes a different device path depending on the target: below sm_90
    # there is no ordered bf16 atomic in any width, so AtomicAddx2 falls back to
    # fences around a relaxed add, while sm_90 uses the ordered instruction. A
    # given run covers whichever side the test GPU selects.
    run_atomic_add_vectorized_memory_order(64, 32, memory_order, memory_order_id, dtype=T.bfloat16)


def test_atomic_addx4():
    run_atomic_addx4(16, 64, 4, 4)


def test_atomic_addx4_sliced_dst_compile():
    run_atomic_addx4_sliced_dst_compile(32, 8)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
def test_atomic_addx4_16bit():
    for dtype in (T.float16, T.bfloat16):
        for offset in (0, 4):
            run_atomic_addx4_16bit(dtype, offset=offset, nthreads=2)


def test_atomic_return_prev():
    run_atomic_return_prev(32, 32, 8, 8)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_atomic_return_prev_16bit(dtype):
    run_atomic_return_prev(32, 32, 8, 8, dtype=dtype)


def test_atomic_add():
    run_atomic_add(8, 128, 128, 32, 32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_atomic_add_auto_vectorized():
    run_atomic_add_auto_vectorized(8, 128, 128, 32, 32, dtype=T.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_atomic_add_auto_vectorized_unit_test():
    run_atomic_add_auto_vectorized_unit_test(2, dtype=T.float32)
    run_atomic_add_auto_vectorized_unit_test(4, dtype=T.float32)
    run_atomic_add_auto_vectorized_unit_test(2, dtype=T.float16)
    run_atomic_add_auto_vectorized_unit_test(2, dtype=T.bfloat16)


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_atomic_add_complicated_parallel():
    run_atomic_add_complicated_parallel(8, 128, 128, 32, 32, dtype=T.float32)


# ======================= Tile-level atomic add =======================


@tilelang.jit
def tile_atomic_add_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_add(B[bx * block_M, by * block_N], A_shared)

    return atomic_add


def run_tile_atomic_add(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_add_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B + A.sum(dim=0)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_add_expr_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            T.atomic_add(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], 1.0)

    return atomic_add


def run_tile_atomic_add_expr(M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_add_expr_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A):
        for i in range(M):
            for j in range(N):
                A[i, j] += 1

    A = torch.zeros(M, N, dtype=torch.float32).cuda()
    ref_A = A.clone()
    ref_program(ref_A)
    kernel(A)
    torch.testing.assert_close(A, ref_A, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_add_scalar_program(dtype=T.float32):
    @T.prim_func
    def atomic_add(A: T.Tensor((1), dtype), B: T.Tensor((1), dtype)):
        with T.Kernel(
            1,
        ) as _:
            A_local = T.alloc_local([1], dtype)
            T.copy(A, A_local)
            T.clear(B)
            T.atomic_add(B, A_local)
            T.atomic_add(B, 1)

    return atomic_add


def run_tile_atomic_add_scalar(dtype=T.float32):
    kernel = tile_atomic_add_scalar_program(dtype=dtype)
    import torch

    def ref_program(A, B):
        B[0] = A[0] + 1

    A = torch.randn(1, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(1, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


def test_tile_atomic_add():
    run_tile_atomic_add(8, 128, 128, 32, 32)


def test_tile_atomic_add_expr():
    run_tile_atomic_add_expr(128, 128, 32, 32)


def test_tile_atomic_add_scalar():
    run_tile_atomic_add_scalar()


# ======================= Thread-level atomic max/min/load store =======================


@tilelang.jit
def atomic_max_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_max(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_max(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_max


def run_atomic_max(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_max_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = max(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_min_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_min(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            for i, j in T.Parallel(block_M, block_N):
                T.atomic_min(B[bx * block_M + i, by * block_N + j], A_shared[i, j])

    return atomic_min


def run_atomic_min(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_min_program(K, M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = min(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.full((M, N), float("inf"), dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def atomic_load_store_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_load_store(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                idx_i = bx * block_M + i
                idx_j = by * block_N + j
                if idx_i < M and idx_j < N:
                    val = T.atomic_load(A[idx_i, idx_j])
                    T.atomic_store(B[idx_i, idx_j], val)

    return atomic_load_store


def run_atomic_load_store(M, N, block_M, block_N, dtype=T.float32):
    kernel = atomic_load_store_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    A = torch.randn(M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.zeros(M, N, dtype=getattr(torch, dtype)).cuda()
    kernel(A, B)
    torch.testing.assert_close(B, A, atol=1e-3, rtol=1e-3)


def _build_atomic_load_with_memory_order(memory_order):
    @T.prim_func
    def kernel(source: T.Tensor((1,), T.int32), output: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            output[0] = T.atomic_load(source[0], memory_order=memory_order)

    return kernel


def _build_atomic_store_with_memory_order(memory_order):
    @T.prim_func
    def kernel(destination: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            T.atomic_store(destination[0], 1, memory_order=memory_order)

    return kernel


@pytest.mark.parametrize("memory_order", ["relaxed", "consume", "acquire", "seq_cst"])
def test_atomic_load_accepts_valid_memory_orders(memory_order):
    assert _build_atomic_load_with_memory_order(memory_order) is not None


@pytest.mark.parametrize("memory_order", ["release", "acq_rel", "invalid"])
def test_atomic_load_rejects_unsupported_memory_orders(memory_order):
    with pytest.raises(ValueError, match="atomic_load does not support memory_order"):
        _build_atomic_load_with_memory_order(memory_order)


@pytest.mark.parametrize("memory_order", ["relaxed", "release", "seq_cst"])
def test_atomic_store_accepts_valid_memory_orders(memory_order):
    assert _build_atomic_store_with_memory_order(memory_order) is not None


@pytest.mark.parametrize("memory_order", ["consume", "acquire", "acq_rel", "invalid"])
def test_atomic_store_rejects_unsupported_memory_orders(memory_order):
    with pytest.raises(ValueError, match="atomic_store does not support memory_order"):
        _build_atomic_store_with_memory_order(memory_order)


@tilelang.testing.requires_cuda
def test_atomic_or_codegen():
    @T.prim_func
    def atomic_or_kernel(A: T.Tensor((1,), T.int32), mask: T.int32):
        with T.Kernel(1, threads=32):
            T.atomic_or(A[0], mask, memory_order="release")

    kernel = tilelang.compile(atomic_or_kernel, out_idx=[0], target="cuda")
    source = kernel.get_kernel_source()
    assert "AtomicOr" in source


def test_atomic_max():
    run_atomic_max(4, 64, 64, 16, 16)


def test_atomic_min():
    run_atomic_min(4, 64, 64, 16, 16)


# ======== fp16/bf16 scalar max/min value-dtype conversion (issue #2758) ========


def run_atomic_max_scalar_literal(dtype):
    @tilelang.jit
    def wrapper():
        @T.prim_func
        def kernel(dst: T.Tensor((1,), dtype)):
            with T.Kernel(1, threads=1):
                T.atomic_max(dst[0], 42.0)  # python float literal is fp32

        return kernel

    kernel = wrapper()
    dst = torch.full((1,), -1e4, device="cuda", dtype=getattr(torch, dtype))
    kernel(dst)
    torch.testing.assert_close(dst, torch.full((1,), 42.0, device="cuda", dtype=getattr(torch, dtype)))


def run_atomic_min_scalar_literal(dtype):
    @tilelang.jit
    def wrapper():
        @T.prim_func
        def kernel(dst: T.Tensor((1,), dtype)):
            with T.Kernel(1, threads=1):
                T.atomic_min(dst[0], 42.0)  # python float literal is fp32

        return kernel

    kernel = wrapper()
    dst = torch.full((1,), 1e4, device="cuda", dtype=getattr(torch, dtype))
    kernel(dst)
    torch.testing.assert_close(dst, torch.full((1,), 42.0, device="cuda", dtype=getattr(torch, dtype)))


@tilelang.testing.requires_cuda
def test_atomic_max_scalar_fp16():
    run_atomic_max_scalar_literal("float16")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_atomic_max_scalar_bf16():
    run_atomic_max_scalar_literal("bfloat16")


@tilelang.testing.requires_cuda
def test_atomic_min_scalar_fp16():
    run_atomic_min_scalar_literal("float16")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_atomic_min_scalar_bf16():
    run_atomic_min_scalar_literal("bfloat16")


def test_atomic_load_store():
    run_atomic_load_store(64, 64, 16, 16)


# ======================= Tile-level atomic max/min =======================


@tilelang.jit
def tile_atomic_max_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def tile_atomic_max(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_max(B[bx * block_M, by * block_N], A_shared)

    return tile_atomic_max


def run_tile_atomic_max(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_max_program(K, M, N, block_M, block_N, dtype=dtype)

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = max(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.full((M, N), float("-inf"), dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_min_program(K, M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def tile_atomic_min(A: T.Tensor((K, M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), K, threads=32) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bz, bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], A_shared)

            T.atomic_min(B[bx * block_M, by * block_N], A_shared)

    return tile_atomic_min


def run_tile_atomic_min(K, M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_min_program(K, M, N, block_M, block_N, dtype=dtype)

    def ref_program(A, B):
        for k in range(K):
            for i in range(M):
                for j in range(N):
                    B[i, j] = min(B[i, j], A[k, i, j])

    A = torch.randn(K, M, N, dtype=getattr(torch, dtype)).cuda()
    B = torch.full((M, N), float("inf"), dtype=getattr(torch, dtype)).cuda()
    ref_B = B.clone()
    ref_program(A, ref_B)
    kernel(A, B)
    torch.testing.assert_close(B, ref_B, atol=1e-3, rtol=1e-3)


@tilelang.jit
def tile_atomic_max_expr_program(M, N, block_M, block_N, dtype=T.float32):
    @T.prim_func
    def atomic_max(A: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=32) as (bx, by):
            T.atomic_max(A[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N], 0.5)

    return atomic_max


def run_tile_atomic_max_expr(M, N, block_M, block_N, dtype=T.float32):
    kernel = tile_atomic_max_expr_program(M, N, block_M, block_N, dtype=dtype)
    import torch

    def ref_program(A):
        for i in range(M):
            for j in range(N):
                A[i, j] = max(A[i, j], 0.5)

    A = torch.randn(M, N, dtype=torch.float32).cuda()
    ref_A = A.clone()
    ref_program(ref_A)
    kernel(A)
    torch.testing.assert_close(A, ref_A, atol=1e-3, rtol=1e-3)


def test_tile_atomic_max():
    run_tile_atomic_max(8, 128, 128, 32, 32)


def test_tile_atomic_min():
    run_tile_atomic_min(8, 128, 128, 32, 32)


def test_tile_atomic_max_expr():
    run_tile_atomic_max_expr(128, 128, 32, 32)


def atomic_scalar_return_prev_program(op_name, val):
    atom = getattr(T, op_name)

    @T.prim_func
    def main(Dst: T.Tensor((1,), "float32"), Prev: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            Prev[0] = atom(Dst[0], val, return_prev=True)

    return main


def run_atomic_scalar_return_prev(op_name):
    # Pick val so the op actually changes dst (else a silent no-op would still
    # pass): max needs val > dst, min needs val < dst.
    val = 5.0 if op_name == "atomic_max" else 1.0
    kernel = tilelang.compile(atomic_scalar_return_prev_program(op_name, val))
    dst = torch.tensor([3.0], dtype=torch.float32).cuda()
    prev = torch.zeros(1, dtype=torch.float32).cuda()
    kernel(dst, prev)
    assert prev.item() == 3.0, f"{op_name} return_prev should be the old value"
    expected = max(3.0, val) if op_name == "atomic_max" else min(3.0, val)
    assert dst.item() == expected, f"{op_name} should still update Dst"


@tilelang.testing.requires_cuda
def test_atomic_scalar_return_prev():
    run_atomic_scalar_return_prev("atomic_max")
    run_atomic_scalar_return_prev("atomic_min")


def atomic_addx2_return_prev_program(dtype=T.float32):
    @T.prim_func
    def main(Dst: T.Tensor((2,), dtype), Val: T.Tensor((2,), dtype), Prev: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            Prev[0:2] = T.atomic_addx2(Dst[0:2], Val[0:2], return_prev=True)

    return main


def atomic_addx2_return_prev_let_bound_program(dtype=T.float32):
    @T.prim_func
    def main(Dst: T.Tensor((2,), dtype), Val: T.Tensor((2,), dtype), Prev: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            dst = Dst[0:2]
            val = Val[0:2]
            Prev[0:2] = T.atomic_addx2(dst, val, return_prev=True)

    return main


def atomic_addx2_return_prev_ramp_program(dtype=T.float32):
    @T.prim_func
    def main(Dst: T.Tensor((2,), dtype), Val: T.Tensor((2,), dtype), Prev: T.Tensor((2,), dtype)):
        with T.Kernel(1, threads=1):
            Prev[0:2] = T.atomic_addx2(
                Dst[T.Ramp(0, 1, 2)],
                Val[T.Ramp(0, 1, 2)],
                return_prev=True,
            )

    return main


def atomic_addx4_return_prev_program(dtype=T.float32):
    @T.prim_func
    def main(Dst: T.Tensor((4,), dtype), Val: T.Tensor((4,), dtype), Prev: T.Tensor((4,), dtype)):
        with T.Kernel(1, threads=1):
            Prev[0:4] = T.atomic_addx4(Dst[0:4], Val[0:4], return_prev=True)

    return main


def test_atomic_addx2_return_prev_accepts_sliced_destination():
    atomic_addx2_return_prev_program(T.float32)


def test_atomic_addx2_return_prev_accepts_let_bound_slice():
    atomic_addx2_return_prev_let_bound_program(T.float32)


def test_atomic_addx2_return_prev_accepts_ramp():
    atomic_addx2_return_prev_ramp_program(T.float32)


def test_atomic_addx2_return_prev():
    kernel = tilelang.compile(atomic_addx2_return_prev_let_bound_program(T.float32))
    assert "AtomicAddx2Ret" in kernel.get_kernel_source()
    dst = torch.tensor([1.0, 2.0], dtype=torch.float32).cuda()
    val = torch.tensor([10.0, 20.0], dtype=torch.float32).cuda()
    prev = torch.zeros(2, dtype=torch.float32).cuda()
    kernel(dst, val, prev)
    torch.testing.assert_close(prev, torch.tensor([1.0, 2.0], device="cuda"))
    torch.testing.assert_close(dst, torch.tensor([11.0, 22.0], device="cuda"))


def test_atomic_addx4_return_prev():
    kernel = tilelang.compile(atomic_addx4_return_prev_program(T.float32))
    assert "AtomicAddx4Ret" in kernel.get_kernel_source()
    dst = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).cuda()
    val = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32).cuda()
    prev = torch.zeros(4, dtype=torch.float32).cuda()
    kernel(dst, val, prev)
    torch.testing.assert_close(prev, torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"))
    torch.testing.assert_close(dst, torch.tensor([11.0, 22.0, 33.0, 44.0], device="cuda"))


def run_atomic_addx2_return_prev_16bit(dtype, torch_dtype):
    # The 16-bit packed 2-lane vector is stored as uint1 in codegen, while
    # AtomicAddx2Ret returns the native __half2 / __nv_bfloat162. The ret path
    # must bridge with tl::to_uint1 so the store LHS (uint1) matches; otherwise
    # nvcc fails with `no operator "=" ... uint1 = half2`.
    kernel = tilelang.compile(atomic_addx2_return_prev_program(dtype))
    src = kernel.get_kernel_source()
    assert "tl::to_uint1(AtomicAddx2Ret" in src, src
    dst = torch.tensor([1.0, 2.0], dtype=torch_dtype).cuda()
    val = torch.tensor([10.0, 20.0], dtype=torch_dtype).cuda()
    prev = torch.zeros(2, dtype=torch_dtype).cuda()
    kernel(dst, val, prev)
    torch.testing.assert_close(prev, torch.tensor([1.0, 2.0], dtype=torch_dtype, device="cuda"))
    torch.testing.assert_close(dst, torch.tensor([11.0, 22.0], dtype=torch_dtype, device="cuda"))


@tilelang.testing.requires_cuda
def test_atomic_addx2_return_prev_fp16():
    run_atomic_addx2_return_prev_16bit(T.float16, torch.float16)


@tilelang.testing.requires_cuda
def test_atomic_addx2_return_prev_bf16():
    run_atomic_addx2_return_prev_16bit(T.bfloat16, torch.bfloat16)


def run_atomic_addx4_return_prev_16bit(dtype, torch_dtype):
    # There is NO single-atomic fp16x4/bf16x4 add in hardware (PTX vector
    # atomic .v4 tops out at .f16x2 for 16-bit types). The x4 ret path realizes
    # the quad as two per-pair AtomicAddx2Ret calls and returns them packed as
    # a uint2 (a half4/bf16x4 is stored as uint2), matching the store LHS. This
    # is per-pair atomic, not whole-quad -- the same contract as the fp32-x4
    # scalar fallback.
    kernel = tilelang.compile(atomic_addx4_return_prev_program(dtype))
    src = kernel.get_kernel_source()
    assert "AtomicAddx4Ret" in src, src
    dst = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype).cuda()
    val = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch_dtype).cuda()
    prev = torch.zeros(4, dtype=torch_dtype).cuda()
    kernel(dst, val, prev)
    torch.testing.assert_close(prev, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"))
    torch.testing.assert_close(dst, torch.tensor([11.0, 22.0, 33.0, 44.0], dtype=torch_dtype, device="cuda"))


@tilelang.testing.requires_cuda
def test_atomic_addx4_return_prev_fp16():
    run_atomic_addx4_return_prev_16bit(T.float16, torch.float16)


@tilelang.testing.requires_cuda
def test_atomic_addx4_return_prev_bf16():
    run_atomic_addx4_return_prev_16bit(T.bfloat16, torch.bfloat16)


# ======================= Atomic return value materialization =======================


@tilelang.jit
def atomic_add_return_prev_compaction_program(n, cap, block, dtype=T.float32):
    @T.prim_func
    def compact_positive(
        x: T.Tensor((n,), dtype),
        out: T.Tensor((cap,), T.int32),
        counter: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            tx = T.get_thread_binding()
            i = bx * block + tx
            if i < n and x[i] > 0:
                # Side-effecting bind: the returned previous counter value is
                # used at TWO sites (bound check + store index). The lowering
                # must materialize the atomic exactly once; replaying it per
                # use site executes the atomic twice per element.
                pos = T.atomic_add(counter[0], 1, return_prev=True)
                if pos < cap:
                    out[pos] = i

    return compact_positive


def test_atomic_add_return_prev_materialized_once():
    n, cap, block = 4096, 4096, 256
    kernel = atomic_add_return_prev_compaction_program(n, cap, block)

    # Static check: the returned atomic must appear exactly once in the
    # generated source (regression: duplicated AtomicAddRet calls).
    assert kernel.get_kernel_source().count("AtomicAddRet") == 1

    # Functional check: with all-positive input the counter must equal n and
    # out must be a permutation of 0..n-1 (no unwritten slots).
    x = torch.ones(n, dtype=torch.float32).cuda()
    out = torch.full((cap,), -1, dtype=torch.int32).cuda()
    counter = torch.zeros(1, dtype=torch.int32).cuda()
    kernel(x, out, counter)
    torch.cuda.synchronize()

    assert counter.item() == n, f"atomic executed {counter.item()} times, expected {n}"
    assert (out < 0).sum().item() == 0, "unwritten slots: atomic return value was re-evaluated"
    assert torch.equal(out.sort().values.long(), torch.arange(n).cuda())


# ======================= Contended scalar atomic add =======================


@tilelang.jit
def atomic_add_contended_program(N, threads, dtype):
    @T.prim_func
    def atomic_add_contended(A: T.Tensor((N,), dtype), Out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=threads) as _:
            for i in T.Parallel(N):
                T.atomic_add(Out[0], A[i])

    return atomic_add_contended


def run_atomic_add_contended(N, threads, dtype):
    kernel = atomic_add_contended_program(N, threads, dtype)

    # Every thread targets the same cell, so a non-atomic read-modify-write
    # loses updates. Ones keep every partial sum exactly representable.
    a = torch.ones(N, dtype=getattr(torch, dtype)).cuda()
    out = torch.zeros(1, dtype=getattr(torch, dtype)).cuda()

    kernel(a, out)

    assert float(out[0]) == float(N), f"lost updates: got {float(out[0])}, expected {N}"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 0)
def test_atomic_add_contended_bf16():
    # bf16 is the dtype whose pre-SM80 add is a CAS loop; float32 contention is
    # already covered by the multi-block test_atomic_add above.
    run_atomic_add_contended(128, 128, T.bfloat16)


@tilelang.jit(target={"kind": "cuda", "arch": "sm_75"})
def atomic_add_bf16_sm75_program(N):
    @T.prim_func
    def atomic_add_bf16_sm75(A: T.Tensor((N,), T.bfloat16), B: T.Tensor((N,), T.bfloat16), Prev: T.Tensor((N,), T.bfloat16)):
        with T.Kernel(1, threads=32) as _:
            # Outside a parallel loop, so it stays scalar instead of packed.
            T.atomic_add(B[1], A[1])
            for i in T.Parallel(N // 2):
                T.atomic_addx2(B[i * 2], A[i * 2])
            for i in T.Parallel(N // 4):
                T.atomic_addx4(B[i * 4], A[i * 4])
            Prev[0] = T.atomic_add(B[0], A[0], return_prev=True)
            Prev[0:2] = T.atomic_addx2(B[0:2], A[0:2], return_prev=True)
            Prev[0:4] = T.atomic_addx4(B[0:4], A[0:4], return_prev=True)

    return atomic_add_bf16_sm75


@tilelang.testing.requires_cuda
def test_atomic_add_bf16_compiles_for_sm75():
    """The native __nv_bfloat16 atomicAdd and `atom.*.v2.bf16` are SM80+, and
    atomic.h reaches every generated kernel, so the pre-SM80 fallbacks must keep
    compiling. Pinned with an explicit sm_75 target because the tests above use
    the runner's native target and would miss this fallback on a modern GPU."""
    source = atomic_add_bf16_sm75_program(64).get_kernel_source()

    # Auto-vectorization rewrites scalar adds into packed ones, so check the
    # kernel above still reaches each fallback.
    for helper in (
        "AtomicAdd(",
        "AtomicAddx2(",
        "AtomicAddx4(",
        "AtomicAddRet(",
        "AtomicAddx2Ret(",
        "AtomicAddx4Ret(",
    ):
        assert helper in source, f"{helper} not exercised by the sm_75 kernel"


if __name__ == "__main__":
    tilelang.testing.main()
