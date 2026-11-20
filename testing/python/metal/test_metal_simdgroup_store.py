"""Test Metal simdgroup register GEMM with direct simdgroup_store to device memory.

These tests verify the simdgroup register accumulation path where C is allocated
in metal.simdgroup scope. This eliminates C_simd load/store round-trips through
shared memory on each K iteration. The final T.copy(C_local, C[...]) is lowered
to simdgroup_store directly to device memory via LowerSIMDGroupStore.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import torch
import pytest


def _make_simdgroup_gemm_func(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


matmul_simdgroup = tilelang.jit(_make_simdgroup_gemm_func)


def assert_simdgroup_store_correctness(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, atol=1e-2):
    kernel = matmul_simdgroup(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    torch_accum_dtype = accum_dtype.as_torch()
    a = torch.randn(M, K, dtype=torch_dtype, device="mps")
    b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_accum_dtype, device="mps")

    kernel(a, b, c)

    ref = a.to(torch_accum_dtype) @ b.to(torch_accum_dtype)
    assert torch.allclose(ref, c, atol=atol), (
        f"Result mismatch for M={M}, N={N}, K={K}, "
        f"block=({block_M},{block_N},{block_K}), dtype={dtype}\n"
        f"max diff: {(ref - c).abs().max().item()}"
    )


def assert_simdgroup_store_codegen(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    func = _make_simdgroup_gemm_func(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(func, target="metal")

    src = artifact.kernel_source
    assert src is not None
    assert "[[kernel" in src or "kernel void" in src
    assert "simdgroup_multiply_accumulate" in src
    assert "make_filled_simdgroup_matrix" in src
    assert "MetalPerformancePrimitives" not in src

    assert "simdgroup_float8x8" in src or "simdgroup_half8x8" in src, "Expected simdgroup_float8x8 or simdgroup_half8x8 for C accumulator"

    store_to_device = src.count("simdgroup_store(C_local")
    assert store_to_device > 0, "Expected simdgroup_store of C_local to device memory"

    load_c_from_shared = [line for line in src.split("\n") if "simdgroup_load" in line and "C_local" in line]
    assert len(load_c_from_shared) == 0, f"C_local should not be loaded from shared memory, but found: {load_c_from_shared}"


# --- Codegen tests (cross-platform) ---


def test_codegen_square_small():
    assert_simdgroup_store_codegen(64, 64, 64, 16, 16, 16)


def test_codegen_square_large():
    assert_simdgroup_store_codegen(128, 128, 128, 32, 32, 32)


def test_codegen_non_square():
    assert_simdgroup_store_codegen(128, 128, 128, 32, 64, 16)


def test_codegen_float32_accum():
    assert_simdgroup_store_codegen(64, 64, 64, 16, 16, 16, dtype=T.float32, accum_dtype=T.float32)


# --- Correctness tests (require Metal hardware) ---


@tilelang.testing.requires_metal
def test_correctness_16x16x16():
    assert_simdgroup_store_correctness(128, 128, 128, 16, 16, 16)


@tilelang.testing.requires_metal
def test_correctness_32x32x32():
    assert_simdgroup_store_correctness(128, 128, 128, 32, 32, 32)


@tilelang.testing.requires_metal
def test_correctness_non_square_block():
    assert_simdgroup_store_correctness(128, 128, 128, 32, 64, 16)


@tilelang.testing.requires_metal
def test_correctness_64x64x32():
    assert_simdgroup_store_correctness(128, 128, 128, 64, 64, 32)


@tilelang.testing.requires_metal
def test_correctness_large_matrix():
    assert_simdgroup_store_correctness(1024, 1024, 1024, 32, 32, 32, atol=1.0)


@tilelang.testing.requires_metal
def test_correctness_non_square_matrix():
    assert_simdgroup_store_correctness(256, 512, 128, 32, 32, 16)


def _lower_copy_bounds(func):
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        return tilelang.lower(func, target="metal")


def _make_copy_bounds_kernel(m, n, block, threads, dtype, shared=False, staged=False):
    @T.prim_func
    def kernel(output: T.Tensor((m, n), dtype)):
        with T.Kernel(T.ceildiv(n, block), T.ceildiv(m, block), threads=threads) as (bx, by):
            tile = T.alloc_local((block, block), dtype, scope="metal.simdgroup")
            T.fill(tile, T.cast(3, dtype))
            if staged:
                scratch = T.alloc_shared((block, block), dtype)
                T.copy(tile, scratch)
                T.copy(scratch, output[by * block, bx * block])
            elif shared:
                destination = T.alloc_shared((m, n), dtype)
                T.copy(tile, destination[by * block, bx * block])
                T.copy(destination, output)
            else:
                T.copy(tile, output[by * block, bx * block])

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "m,n,block,threads,dtype,shared",
    [
        (8, 1, 8, 32, "float32", False),  # #3907: seven guard elements overwritten
        (8, 1, 8, 32, "float16", False),
        (1, 8, 8, 32, "float32", False),  # row tail
        (16, 17, 16, 128, "float32", False),  # multi-group, block-dependent offset
        (16, 18, 16, 128, "float32", False),  # not specific to odd strides
        (8, 1, 8, 32, "float32", True),  # shared destination uses the same store
    ],
)
def test_simdgroup_copy_rejects_destination_tail(m, n, block, threads, dtype, shared):
    with pytest.raises(ValueError, match="Metal SIMD-group copy:.*in-bounds"):
        _lower_copy_bounds(_make_copy_bounds_kernel(m, n, block, threads, dtype, shared))


@tilelang.testing.requires_metal
def test_simdgroup_copy_rejects_short_destination_region():
    @T.prim_func
    def kernel(output: T.Tensor((16, 16), T.float32)):
        with T.Kernel(1, threads=32):
            tile = T.alloc_local((8, 8), T.float32, scope="metal.simdgroup")
            T.fill(tile, T.float32(3))
            T.copy(tile, output[0:8, 0:4])

    with pytest.raises(ValueError, match="Metal SIMD-group copy:.*in-bounds"):
        _lower_copy_bounds(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize("guarded", [False, True])
def test_simdgroup_copy_symbolic_destination(guarded):
    @T.prim_func
    def kernel(output: T.Tensor((8, 16), T.float32), offset: T.int32):
        with T.Kernel(1, threads=32):
            tile = T.alloc_local((8, 8), T.float32, scope="metal.simdgroup")
            T.fill(tile, T.float32(3))
            if guarded:
                if offset >= 0 and offset <= 8:
                    T.copy(tile, output[0, offset])
            else:
                T.copy(tile, output[0, offset])

    if guarded:
        _lower_copy_bounds(kernel)
    else:
        with pytest.raises(ValueError, match="Metal SIMD-group copy:.*in-bounds"):
            _lower_copy_bounds(kernel)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "m,n,block,threads,dtype,shared,staged",
    [
        (8, 8, 8, 32, "float32", False, False),
        (8, 8, 8, 32, "float16", False, False),
        (16, 32, 16, 128, "float32", False, False),
        (8, 8, 8, 32, "float32", True, False),
        (8, 1, 8, 32, "float32", False, True),  # supported tail workaround
    ],
)
def test_simdgroup_copy_preserves_destination_guard(m, n, block, threads, dtype, shared, staged):
    func = _make_copy_bounds_kernel(m, n, block, threads, dtype, shared, staged)
    kernel = tilelang.compile(func, out_idx=[], target="metal", target_host="c", execution_backend="tvm_ffi")
    backing = torch.full((m * n + 64,), -9, dtype=getattr(torch, dtype), device="mps")
    output = backing[: m * n].view(m, n)
    kernel(output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), torch.full((m, n), 3, dtype=getattr(torch, dtype)), rtol=0, atol=0)
    assert torch.all(backing[m * n :].cpu() == -9).item()


def _make_layout_store_kernel(layout_kind, column):
    layouts = {
        "transpose": tilelang.Layout((8, 16), lambda i, j: (j, i)),
        "identity": tilelang.Layout((8, 16), lambda i, j: (i, j)),
        "flatten": tilelang.Layout((8, 16), lambda i, j: i * 16 + j),
        "reshape": tilelang.Layout((8, 16), lambda i, j: ((i * 16 + j) // 32, (i * 16 + j) % 32)),
    }
    layout = layouts[layout_kind]

    @T.prim_func
    def kernel(output: T.Tensor((8, 16), T.float32)):
        with T.Kernel(1, threads=32):
            tile = T.alloc_local((8, 8), T.float32, scope="metal.simdgroup")
            shared = T.alloc_shared((8, 16), T.float32)
            T.annotate_layout({shared: layout})
            T.clear(shared)
            T.fill(tile, T.float32(3))
            T.copy(tile, shared[0, column])
            T.copy(shared, output)

    return kernel


@tilelang.testing.requires_metal
@pytest.mark.parametrize("column", [0, 8])
def test_simdgroup_copy_rejects_transposed_destination(column):
    # Column 8 writes beyond the allocation; column 0 stays in the allocation
    # but the row-major store still disagrees with the transposed mapping.
    with pytest.raises(ValueError, match="Metal SIMD-group copy:.*row-major"):
        _lower_copy_bounds(_make_layout_store_kernel("transpose", column))


@tilelang.testing.requires_metal
@pytest.mark.parametrize("layout_kind", ["identity", "flatten", "reshape"])
def test_simdgroup_copy_accepts_equivalent_destination_layout(layout_kind):
    func = _make_layout_store_kernel(layout_kind, 8)
    kernel = tilelang.compile(func, out_idx=[], target="metal", target_host="c", execution_backend="tvm_ffi")
    backing = torch.full((8 * 16 + 64,), -9.0, dtype=torch.float32, device="mps")
    output = backing[: 8 * 16].view(8, 16)
    kernel(output)
    torch.mps.synchronize()
    expected = torch.zeros((8, 16))
    expected[:, 8:] = 3
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.all(backing[8 * 16 :].cpu() == -9).item()


if __name__ == "__main__":
    if torch.mps.is_available():
        tilelang.testing.main()
