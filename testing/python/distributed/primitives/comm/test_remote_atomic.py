"""Codegen tests for distributed remote atomic operations."""

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def atomic_addx2_dst_pe_program(M, dtype=T.bfloat16):
    @T.prim_func
    def atomic_addx2_dst_pe(A: T.Tensor((M,), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            rank = T.get_rank()
            if tx < M // 2:
                T.atomic_addx2(B[tx * 2], A[tx * 2], dst_pe=rank ^ 1)

    return atomic_addx2_dst_pe


def test_atomic_addx2_dst_pe_codegen():
    kernel = atomic_addx2_dst_pe_program(256, dtype=T.bfloat16)
    source = kernel.get_kernel_source()
    assert "get_remote_base_ptr" in source
    assert "AtomicAddx2(reinterpret_cast" in source


@tilelang.jit
def atomic_add_dst_pe_program(M, dtype=T.float32):
    @T.prim_func
    def atomic_add_dst_pe(A: T.Tensor((M,), dtype), B: T.Tensor((M,), dtype)):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            rank = T.get_rank()
            if tx < M:
                T.atomic_add(B[tx], A[tx], dst_pe=rank ^ 1)

    return atomic_add_dst_pe


def test_atomic_add_dst_pe_codegen():
    kernel = atomic_add_dst_pe_program(256, dtype=T.float32)
    source = kernel.get_kernel_source()
    assert "get_remote_base_ptr" in source
    assert "AtomicAdd(reinterpret_cast" in source


@tilelang.jit
def tma_atomic_add_dst_pe_program(M, N, dtype=T.bfloat16):
    @T.prim_func
    def tma_atomic_add_dst_pe(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, threads=128):
            rank = T.get_rank()
            A_shared = T.alloc_shared((M, N), dtype)
            T.copy(A, A_shared)
            T.atomic_add(B, A_shared, use_tma=True, dst_pe=rank)

    return tma_atomic_add_dst_pe


def test_tma_atomic_add_dst_pe_codegen():
    kernel = tma_atomic_add_dst_pe_program(64, 128, dtype=T.bfloat16)
    source = kernel.get_kernel_source()
    host_source = kernel.get_host_source()
    assert "B_desc_pe0" in source
    assert "B_desc_pe7" in source
    assert "__tvm_tensormap_create_remote_tiled" in host_source


if __name__ == "__main__":
    tilelang.testing.main()
