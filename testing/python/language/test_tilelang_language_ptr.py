import math
import torch
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T
from tilelang.utils import map_torch_type


def matmul_test(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        a_ptr: T.ptr,
        b_ptr: T.ptr,
        c_ptr: T.ptr,
        m: T.int32,
        n: T.int32,
        k: T.int32,
    ):
        A = T.make_tensor(a_ptr, (m, k), dtype)
        B = T.make_tensor(b_ptr, (k, n), dtype)
        C = T.make_tensor(c_ptr, (m, n), accum_dtype)

        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def pointer_table_copy_test(N, dtype=T.float16):
    @T.prim_func
    def main(
        src_ptrs: T.Tensor([1], T.ptr),
        out: T.Tensor([N], dtype),
    ):
        with T.Kernel(1, threads=1) as _:
            Src = T.make_tensor(src_ptrs[0], (N,), dtype)
            for i in T.serial(N):
                out[i] = Src[i]

    return main


def pointer_table_multi_copy_test(G, N, dtype=T.float16):
    @T.prim_func
    def main(
        src_ptrs: T.Tensor([G], T.ptr),
        out: T.Tensor([G, N], dtype),
    ):
        with T.Kernel(G, threads=1) as bx:
            Src = T.make_tensor(src_ptrs[bx], (N,), dtype)
            for i in T.serial(N):
                out[bx, i] = Src[i]

    return main


def pointer_table_grouped_matmul_test(batch_sizes_list, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    batch_count = len(batch_sizes_list)
    max_M = max(batch_sizes_list)
    total_m_blocks = sum(math.ceil(size / block_M) for size in batch_sizes_list)

    @T.prim_func
    def main(
        a_ptrs: T.Tensor([batch_count], T.ptr),
        b_ptrs: T.Tensor([batch_count], T.ptr),
        c_ptrs: T.Tensor([batch_count], T.ptr),
        batch_tile_offsets: T.Tensor([batch_count], T.int32),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=32) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_tile_offset = T.alloc_var(dtype=T.int32)

            cur_batch_idx = 0
            cur_tile_offset = 0
            for i in range(batch_count):
                in_cur_batch_idx = bx >= batch_tile_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)
                cur_tile_offset = T.if_then_else(in_cur_batch_idx, batch_tile_offsets[i], cur_tile_offset)

            m_start = (bx - cur_tile_offset) * block_M
            A = T.make_tensor(a_ptrs[cur_batch_idx], (max_M, K), dtype)
            B = T.make_tensor(b_ptrs[cur_batch_idx], (K, N), dtype)
            C = T.make_tensor(c_ptrs[cur_batch_idx], (max_M, N), dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[m_start, ko * block_K], A_shared)
                T.copy(B[ko * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[m_start, by * block_N])

    return main


def run_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    program = matmul_test(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    cython_jit_kernel = tl.compile(program, execution_backend="cython")
    ffi_jit_kernel = tl.compile(program, execution_backend="tvm_ffi")

    def ref_program(a, b):
        return (a @ b.T).to(torch.float32)

    a = torch.randn(M, K, device="cuda", dtype=map_torch_type(dtype))
    b = torch.randn(N, K, device="cuda", dtype=map_torch_type(dtype))
    ffi_c = torch.zeros(M, N, device="cuda", dtype=map_torch_type(accum_dtype))
    cython_c = torch.zeros(M, N, device="cuda", dtype=map_torch_type(accum_dtype))

    ffi_jit_kernel(a, b, ffi_c, M, N, K)
    cython_jit_kernel(a.data_ptr(), b.data_ptr(), cython_c.data_ptr(), M, N, K)
    torch.testing.assert_close(ffi_c, ref_program(a, b), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cython_c, ffi_c, atol=1e-2, rtol=1e-2)


def run_pointer_table_copy(N, dtype=T.float16):
    program = pointer_table_copy_test(N, dtype)
    cython_jit_kernel = tl.compile(program, execution_backend="cython")
    ffi_jit_kernel = tl.compile(program, execution_backend="tvm_ffi")
    src = torch.randn(N, device="cuda", dtype=map_torch_type(dtype))
    src_ptrs = torch.tensor([src.data_ptr()], device="cuda", dtype=torch.int64)
    ffi_out = torch.empty(N, device="cuda", dtype=map_torch_type(dtype))
    cython_out = torch.empty(N, device="cuda", dtype=map_torch_type(dtype))

    ffi_jit_kernel(src_ptrs, ffi_out)
    cython_jit_kernel(src_ptrs, cython_out)

    torch.testing.assert_close(ffi_out, src)
    torch.testing.assert_close(cython_out, src)
    torch.testing.assert_close(cython_out, ffi_out)


def run_pointer_table_multi_copy(G, N, dtype=T.float16):
    program = pointer_table_multi_copy_test(G, N, dtype)
    cython_jit_kernel = tl.compile(program, execution_backend="cython")
    ffi_jit_kernel = tl.compile(program, execution_backend="tvm_ffi")
    srcs = [torch.randn(N, device="cuda", dtype=map_torch_type(dtype)) for _ in range(G)]
    src_ptrs = torch.tensor([src.data_ptr() for src in srcs], device="cuda", dtype=torch.int64)
    ref = torch.stack(srcs, dim=0)
    ffi_out = torch.empty((G, N), device="cuda", dtype=map_torch_type(dtype))
    cython_out = torch.empty((G, N), device="cuda", dtype=map_torch_type(dtype))

    ffi_jit_kernel(src_ptrs, ffi_out)
    cython_jit_kernel(src_ptrs, cython_out)

    torch.testing.assert_close(ffi_out, ref)
    torch.testing.assert_close(cython_out, ref)
    torch.testing.assert_close(cython_out, ffi_out)


def run_pointer_table_grouped_matmul(batch_sizes_list, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    program = pointer_table_grouped_matmul_test(batch_sizes_list, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    compile_kwargs = {"pass_configs": {"tl.disable_warp_specialized": True}}
    cython_jit_kernel = tl.compile(program, execution_backend="cython", **compile_kwargs)
    ffi_jit_kernel = tl.compile(program, execution_backend="tvm_ffi", **compile_kwargs)

    device = "cuda"
    torch_dtype = map_torch_type(dtype)
    torch_accum_dtype = map_torch_type(accum_dtype)
    max_M = max(batch_sizes_list)
    batch_tile_offsets = [0]
    for size in batch_sizes_list[:-1]:
        batch_tile_offsets.append(batch_tile_offsets[-1] + math.ceil(size / block_M))
    batch_tile_offsets_t = torch.tensor(batch_tile_offsets, device=device, dtype=torch.int32)

    a_exact = [torch.randn(size, K, device=device, dtype=torch_dtype) for size in batch_sizes_list]
    b_list = [torch.randn(K, N, device=device, dtype=torch_dtype) for _ in batch_sizes_list]
    ref = [(a @ b).to(torch_accum_dtype) for a, b in zip(a_exact, b_list)]

    def build_backend_buffers():
        a_list = [torch.zeros(max_M, K, device=device, dtype=torch_dtype) for _ in batch_sizes_list]
        c_list = [torch.empty(max_M, N, device=device, dtype=torch_dtype) for _ in batch_sizes_list]
        for buf, src in zip(a_list, a_exact):
            buf[: src.shape[0]].copy_(src)
        return (
            a_list,
            c_list,
            torch.tensor([buf.data_ptr() for buf in a_list], device=device, dtype=torch.int64),
            torch.tensor([buf.data_ptr() for buf in b_list], device=device, dtype=torch.int64),
            torch.tensor([buf.data_ptr() for buf in c_list], device=device, dtype=torch.int64),
        )

    ffi_a_list, ffi_c_list, ffi_a_ptrs, ffi_b_ptrs, ffi_c_ptrs = build_backend_buffers()
    cython_a_list, cython_c_list, cython_a_ptrs, cython_b_ptrs, cython_c_ptrs = build_backend_buffers()

    ffi_jit_kernel(ffi_a_ptrs, ffi_b_ptrs, ffi_c_ptrs, batch_tile_offsets_t)
    cython_jit_kernel(cython_a_ptrs, cython_b_ptrs, cython_c_ptrs, batch_tile_offsets_t)

    for out, expected, size in zip(ffi_c_list, ref, batch_sizes_list):
        torch.testing.assert_close(out[:size].to(torch_accum_dtype), expected, atol=1e-2, rtol=1e-2)
    for out, expected, size in zip(cython_c_list, ref, batch_sizes_list):
        torch.testing.assert_close(out[:size].to(torch_accum_dtype), expected, atol=1e-2, rtol=1e-2)


def test_matmul():
    run_matmul(256, 256, 256, 64, 64, 32)


def test_pointer_table_annotation_lowers_to_int64_buffer():
    program = pointer_table_multi_copy_test(4, 8)
    src_ptrs = program.buffer_map[program.params[0]]

    assert src_ptrs.dtype == "int64"
    assert [int(dim) for dim in src_ptrs.shape] == [4]


def test_pointer_table_copy():
    run_pointer_table_copy(64)


def test_pointer_table_multi_copy():
    run_pointer_table_multi_copy(2, 64)


@tilelang.testing.requires_cuda
def test_pointer_table_grouped_matmul():
    run_pointer_table_grouped_matmul([8, 12, 17], 32, 32, 16, 16, 16)


if __name__ == "__main__":
    tilelang.testing.main()
