import argparse
import math
import time

import torch

import tilelang as tl
import tilelang.language as T


def make_ptr_table(tensors):
    assert tensors, "pointer table requires at least one tensor"
    device = tensors[0].device
    return torch.tensor([tensor.data_ptr() for tensor in tensors], device=device, dtype=torch.int64)


def torch_grouped_gemm_ptr(a_list, b_list):
    assert len(a_list) == len(b_list), "A/B group count mismatch"
    outputs = []
    for a, b in zip(a_list, b_list):
        assert a.shape[1] == b.shape[0], "incompatible GEMM shapes"
        outputs.append(torch.matmul(a, b))
    return outputs


def grouped_gemm_ptr(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    # Keep per-group tensors separate and pass them via pointer tables.
    # We currently use a common max_M storage shape per group because
    # ptr-backed tensors with runtime-varying shapes are not stable enough yet.
    # Multi-stage software pipelining on ptr-backed tensors is not correct yet.
    # Keep a single-stage pipeline so the ptr path can still use T.copy lowering.
    copy_num_stages = 1
    batch_count = len(batch_sizes_list)
    max_M = max(batch_sizes_list)
    batch_tile_offsets = [0]
    for size in batch_sizes_list[:-1]:
        batch_tile_offsets.append(batch_tile_offsets[-1] + math.ceil(size / block_M))
    total_m_blocks = sum(math.ceil(size / block_M) for size in batch_sizes_list)
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        A_ptrs: T.Tensor([batch_count], T.ptr),
        B_ptrs: T.Tensor([batch_count], T.ptr),
        C_ptrs: T.Tensor([batch_count], T.ptr),
        batch_tile_offsets: T.Tensor([batch_count], T.int32),
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
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
            A = T.make_tensor(A_ptrs[cur_batch_idx], (max_M, K), dtype)
            B = T.make_tensor(B_ptrs[cur_batch_idx], (K, N), dtype)
            C = T.make_tensor(C_ptrs[cur_batch_idx], (max_M, N), dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=copy_num_stages):
                T.copy(A[m_start, ko * block_K], A_shared)
                T.copy(B[ko * block_K, by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[m_start, by * block_N])

    return kernel


def construct_inputs(batch_sizes_list, K, N, block_M, device, dtype):
    max_M = max(batch_sizes_list)
    batch_tile_offsets_list = [0]
    for size in batch_sizes_list[:-1]:
        batch_tile_offsets_list.append(batch_tile_offsets_list[-1] + math.ceil(size / block_M))
    # Each group owns an independent padded tensor; nothing is concatenated.
    a_list = [torch.zeros(max_M, K, device=device, dtype=dtype) for _ in batch_sizes_list]
    b_list = [torch.randn(K, N, device=device, dtype=dtype) for _ in batch_sizes_list]
    c_list = [torch.empty(max_M, N, device=device, dtype=dtype) for _ in batch_sizes_list]
    for a, size in zip(a_list, batch_sizes_list):
        a[:size].copy_(torch.randn(size, K, device=device, dtype=dtype))
    a_ptrs = make_ptr_table(a_list)
    b_ptrs = make_ptr_table(b_list)
    c_ptrs = make_ptr_table(c_list)
    batch_tile_offsets = torch.tensor(batch_tile_offsets_list, device=device, dtype=torch.int32)
    return a_list, b_list, c_list, a_ptrs, b_ptrs, c_ptrs, batch_tile_offsets


def verify_outputs(outputs, refs, batch_sizes_list, atol=1e-2, rtol=1e-2):
    for idx, (out, ref, batch_size) in enumerate(zip(outputs, refs, batch_sizes_list)):
        try:
            torch.testing.assert_close(out[:batch_size], ref, atol=atol, rtol=rtol)
        except AssertionError as err:
            raise AssertionError(f"group {idx}: {err}") from err


def benchmark(kernel, inputs, warmup=50, rep=100):
    for _ in range(warmup):
        kernel(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel(*inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def run_tilelang_grouped_gemm_ptr(
    batch_sizes_list,
    K,
    N,
    block_M,
    block_N,
    block_K,
    num_stages=2,
    threads=128,
    backend="tvm_ffi",
    profile=False,
):
    device = torch.device("cuda")
    dtype = torch.float16
    program = grouped_gemm_ptr(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages, threads)
    kernel = tl.compile(
        program,
        execution_backend=backend,
        pass_configs={"tl.disable_warp_specialized": True},
    )
    a_list, b_list, c_list, a_ptrs, b_ptrs, c_ptrs, batch_tile_offsets = construct_inputs(batch_sizes_list, K, N, block_M, device, dtype)
    refs = torch_grouped_gemm_ptr([a[:size] for a, size in zip(a_list, batch_sizes_list)], b_list)

    kernel(a_ptrs, b_ptrs, c_ptrs, batch_tile_offsets)
    verify_outputs(c_list, refs, batch_sizes_list)
    print("✅ TileLang ptr-grouped-gemm matches PyTorch")

    if profile:
        latency = benchmark(kernel, (a_ptrs, b_ptrs, c_ptrs, batch_tile_offsets))
        total_flops = sum(size * K * N * 2 for size in batch_sizes_list)
        print(f"Latency: {latency:.4f} ms")
        print(f"TFlops: {total_flops / (latency * 1e9):.4f}")


def test_grouped_gemm_ptr():
    run_tilelang_grouped_gemm_ptr([16, 33, 64], 128, 96, 32, 32, 32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="64,128,256", help="comma-separated per-group M sizes")
    parser.add_argument("--K", type=int, default=4096, help="reduce dim")
    parser.add_argument("--N", type=int, default=4096, help="output dim")
    parser.add_argument("--backend", type=str, default="tvm_ffi", choices=["tvm_ffi", "cython"], help="execution backend")
    parser.add_argument("--profile", action="store_true", help="benchmark the kernel")
    args = parser.parse_args()

    batch_sizes_list = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    block_M = 64
    block_N = 128
    block_K = 64
    num_stages = 1
    threads = 256

    t0 = time.time()
    run_tilelang_grouped_gemm_ptr(
        batch_sizes_list,
        args.K,
        args.N,
        block_M,
        block_N,
        block_K,
        num_stages=num_stages,
        threads=threads,
        backend=args.backend,
        profile=args.profile,
    )
    print(f"End-to-end: {time.time() - t0:.3f} s")
