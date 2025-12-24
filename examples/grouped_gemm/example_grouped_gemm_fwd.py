import torch
import argparse
import tilelang
import tilelang.language as T
import math


def torch_gmm(a, b, batch_sizes, batch_offsets_tensor, trans_b=False):
    """
    Perform grouped matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Input tensor of shape (N, K).
        b (torch.Tensor): Input tensor of shape (G, K, M).
        batch_sizes (torch.Tensor): 1D tensor containing the sizes of each group.

    Returns:
        torch.Tensor: Resulting tensor after grouped matrix multiplication.
    """
    assert a.shape[0] == sum(batch_sizes), "Sum of batch_sizes must equal the first dimension of a"
    assert b.shape[0] == len(batch_sizes), "The first dimension of b must match the length of batch_sizes"

    # Initialize output tensor
    output = torch.empty((sum(batch_sizes), b.shape[2]), device=a.device, dtype=a.dtype)

    # Perform grouped GEMM
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end]
        part_b = b[i].transpose(0, 1) if trans_b else b[i]
        part_out = torch.mm(part_a, part_b)
        output[start:end] = part_out
        start = end

    return output


@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            m_start_padded = bx * block_M

            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


def construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] + math.ceil((batch_sizes_list[i]) / padding_M) * padding_M)
    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, M, device=device, dtype=dtype)
    C = torch.empty(batch_sum, M, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    # print(batch_sizes_tensor)
    # print(batch_offsets_tensor)
    # print(batch_padded_offsets_tensor)
    return A, B, C, batch_sizes, batch_offsets, batch_padded_offsets


def run_tilelang_grouped_gemm(batch_sizes_list, K, M, block_M, block_N, block_K, trans_b, num_stages=2, threads=128, profile=False):
    padding_M = block_M
    batch_sum = sum(batch_sizes_list)
    kernel = grouped_gemm(tuple(batch_sizes_list), K, M, block_M, block_N, block_K, num_stages, threads)
    # print(kernel.get_kernel_source())

    device = torch.device("cuda")
    dtype = torch.float16

    A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype)
    out = kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
    ref_output = torch_gmm(A, B, batch_sizes, batch_offsets, trans_b)
    # print(out)
    # print(ref_output)
    if torch.allclose(out, ref_output, rtol=0.01, atol=0.01):
        print("✅ Tilelang and Torch match")
    else:
        print("❌ Tilelang and Torch mismatch")

    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(warmup=500, input_tensors=[A, B, batch_sizes, batch_offsets, batch_padded_offsets])
        print(f"Latency: {latency} ms")
        print(f"TFlops: {batch_sum * K * M * 2 / latency * 1e-9} TFlops")


def test_grouped_gemm():
    run_tilelang_grouped_gemm([64], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([64, 128, 256], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([63], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([100, 200, 300, 400], 8192, 8192, 64, 64, 64, False)
    run_tilelang_grouped_gemm([63, 77, 111, 280], 8192, 8192, 64, 64, 64, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=str, default="64, 128", help="comma-separated batch sizes")
    parser.add_argument("--K", type=int, default=8192, help="reduce dim")
    parser.add_argument("--M", type=int, default=8192, help="output dim")
    parser.add_argument("--trans_b", action="store_true", help="transpose B")
    parser.add_argument("--profile", action="store_true", help="profile")
    args = parser.parse_args()

    batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
    K, M, trans_b = args.K, args.M, args.trans_b

    block_M = 64
    block_N = 128
    block_K = 64
    num_stages = 2
    threads = 256

    run_tilelang_grouped_gemm(batch_sizes_list, K, M, block_M, block_N, block_K, trans_b, num_stages, threads, profile=args.profile)
