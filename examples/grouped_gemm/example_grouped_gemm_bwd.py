import torch
import math
import argparse
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2], pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def grouped_gemm_fwd(batch_sum, batch_count, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(batch_sum, block_M) + batch_count, T.ceildiv(N, block_N), threads=threads) as (bx, by):
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


class _GroupedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, batch_sizes):
        block_M = 64
        block_N = 64
        block_K = 64
        padding_M = block_M
        num_stages = 2
        threads = 128
        batch_sum = a.shape[0]
        batch_count = b.shape[0]
        K = a.shape[1]
        N = b.shape[2]

        assert a.shape[1] == b.shape[1]
        assert batch_sizes.shape[0] == batch_count
        assert batch_sizes.sum() == batch_sum

        batch_offsets_list = [0]
        batch_padded_offsets_list = [0]
        for i in range(batch_count - 1):
            batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes[i])
        for i in range(batch_count - 1):
            batch_padded_offsets_list.append(batch_padded_offsets_list[-1] + math.ceil((batch_sizes[i] + 1) / padding_M) * padding_M)
        batch_offsets = torch.tensor(batch_offsets_list, device=a.device, dtype=torch.int32)
        batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=a.device, dtype=torch.int32)

        kernel = grouped_gemm_fwd(batch_sum, batch_count, K, N, block_M, block_N, block_K, num_stages, threads)

        o = kernel(a, b, batch_sizes, batch_offsets, batch_padded_offsets)
        ctx.save_for_backward(a, b, batch_sizes, batch_offsets)
        ctx.batch_sum = batch_sum
        ctx.batch_count = batch_count
        ctx.K = K
        return o

    @staticmethod
    def backward(ctx, grad_output):
        block_M = 64
        block_N = 64
        block_K = 64
        num_stages = 2
        threads = 128

        M = ctx.K
        N = grad_output.shape[1]

        A, B, batch_sizes, batch_offsets = ctx.saved_tensors

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        A, B, batch_sizes = [maybe_contiguous(x) for x in (A, B, batch_sizes)]
        kernel = grouped_gemm_bwd(ctx.batch_sum, ctx.batch_count, M, N, block_M, block_N, block_K, num_stages, threads)

        dB = kernel(A, grad_output, batch_sizes, batch_offsets)
        return None, dB, None


def ref_program(a, b, batch_sizes):
    assert a.shape[0] == sum(batch_sizes)
    assert b.shape[0] == len(batch_sizes)

    output = torch.empty((sum(batch_sizes), b.shape[2]), device=a.device, dtype=a.dtype)

    start = 0
    a_list = []
    b_list = []
    for i, size in enumerate(batch_sizes):
        end = start + size
        part_a = a[start:end]
        part_b = b[i]
        output[start:end] = torch.mm(part_a, part_b)

        a_list.append(part_a)
        b_list.append(part_b)
        start = end

    return output


def construct_inputs(batch_sizes_list, K, M, trans_b, padding_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] + math.ceil((batch_sizes_list[i] + 1) / padding_M) * padding_M)
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


@tilelang.jit(out_idx=[2], pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
def grouped_gemm_bwd(batch_sum, batch_count, M, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    """
    args:
        a (torch.Tensor): Input tensor of shape (M, K).
        b (torch.Tensor): Input tensor of shape (G, K, N).
    """
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, M], dtype),  # type: ignore
        B: T.Tensor([batch_sum, N], dtype),  # type: ignore
        C: T.Tensor([batch_count, M, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), batch_count, threads=threads) as (bx, by, bz):
            A_shared = T.alloc_shared([block_K, block_M], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(batch_sizes[bz], block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_K, block_M):
                    A_shared[i, j] = T.if_then_else(i < batch_sizes[bz], A[batch_offsets[bz] + k * block_K + i, bx * block_M + j], 0)
                for i, j in T.Parallel(block_K, block_N):
                    B_shared[i, j] = T.if_then_else(i < batch_sizes[bz], B[batch_offsets[bz] + k * block_K + i, by * block_N + j], 0)
                T.gemm(A_shared, B_shared, C_local, transpose_A=True)

            T.copy(C_local, C[bz, bx * block_M, by * block_N])

    return kernel


def run_tilelang_grouped_gemm(batch_sizes_list, K, M, block_M, block_N, block_K, trans_b, num_stages=2, threads=128, profile=False):
    padding_M = block_M
    device = torch.device("cuda")
    dtype = torch.float16

    A, B, C, batch_sizes, batch_offsets, batch_padded_offsets = construct_inputs(batch_sizes_list, K, M, False, padding_M, device, dtype)

    A.requires_grad_(False)
    B.requires_grad_(True)
    O_ref = ref_program(A, B, batch_sizes)
    dO = torch.randn_like(O_ref)

    O_ref.backward(dO, retain_graph=True)
    dB_ref, B.grad = B.grad.clone(), None

    GroupedGEMM = _GroupedGEMM.apply
    O = GroupedGEMM(A, B, batch_sizes)
    O.backward(dO, retain_graph=True)
    dB, B.grad = B.grad.clone(), None

    if torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2) and torch.allclose(dB, dB_ref, rtol=1e-2, atol=1e-2):
        print("✅ Tilelang and Torch match")
    else:
        print("❌ Tilelang and Torch mismatch")


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
