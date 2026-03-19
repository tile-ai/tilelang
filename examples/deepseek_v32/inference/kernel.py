import torch
import tilelang
import tilelang.language as T
from typing import Tuple, Optional

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
}

FP8 = T.float8_e4m3fn
BF16 = T.bfloat16
FP32 = T.float32


def fast_log2_ceil(x):
    bits_x = T.reinterpret(x, T.uint32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.cast(exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0), T.int32)


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret(bits_x, T.float32)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    X,
    Y,
    S,
    N: int = 128,
    in_dtype=BF16,
    out_dtype=FP8,
    scale_dtype=FP32,
    round_scale: bool = False,
):
    M, _ = T.const("M _")
    X: T.Tensor[[M, N], in_dtype]
    Y: T.Tensor[[M, N], out_dtype]
    S: T.Tensor[[M, T.ceildiv(N, 128)], scale_dtype]

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    with T.Kernel(
            T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
                pid_m,
                pid_n,
            ):
        x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
        x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
        amax_local = T.alloc_fragment((blk_m,), scale_dtype)
        s_local = T.alloc_fragment((blk_m,), scale_dtype)
        y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
        y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

        for _ in T.Pipelined(1, num_stages=num_stages):
            T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
            T.copy(x_shared, x_local)
            T.reduce_absmax(x_local, amax_local, dim=1)
            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 1e-4)
                if round_scale:
                    s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                else:
                    s_local[i] = amax_local[i] * fp8_max_inv
            for i, j in T.Parallel(blk_m, group_size):
                y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
            for i in T.Parallel(blk_m):
                S[pid_m * blk_m + i, pid_n] = s_local[i]
            T.copy(y_local, y_shared)
            T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])


def act_quant(x: torch.Tensor,
              block_size: int = 128,
              scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})")
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    act_quant_kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size), N, round_scale=scale_fmt is not None)
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(
    A,
    B,
    C,
    scales_a,
    scales_b,
    N: int = 128,
    K: int = 128,
    out_dtype=BF16,
    accum_dtype=T.float32,
):
    assert out_dtype in [BF16, T.float32]

    M, _ = T.const("M _")
    A: T.Tensor[[M, K], FP8]
    B: T.Tensor[[N, K], FP8]
    C: T.Tensor[[M, N], out_dtype]
    scales_a: T.Tensor[[M, T.ceildiv(K, 128)], FP32]
    scales_b: T.Tensor[[T.ceildiv(N, 128), T.ceildiv(K, 128)], FP32]

    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
                bx,
                by,
            ):
        A_shared = T.alloc_shared((block_M, block_K), FP8)
        B_shared = T.alloc_shared((block_N, block_K), FP8)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)
        Scale_C_shared = T.alloc_shared((block_M), FP32)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

        # Improve L2 Cache
        T.use_swizzle(panel_size=10)

        T.clear(C_local)
        T.clear(C_local_accum)
        K_iters = T.ceildiv(K, block_K)
        for k in T.Pipelined(K_iters, num_stages=4):
            # Load A into shared memory
            T.copy(A[by * block_M, k * block_K], A_shared)
            # Load B into shared memory
            T.copy(B[bx * block_N, k * block_K], B_shared)
            # Load scale into shared memory
            Scale_B = scales_b[bx * block_N // group_size, k]
            for i in T.Parallel(block_M):
                Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            # Promote to enable 2xAcc
            for i, j in T.Parallel(block_M, block_N):
                C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
            T.clear(C_local)
        # TMA store
        T.copy(C_local_accum, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor,
             b_s: torch.Tensor) -> torch.Tensor:
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous")
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    fp8_gemm_kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s, N, K)
    return c


@tilelang.jit(pass_configs=pass_configs)
def fp8_index_kernel(
    q,
    q_s,
    k,
    k_s,
    h: int = 1,
    d: int = 64,
):
    b, m, _, _ = T.const("b m _ _")
    _, n, _ = T.const("_ n _")
    q: T.Tensor[[b, m, h, d], FP8]
    q_s: T.Tensor[[b, m, h], FP32]
    k: T.Tensor[[b, n, d], FP8]
    k_s: T.Tensor[[b, n], FP32]

    blk_n1 = 512
    blk_n2 = 128

    o = T.empty([b, m, n], FP32)

    with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
        q_smem = T.alloc_shared((h, d), FP8)
        T.copy(q[i_b, i_m, 0, 0], q_smem)

        q_s_frag = T.alloc_fragment(h, FP32)
        T.copy(q_s[i_b, i_m, 0], q_s_frag)

        for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
            k_smem = T.alloc_shared((blk_n2, d), FP8)
            T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

            k_s_frag = T.alloc_fragment(blk_n2, FP32)
            T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

            logits = T.alloc_fragment((blk_n2, h), FP32)
            T.gemm(
                k_smem,
                q_smem,
                logits,
                transpose_A=False,
                transpose_B=True,
                clear_accum=True,
            )

            for i_h, i3_n in T.Parallel(h, blk_n2):
                logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

            logits_sum = T.alloc_fragment(blk_n2, FP32)
            T.reduce_sum(logits, logits_sum, dim=1)

            for i3_n in T.Parallel(blk_n2):
                logits_sum[i3_n] *= k_s_frag[i3_n]

            T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

    return o


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.

    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    return fp8_index_kernel(q, q_s, k, k_s, q.shape[2], q.shape[3])
