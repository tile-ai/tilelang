import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling(
    max_num_pooling_blocks: int,
    pooling_block_size: int,
    dim: int,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 256,
):
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32

    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    scale_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]
    blocked_k_scale_size = [max_num_pooling_blocks]
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def fp8_native_block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype),                        # type: ignore
        KScale: T.Tensor(scale_size, dtype=accum_dtype),         # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=dtype),         # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_size, dtype=accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                T.copy(K[tl_block_s:tl_block_s + block_N, :], index_k)
                for bn_i in T.Parallel(block_N):
                    scale[bn_i] = KScale[tl_block_s + bn_i]

                for bn_i, d_i in T.Parallel(block_N, dim):
                    index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

                cur_tl_block_size = tl_block_e - tl_block_s
                for n_i in T.parallel(block_N):
                    for d_i in T.parallel(dim):
                        if n_i >= cur_tl_block_size:
                            index_k[n_i, d_i] = T.cast(0, accum_dtype)

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            inv_count = T.cast(1.0, accum_dtype) / T.cast(cur_pooling_block_size, accum_dtype)
            for d_i in T.Parallel(dim):
                acc[d_i] = acc[d_i] * inv_count

            # Re-quantize f32 mean to fp8 with a per-block scale.
            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype), T.cast(1e-10, accum_dtype))
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

            for d_i in T.Parallel(dim):
                BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[bx] = block_scale

    return fp8_native_block_mean_pooling_kernel


def fp8_native_block_mean_pooling_interface(k: torch.Tensor, k_scale: torch.Tensor, k_block_size: int):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty((max_num_pooling_blocks, d), device=k.device, dtype=torch.float8_e4m3fn)
    blocked_k_scale = torch.empty((max_num_pooling_blocks,), device=k.device, dtype=torch.float32)
    kernel = fp8_native_block_mean_pooling(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
    )
    kernel(k, k_scale, blocked_k, blocked_k_scale)
    return blocked_k, blocked_k_scale


def ref_fp8_block_mean_pooling(k_fp8: torch.Tensor, k_scale: torch.Tensor, k_block_size: int) -> torch.Tensor:
    """Spec: per-token dequant + per-block mean (dividing by actual valid count).
    Returns the f32 mean (caller can compare against fp8*scale re-quant of the kernel)."""
    N, D = k_fp8.shape
    dequant = k_fp8.float() * k_scale[:, None]
    num_blocks = (N + k_block_size - 1) // k_block_size
    out = torch.empty(num_blocks, D, device=k_fp8.device, dtype=torch.float32)
    for b in range(num_blocks):
        s = b * k_block_size
        e = min(s + k_block_size, N)
        out[b] = dequant[s:e].sum(dim=0) / (e - s)
    return out


def test_fp8_block_mean_pooling(N: int = 16384, D: int = 128, k_block_size: int = 128):
    torch.manual_seed(0)
    k_bf16 = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    k = k_bf16.to(torch.float8_e4m3fn)
    k_scale = (0.1 + 0.01 * torch.rand(N, device="cuda", dtype=torch.float32)).contiguous()

    # Correctness.
    blocked_k_fp8, blocked_k_scale = fp8_native_block_mean_pooling_interface(k, k_scale, k_block_size)
    got = blocked_k_fp8.float() * blocked_k_scale[:, None]
    ref = ref_fp8_block_mean_pooling(k, k_scale, k_block_size)
    # fp8 re-quant: ~1/256 rel error on top of bf16-level precision.
    torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-3)
    print(f"  correctness: PASS  (N={N}, D={D}, k_block_size={k_block_size})")

    # Speed.
    def fn():
        return fp8_native_block_mean_pooling_interface(k, k_scale, k_block_size)
    ms = do_bench(fn, warmup=50, rep=200)
    num_blocks = (N + k_block_size - 1) // k_block_size
    # Bytes moved: read N * D fp8 (K) + N * 4 f32 (scale) + write num_blocks * D fp8 + num_blocks * 4 f32.
    bytes_moved = N * D + N * 4 + num_blocks * D + num_blocks * 4
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    print(f"  latency: {ms:.4f} ms  ({gbps:.1f} GB/s)")


if __name__ == "__main__":
    for cfg in [(4096, 128, 128), (16384, 128, 128), (65536, 128, 128), (131072, 128, 128)]:
        test_fp8_block_mean_pooling(*cfg)
