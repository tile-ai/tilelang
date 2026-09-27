"""SM120 NVFP4 block-scaled GEMM with A taken from a register fragment.

Maintenance benchmark for the register-A (``is_gemm_rs``) lowering of
``T.gemm_blockscaled`` on SM120: A is staged through shared memory into a
full ``(block_M, block_K)`` fragment, B and the row-major packed scales stay in
shared memory. It reports latency and TFLOPS and, with ``--verify``, checks the
result bit for bit against a float32 reference.

Run from the repository root:

    python -m maint.gemm.gemm_sm120.benchmark_sm120_nvfp4_fragment_a_gemm --m 4096 --n 4096 --k 4096 --verify
"""

import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

_FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@tilelang.jit(out_idx=[4])
def fragment_a_blockscaled_gemm(M, N, K, block_M, block_N, block_K, num_stages, threads, warp_policy):
    in_dtype = T.float4_e2m1fn
    words = block_K // 64

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, K // 64), T.uint32),
        SFB: T.Tensor((N, K // 64), T.uint32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, words), T.uint32)
            A_frag = T.alloc_fragment((block_M, block_K), in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_frag)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.copy(SFA[by * block_M, ko * words], SFA_shared)
                T.copy(SFB[bx * block_N, ko * words], SFB_shared)
                for i, k in T.Parallel(block_M, block_K):
                    A_frag[i, k] = A_shared[i, k]
                T.gemm_blockscaled(
                    A_frag,
                    B_shared,
                    C_frag,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    policy=warp_policy,
                    clear_accum=False,
                    # SFA_shared / SFB_shared hold only this K block's scale words
                    k_start=0,
                    sf_a_granularity_k=16,
                    sf_b_granularity_k=16,
                    sf_layout="rowmajor",
                )
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def _decode_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.view(torch.uint8)
    lut = torch.tensor(_FP4_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[(u >> 4).long()]
    return out


def _make_scales(rows: int, k: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # Power-of-two ue4m3 scales keep the float32 reference exact.
    choices = torch.tensor([0x30, 0x38, 0x40], device="cuda", dtype=torch.uint8)
    idx = torch.randint(0, 3, (rows, k // 16), device="cuda", generator=generator)
    scale_bytes = choices[idx]
    s = scale_bytes.to(torch.int64).reshape(rows, -1, 4)
    words = s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
    return words.to(torch.uint32).contiguous(), scale_bytes


def _scale_values(scale_bytes: torch.Tensor) -> torch.Tensor:
    u = scale_bytes.to(torch.int32)
    return (1.0 + (u & 7).float() / 8.0) * torch.pow(2.0, ((u >> 3) & 15).float() - 7.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--policy", choices=["FullRow", "FullCol", "Square"], default="FullRow")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dump-source")
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    kernel = fragment_a_blockscaled_gemm(
        m,
        n,
        k,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_stages,
        args.threads,
        getattr(T.GemmWarpPolicy, args.policy),
    )
    if args.dump_source:
        with open(args.dump_source, "w") as f:
            f.write(kernel.get_kernel_source())

    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randint(-128, 128, (m, k // 2), device="cuda", dtype=torch.int8, generator=gen)
    b = torch.randint(-128, 128, (n, k // 2), device="cuda", dtype=torch.int8, generator=gen)
    sfa, sfa_bytes = _make_scales(m, k, gen)
    sfb, sfb_bytes = _make_scales(n, k, gen)

    c = kernel(a, b, sfa, sfb)
    if args.verify:
        a_f32 = _decode_fp4(a, m, k) * _scale_values(sfa_bytes).repeat_interleave(16, dim=1)
        b_f32 = _decode_fp4(b, n, k) * _scale_values(sfb_bytes).repeat_interleave(16, dim=1)
        ref = a_f32 @ b_f32.T
        torch.testing.assert_close(c, ref, rtol=0, atol=0)
        print("verify: exact")

    latency_ms = do_bench(lambda: kernel(a, b, sfa, sfb), warmup=25, rep=100)
    tflops = 2.0 * m * n * k / (latency_ms * 1e-3) / 1e12
    print(f"M={m} N={n} K={k} block={args.block_m}x{args.block_n}x{args.block_k} stages={args.num_stages} policy={args.policy}")
    print(f"latency: {latency_ms:.4f} ms  TFLOPS: {tflops:.1f}")


if __name__ == "__main__":
    main()
