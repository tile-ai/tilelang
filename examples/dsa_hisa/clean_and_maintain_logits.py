import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench
import torch

from tilelang_utils import prepare_ks_ke_from_cu_seqlens


@tilelang.jit
def clean_and_maintain_logits_(
    Logits,
    CuSeqLenKS,
    CuSeqLenKE,
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len, seq_len_kv = T.const("seq_len, seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    Logits: T.Tensor[[seq_len, seq_len_kv], dtype]
    CuSeqLenKS: T.Tensor[[seq_len], indices_dtype]
    CuSeqLenKE: T.Tensor[[seq_len], indices_dtype]

    with T.Kernel(seq_len, threads=threads) as bx:
        tx = T.thread_binding(0, threads, thread="threadIdx.x")
        cu_k_s = CuSeqLenKS[bx]
        cu_k_e = CuSeqLenKE[bx]

        for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
            for k_i in T.serial(block_K // threads):
                idx = n_i * block_K + k_i * threads + tx
                if idx == cu_k_s or idx == cu_k_e - 1:
                    Logits[bx, idx] = T.infinity(dtype)
                if idx < cu_k_s or idx >= cu_k_e:
                    Logits[bx, idx] = -T.infinity(dtype)


def clean_and_maintain_logits_interface(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    """In-place: applies +inf/-inf mask based on per-row [ks, ke)."""
    clean_and_maintain_logits_(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits


def ref_clean_and_maintain_logits(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Pure torch equivalent. Returns a new tensor (doesn't mutate the input)."""
    M, N = logits.shape
    out = logits.clone()
    n = torch.arange(N, device=logits.device)[None, :]
    mask_out = (n < cu_seqlen_ks.long()[:, None]) | (n >= cu_seqlen_ke.long()[:, None])
    out = out.masked_fill(mask_out, float("-inf"))
    m_idx = torch.arange(M, device=logits.device)
    out[m_idx, cu_seqlen_ks.long()] = float("inf")
    out[m_idx, (cu_seqlen_ke - 1).clamp(min=0).long()] = float("inf")
    return out


def test_clean_and_maintain_logits(M: int = 4096, N: int = 4096, num_seqs: int = 1):
    """Correctness + speed test where `M` query rows are packed from
    `num_seqs` equal-length causal sequences. Per-row ``cu_ks / cu_ke``
    is derived from ``prepare_ks_ke_from_cu_seqlens`` so each row sees
    only the prefix of its own sequence (causal self-attention)."""
    torch.manual_seed(0)
    assert M % num_seqs == 0, f"M ({M}) must be divisible by num_seqs ({num_seqs})"
    assert (M // num_seqs) <= N, "N must accommodate the longest sequence"

    per_seq = M // num_seqs
    cu_seqlens = torch.arange(num_seqs + 1, device="cuda", dtype=torch.long) * per_seq
    ks_long, ke_long = prepare_ks_ke_from_cu_seqlens(cu_seqlens)
    cu_ks = ks_long.to(torch.int32).contiguous()
    cu_ke = ke_long.to(torch.int32).clamp(max=N).contiguous()

    logits_init = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Run kernel in place on a copy.
    got = logits_init.clone()
    clean_and_maintain_logits_interface(got, cu_ks, cu_ke)

    # Ref.
    ref = ref_clean_and_maintain_logits(logits_init, cu_ks, cu_ke)

    # Exact equality: this kernel only writes +/-inf, other positions untouched
    # (ref clones the input and does the same). Compare directly.
    assert torch.equal(torch.isposinf(got), torch.isposinf(ref)), "pos-inf mask differs"
    assert torch.equal(torch.isneginf(got), torch.isneginf(ref)), "neg-inf mask differs"
    finite = torch.isfinite(got) & torch.isfinite(ref)
    torch.testing.assert_close(got[finite], ref[finite], rtol=0.0, atol=0.0)
    print(f"  correctness: PASS  (M={M}, N={N}, num_seqs={num_seqs}, per_seq={per_seq})")

    # Speed.
    def fn():
        logits = torch.randn(M, N, device="cuda", dtype=torch.float32)  # fresh copy each iter
        clean_and_maintain_logits_interface(logits, cu_ks, cu_ke)
        return logits

    ms = do_bench(fn, warmup=50, rep=200)
    # ~2 reads + 1 write of [M, N] f32, but mostly no-op except at mask boundaries.
    bytes_moved = 2 * M * N * 4
    gbps = bytes_moved / (ms * 1e-3) / 1e9
    print(f"  latency: {ms:.4f} ms  ({gbps:.1f} GB/s)")


if __name__ == "__main__":
    # (M, N, num_seqs)
    for cfg in [
        (4096, 4096, 1),
        (4096, 4096, 4),
        (16384, 16384, 1),
        (16384, 16384, 8),
        (65536, 65536, 16),
    ]:
        test_clean_and_maintain_logits(*cfg)
