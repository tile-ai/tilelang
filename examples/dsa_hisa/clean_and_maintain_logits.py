import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench
import torch


@tilelang.jit
def clean_and_maintain_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_and_maintain_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),           # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),           # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),           # type: ignore
    ):
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

    return clean_and_maintain_logits_kernel


def clean_and_maintain_logits_interface(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    """In-place: applies +inf/-inf mask based on per-row [ks, ke)."""
    kernel = clean_and_maintain_logits_()
    kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
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


def test_clean_and_maintain_logits(M: int = 4096, N: int = 4096):
    torch.manual_seed(0)
    # Build causal prefill ranges: cu_ks[m] = 0, cu_ke[m] = m + 1.
    logits_init = torch.randn(M, N, device="cuda", dtype=torch.float32)
    cu_ks = torch.zeros(M, device="cuda", dtype=torch.int32)
    cu_ke = (torch.arange(M, device="cuda") + 1).to(torch.int32).clamp(max=N)

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
    print(f"  correctness: PASS  (M={M}, N={N})")

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
    for cfg in [(4096, 4096), (16384, 16384), (65536, 65536)]:
        test_clean_and_maintain_logits(*cfg)
