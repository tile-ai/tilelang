import tilelang.testing

import example_gqa_bwd
import example_mha_bwd_bshd
import example_mha_bwd_bhsd
import example_gqa_fwd_bshd
import example_mha_fwd_bshd
import example_mha_fwd_varlen
import example_mha_fwd_bhsd
import example_gqa_bwd_tma_reduce_varlen
import example_gqa_fwd_varlen


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_gqa_bwd_tma_reduce_varlen():
    example_gqa_bwd_tma_reduce_varlen.main()


@tilelang.testing.requires_cuda
def test_example_gqa_bwd():
    example_gqa_bwd.main()


@tilelang.testing.requires_cuda
def test_example_mha_bwd():
    example_mha_bwd_bshd.main(
        BATCH=1,
        H=16,
        N_CTX=512,
        D_HEAD=64,
        causal=False,
    )


@tilelang.testing.requires_cuda
def test_example_mha_bwd_bhsd():
    example_mha_bwd_bhsd.main(
        BATCH=1,
        H=16,
        N_CTX=512,
        D_HEAD=64,
        causal=False,
    )


@tilelang.testing.requires_cuda
def test_example_gqa_fwd_bshd():
    example_gqa_fwd_bshd.main(batch=1, heads=16, seq_len=1024, dim=128, is_causal=False, groups=16, tune=False)


@tilelang.testing.requires_cuda
def test_example_mha_fwd_bhsd():
    example_mha_fwd_bhsd.main()


@tilelang.testing.requires_cuda
def test_example_mha_fwd_bshd():
    example_mha_fwd_bshd.main(batch=1, seq_len=256)


@tilelang.testing.requires_cuda
def test_example_mha_fwd_varlen():
    example_mha_fwd_varlen.main(batch=4, heads=16, seq_len=512, dim=64, causal=False)
    example_mha_fwd_varlen.main(batch=4, heads=16, seq_len=512, dim=64, causal=True)


@tilelang.testing.requires_cuda
def test_example_gqa_fwd_varlen():
    example_gqa_fwd_varlen.main(batch=4, heads=16, q_seqlen=512, k_seqlen=512, dim=64, is_causal=False)
    example_gqa_fwd_varlen.main(batch=4, heads=16, q_seqlen=512, k_seqlen=512, dim=64, is_causal=True)


if __name__ == "__main__":
    tilelang.testing.main()
