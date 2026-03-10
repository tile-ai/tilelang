# ruff: noqa
import tilelang
import tilelang.testing
import torch

import topk_selector
import fp8_lighting_indexer
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import sparse_mla_bwd
from utils import generate_random_cu_seqlens, per_custom_dims_cast_to_fp8


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_topk_selector():
    topk_selector.test_topk_selector()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_fp8_lighting_indexer():
    fp8_lighting_indexer.test_fp8_lighting_indexer(S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_fp8_lighting_indexer_ws_single_launch():
    S = 128
    SKV = 256
    H = 32
    D = 64

    torch.manual_seed(0)
    q = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(SKV, D, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(S, H, device="cuda", dtype=torch.float32)
    ks, ke = generate_random_cu_seqlens(
        per_cp_seqlen=S,
        cp_size=4,
        cp_rank=3,
        kv_stride=1,
        average_q_len=64,
    )

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kernel = fp8_lighting_indexer.mqa_attn_return_logits(heads=H, index_dim=D)
    source = kernel.get_kernel_source()

    assert "mbarrier[0].init(128);" in source
    assert "mbarrier[3].init(512);" in source
    assert "mbarrier[6].expect_transaction(8192);" in source
    assert "tl::mbarrier_cp_async_arrive(mbarrier[(nbn_i % 3)]);" in source

    logits = torch.empty([S, SKV], device="cuda", dtype=torch.float32)
    kernel(q_fp8.view(S * H, D), kv_fp8, kv_scales, logits, weights, ks, ke)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_sparse_mla_fwd():
    # small shapes for testing
    sparse_mla_fwd.test_sparse_mla_fwd(S=256, SKV=1024, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined(S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_example_sparse_mla_bwd():
    sparse_mla_bwd.test_sparse_mla_bwd(S=256, SKV=512, H=64, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False)
    sparse_mla_bwd.test_sparse_mla_bwd(
        S=256, SKV=512, H=128, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False
    )  # test for large H


if __name__ == "__main__":
    tilelang.testing.main()
