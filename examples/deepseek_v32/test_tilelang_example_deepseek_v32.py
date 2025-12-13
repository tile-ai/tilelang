# ruff: noqa
import tilelang
import tilelang.testing

import topk_selector
import fp8_lighting_indexer
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import sparse_mla_bwd


def test_example_topk_selector():
    topk_selector.test_topk_selector()


def test_example_fp8_lighting_indexer():
    fp8_lighting_indexer.test_fp8_lighting_indexer(S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_sparse_mla_fwd():
    # small shapes for testing
    sparse_mla_fwd.test_sparse_mla_fwd(S=256, SKV=1024, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined(S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_sparse_mla_bwd():
    sparse_mla_bwd.test_sparse_mla_bwd(S=256, SKV=512, H=64, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False)


if __name__ == "__main__":
    tilelang.testing.main()
