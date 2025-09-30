# ruff: noqa
import tilelang.testing

from topk_selector import test_topk_selector
from fp8_lighting_indexer import test_fp8_lighting_indexer
from sparse_mla_fwd import test_sparse_mla_fwd
from sparse_mla_fwd_pipelined import test_sparse_mla_fwd_pipelined


def test_example_topk_selector():
    test_topk_selector()


def test_example_fp8_lighting_indexer():
    test_fp8_lighting_indexer()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_sparse_mla_fwd():
    # small shapes for testing
    test_sparse_mla_fwd(S=1024, SKV=2048, H=128, HKV=1, DQK=576, DV=512, topk=256)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    test_sparse_mla_fwd_pipelined(S=1024, SKV=2048, H=128, HKV=1, DQK=576, DV=512, topk=256)


if __name__ == "__main__":
    tilelang.testing.main()
