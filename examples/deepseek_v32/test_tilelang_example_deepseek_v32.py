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

def test_example_sparse_mla_fwd():
    test_sparse_mla_fwd()

def test_example_sparse_mla_fwd_pipelined():
    test_sparse_mla_fwd_pipelined()


if __name__ == "__main__":
    tilelang.testing.main()
