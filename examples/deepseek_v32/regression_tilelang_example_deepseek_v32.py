import tilelang.testing
import fp8_lighting_indexer
import sparse_mla_bwd
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import topk_selector


def regression_topk_selector():
    tilelang.testing.process_func(topk_selector.run_regression_perf)


def regression_fp8_lighting_indexer():
    tilelang.testing.process_func(fp8_lighting_indexer.run_regression_perf, S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def regression_sparse_mla_fwd():
    tilelang.testing.process_func(sparse_mla_fwd.run_regression_perf, S=256, SKV=1024, H=64, HKV=1, DQK=576, DV=512, topk=256)


def regression_sparse_mla_fwd_pipelined():
    tilelang.testing.process_func(sparse_mla_fwd_pipelined.run_regression_perf, S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256)


def regression_sparse_mla_bwd():
    tilelang.testing.process_func(sparse_mla_bwd.run_regression_perf, S=256, SKV=512, H=64, HKV=1, DQKV=576, DV=512, topk=256)


if __name__ == "__main__":
    tilelang.testing.regression()
