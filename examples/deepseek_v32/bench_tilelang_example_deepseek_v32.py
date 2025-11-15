import tilelang.tools.bench
import fp8_lighting_indexer
import sparse_mla_bwd
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import topk_selector


def bench_topk_selector():
    tilelang.tools.bench.process_func(topk_selector.test_topk_selector)


def bench_fp8_lighting_indexer():
    tilelang.tools.bench.process_func(fp8_lighting_indexer.test_fp8_lighting_indexer, S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def bench_sparse_mla_fwd():
    tilelang.tools.bench.process_func(sparse_mla_fwd.test_sparse_mla_fwd, S=256, SKV=1024, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def bench_sparse_mla_fwd_pipelined():
    tilelang.tools.bench.process_func(sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined, S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def bench_sparse_mla_bwd():
    tilelang.tools.bench.process_func(sparse_mla_bwd.test_sparse_mla_bwd,  S=256, SKV=512, H=64, HKV=1, DQKV=576, DV=512, topk=256, check_correctness=False)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
