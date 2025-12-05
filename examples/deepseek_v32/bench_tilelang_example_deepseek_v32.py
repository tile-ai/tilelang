import tilelang.tools.bench
import fp8_lighting_indexer
import sparse_mla_bwd
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import topk_selector


def bench_topk_selector():
    tilelang.tools.bench.process_func(topk_selector.benchmark)


def bench_fp8_lighting_indexer():
    tilelang.tools.bench.process_func(
        fp8_lighting_indexer.benchmark, S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def bench_sparse_mla_fwd():
    tilelang.tools.bench.process_func(
        sparse_mla_fwd.benchmark,
        S=256,
        SKV=1024,
        H=64,
        HKV=1,
        DQK=576,
        DV=512,
        topk=256,
        check_correctness=False)


def bench_sparse_mla_fwd_pipelined():
    tilelang.tools.bench.process_func(
        sparse_mla_fwd_pipelined.benchmark,
        S=256,
        SKV=512,
        H=64,
        HKV=1,
        DQK=576,
        DV=512,
        topk=256,
        check_correctness=False)


def bench_sparse_mla_bwd():
    tilelang.tools.bench.process_func(
        sparse_mla_bwd.benchmark,
        S=256,
        SKV=512,
        H=64,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=256,
        check_correctness=False)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
