import tilelang.tools.bench
import example_gqa_bwd
import example_gqa_bwd_tma_reduce_varlen
import example_gqa_bwd_wgmma_pipelined
import example_gqa_fwd_bshd
import example_gqa_fwd_bshd_wgmma_pipelined
import example_mha_bwd
import example_mha_bwd_bhsd
import example_mha_bwd_wgmma_pipelined
import example_mha_fwd_bhsd
import example_mha_fwd_bhsd_wgmma_pipelined
import example_mha_fwd_bshd
import example_mha_fwd_bshd_wgmma_pipelined
import example_mha_fwd_varlen


def bench_example_gqa_bwd_tma_reduce_varlen():
    tilelang.tools.bench.process_func(example_gqa_bwd_tma_reduce_varlen.main)


def bench_example_gqa_bwd():
    tilelang.tools.bench.process_func(example_gqa_bwd.main)


def bench_example_gqa_bwd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_gqa_bwd_wgmma_pipelined.main)


def bench_example_mha_bwd():
    tilelang.tools.bench.process_func(example_mha_bwd.main)


def bench_example_mha_bwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_bwd_bhsd.main)


def bench_example_mha_bwd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_mha_bwd_wgmma_pipelined.main)


def bench_example_gqa_fwd_bshd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_gqa_fwd_bshd_wgmma_pipelined.main)


def bench_example_gqa_fwd_bshd():
    tilelang.tools.bench.process_func(example_gqa_fwd_bshd.main)


def bench_example_mha_fwd_bhsd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_mha_fwd_bhsd_wgmma_pipelined.main)


def bench_example_mha_fwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_fwd_bhsd.main)


def bench_example_mha_fwd_bshd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_mha_fwd_bshd_wgmma_pipelined.main)


def bench_example_mha_fwd_bshd():
    tilelang.tools.bench.process_func(example_mha_fwd_bshd.main)


def bench_example_mha_fwd_varlen():
    tilelang.tools.bench.process_func(example_mha_fwd_varlen.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
