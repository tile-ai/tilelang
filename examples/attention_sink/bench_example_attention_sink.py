import tilelang.tools.bench
import example_gqa_sink_bwd_bhsd
import example_gqa_sink_fwd_bhsd_wgmma_pipelined
import example_mha_sink_bwd_bhsd
import example_mha_sink_fwd_bhsd
import example_mha_sink_fwd_bhsd_wgmma_pipelined


@tilelang.testing.requires_cuda
def bench_example_mha_sink_fwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_sink_fwd_bhsd.main)


def bench_example_mha_sink_fwd_bhsd_sliding_window():
    tilelang.tools.bench.process_func(example_mha_sink_fwd_bhsd.main(window_size=128))


def bench_example_mha_sink_fwd_bhsd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_mha_sink_fwd_bhsd_wgmma_pipelined.main)


def bench_example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    tilelang.tools.bench.process_func(
        example_mha_sink_fwd_bhsd_wgmma_pipelined.main(window_size=128))


def bench_example_gqa_sink_fwd_bhsd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_gqa_sink_fwd_bhsd_wgmma_pipelined.main)


def bench_example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    tilelang.tools.bench.process_func(
        example_gqa_sink_fwd_bhsd_wgmma_pipelined.main(window_size=128))


def bench_example_mha_sink_bwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_sink_bwd_bhsd.main)


def bench_example_mha_sink_bwd_bhsd_sliding_window():
    tilelang.tools.bench.process_func(example_mha_sink_bwd_bhsd.main(window_size=128))


def bench_example_gqa_sink_bwd_bhsd():
    tilelang.tools.bench.process_func(example_gqa_sink_bwd_bhsd.main)


def bench_example_gqa_sink_bwd_bhsd_sliding_window():
    tilelang.tools.bench.process_func(example_gqa_sink_bwd_bhsd.main(window_size=128))


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
