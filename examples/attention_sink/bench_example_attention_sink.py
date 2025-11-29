import tilelang.tools.bench
import example_gqa_sink_bwd_bhsd
import example_gqa_sink_fwd_bhsd_wgmma_pipelined
import example_mha_sink_bwd_bhsd
import example_mha_sink_fwd_bhsd
import example_mha_sink_fwd_bhsd_wgmma_pipelined


# def bench_example_mha_sink_fwd_bhsd():
#     tilelang.tools.bench.process_func(example_mha_sink_fwd_bhsd.benchmark)


# def bench_example_mha_sink_fwd_bhsd_sliding_window():
#     tilelang.tools.bench.process_func(
#         example_mha_sink_fwd_bhsd.benchmark,
#         name="example_mha_sink_fwd_bhsd_sliding_window",
#         window_size=128)


# def bench_example_mha_sink_fwd_bhsd_wgmma_pipelined():
#     tilelang.tools.bench.process_func(example_mha_sink_fwd_bhsd_wgmma_pipelined.benchmark)


# def bench_example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
#     tilelang.tools.bench.process_func(
#         example_mha_sink_fwd_bhsd_wgmma_pipelined.benchmark,
#         name="example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window",
#         window_size=128)


# def bench_example_gqa_sink_fwd_bhsd_wgmma_pipelined():
#     tilelang.tools.bench.process_func(example_gqa_sink_fwd_bhsd_wgmma_pipelined.benchmark)


# def bench_example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
#     tilelang.tools.bench.process_func(
#         example_gqa_sink_fwd_bhsd_wgmma_pipelined.benchmark,
#         name="example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window",
#         window_size=128)


def bench_example_mha_sink_bwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_sink_bwd_bhsd.benchmark)


# def bench_example_mha_sink_bwd_bhsd_sliding_window():
#     tilelang.tools.bench.process_func(
#         example_mha_sink_bwd_bhsd.benchmark,
#         name="example_mha_sink_bwd_bhsd_sliding_window",
#         window_size=128)


# def bench_example_gqa_sink_bwd_bhsd():
#     tilelang.tools.bench.process_func(example_gqa_sink_bwd_bhsd.benchmark)


# def bench_example_gqa_sink_bwd_bhsd_sliding_window():
#     tilelang.tools.bench.process_func(
#         example_gqa_sink_bwd_bhsd.benchmark,
#         name="example_gqa_sink_bwd_bhsd_sliding_window",
#         window_size=128)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
