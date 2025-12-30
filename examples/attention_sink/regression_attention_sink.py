import tilelang.testing
import example_mha_sink_fwd_bhsd
import example_mha_sink_fwd_bhsd_wgmma_pipelined
import example_mha_sink_bwd_bhsd
import example_gqa_sink_bwd_bhsd
import example_gqa_sink_fwd_bhsd_wgmma_pipelined


def regression_example_mha_sink_fwd_bhsd():
    tilelang.testing.process_func(example_mha_sink_fwd_bhsd.run_regression_perf)


def regression_example_mha_sink_fwd_bhsd_sliding_window():
    tilelang.testing.process_func(
        example_mha_sink_fwd_bhsd.run_regression_perf, "regression_example_mha_sink_fwd_bhsd_sliding_window", window_size=128
    )


def regression_example_mha_sink_fwd_bhsd_wgmma_pipelined():
    tilelang.testing.process_func(example_mha_sink_fwd_bhsd_wgmma_pipelined.run_regression_perf)


def regression_example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    tilelang.testing.process_func(
        example_mha_sink_fwd_bhsd_wgmma_pipelined.run_regression_perf,
        "regression_example_mha_sink_fwd_bhsd_wgmma_pipelined_sliding_window",
        window_size=128,
    )


def regression_example_gqa_sink_fwd_bhsd_wgmma_pipelined():
    tilelang.testing.process_func(example_gqa_sink_fwd_bhsd_wgmma_pipelined.run_regression_perf)


def regression_example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window():
    tilelang.testing.process_func(
        example_gqa_sink_fwd_bhsd_wgmma_pipelined.run_regression_perf,
        "regression_example_gqa_sink_fwd_bhsd_wgmma_pipelined_sliding_window",
        window_size=128,
    )


def regression_example_mha_sink_bwd_bhsd():
    tilelang.testing.process_func(example_mha_sink_bwd_bhsd.run_regression_perf)


def regression_example_mha_sink_bwd_bhsd_sliding_window():
    tilelang.testing.process_func(
        example_mha_sink_bwd_bhsd.run_regression_perf, "regression_example_mha_sink_bwd_bhsd_sliding_window", window_size=128
    )


def regression_example_gqa_sink_bwd_bhsd():
    tilelang.testing.process_func(example_gqa_sink_bwd_bhsd.run_regression_perf)


def regression_example_gqa_sink_bwd_bhsd_sliding_window():
    tilelang.testing.process_func(
        example_gqa_sink_bwd_bhsd.run_regression_perf, "regression_example_gqa_sink_bwd_bhsd_sliding_window", window_size=128
    )


if __name__ == "__main__":
    tilelang.testing.regression()
