import tilelang.tools.bench
import example_gqa_fwd_bshd
import example_gqa_fwd_bshd_wgmma_pipelined
import example_mha_fwd_bhsd
import example_mha_fwd_bhsd_wgmma_pipelined
import example_mha_fwd_bshd
import example_mha_fwd_bshd_wgmma_pipelined
import example_mha_fwd_varlen
import example_gqa_bwd_tma_reduce_varlen
import example_gqa_bwd
import example_gqa_bwd_wgmma_pipelined
import example_mha_bwd_bshd
import example_mha_bwd_bhsd
import example_mha_bwd_bshd_wgmma_pipelined


def bench_example_gqa_bwd_tma_reduce_varlen():
    tilelang.tools.bench.process_func(
        example_gqa_bwd_tma_reduce_varlen.benchmark, name="example_gqa_bwd_tma_reduce_varlen")


def bench_example_gqa_bwd():
    tilelang.tools.bench.process_func(example_gqa_bwd.benchmark, name="example_gqa_bwd")


def bench_example_gqa_bwd_wgmma_pipelined():
    tilelang.tools.bench.process_func(
        example_gqa_bwd_wgmma_pipelined.benchmark, name="example_gqa_bwd_wgmma_pipelined")


def bench_example_mha_bwd_bshd():
    tilelang.tools.bench.process_func(example_mha_bwd_bshd.benchmark, name="example_mha_bwd_bshd")


def bench_example_mha_bwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_bwd_bhsd.benchmark, name="example_mha_bwd_bhsd")


def bench_example_mha_bwd_bshd_wgmma_pipelined():
    tilelang.tools.bench.process_func(
        example_mha_bwd_bshd_wgmma_pipelined.benchmark, name="example_mha_bwd_bshd_wgmma_pipelined")


def bench_example_gqa_fwd_bshd_wgmma_pipelined():
    tilelang.tools.bench.process_func(
        example_gqa_fwd_bshd_wgmma_pipelined.benchmark,
        batch=1,
        heads=16,
        seq_len=1024,
        dim=128,
        is_causal=False,
        groups=16,
        tune=False)


def bench_example_gqa_fwd_bshd():
    tilelang.tools.bench.process_func(
        example_gqa_fwd_bshd.benchmark,
        batch=1,
        heads=16,
        seq_len=1024,
        dim=128,
        is_causal=False,
        groups=16,
        tune=False)


def bench_example_mha_fwd_bhsd_wgmma_pipelined():
    tilelang.tools.bench.process_func(example_mha_fwd_bhsd_wgmma_pipelined.benchmark)


def bench_example_mha_fwd_bhsd():
    tilelang.tools.bench.process_func(example_mha_fwd_bhsd.benchmark)


def bench_example_mha_fwd_bshd_wgmma_pipelined():
    tilelang.tools.bench.process_func(
        example_mha_fwd_bshd_wgmma_pipelined.benchmark, batch=1, heads=32, seq_len=256)


def bench_example_mha_fwd_bshd():
    tilelang.tools.bench.process_func(example_mha_fwd_bshd.benchmark, batch=1, seq_len=256)


def bench_example_mha_fwd_varlen():
    tilelang.tools.bench.process_func(
        example_mha_fwd_varlen.benchmark, batch=4, heads=16, seq_len=512, dim=64)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
