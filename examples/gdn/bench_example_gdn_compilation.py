import tilelang.tools.bench
import example_chunk_delta_bwd
import example_chunk_delta_h
import example_chunk_o
import example_chunk_o_bwd
import example_chunk_scaled_dot_kkt
import example_cumsum
import example_wy_fast
import example_wy_fast_bwd_split
import torch


def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.tilelang_recompute_w_u_fwd)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)



def bench_example_cumsum():
    tilelang.tools.bench.process_func(example_cumsum.tilelang_chunk_local_cumsum_scalar)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)



def bench_example_wy_fast():
    tilelang.tools.bench.process_func(example_wy_fast.prepare_input)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
