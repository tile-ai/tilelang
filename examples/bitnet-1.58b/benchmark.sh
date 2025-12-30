#!/usr/bin/env bash

python benchmark_generate.py --bs 16 --in_seq_len 32 --out_seq_len 128 | tee b16_i32_o128.log

python benchmark_generate.py --bs 1 --in_seq_len 512 --out_seq_len 64 | tee b1_i512_o64.log

python benchmark_generate.py --bs 32 --in_seq_len 32 --out_seq_len 128 | tee b32_i32_o128.log

python benchmark_generate.py --bs 16 --in_seq_len 32 --out_seq_len 128 --bitblas | tee b16_i32_o128_bitblas.log

python benchmark_generate.py --bs 1 --in_seq_len 512 --out_seq_len 64 --bitblas | tee b1_i512_o64_bitblas.log

python benchmark_generate.py --bs 32 --in_seq_len 32 --out_seq_len 128 --bitblas | tee b32_i32_o128_bitblas.log
