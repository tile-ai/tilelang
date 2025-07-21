#!/bin/bash
# We use reduce-scatter as a comprehensive test for our pynvshmem implementation.

TILELANG_PATH=$NVSHMEM_PATH/../..
bash $TILELANG_PATH/tilelang/distributed/launch.sh \
    $TILELANG_PATH/benchmark/distributed/benchmark_reduce_scatter.py