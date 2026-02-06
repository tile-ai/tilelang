# Benchmarks for IPC communication

This benchmark aims to measure and compare the bandwidth of different implementations of IPC communication:
We launch only one block on each rank to avoid NVLink bandwidth as the bottleneck.

## NVSHMEM-based push/pull
```bash
GPUS=2 bash tilelang/distributed/launch.sh benchmark/distributed/ipc_impls/benchmark_nvshmem_p2p.py
```

## Unrolled-copy implemented in TileScale (*ours*)
```bash
export TILELANG_USE_DISTRIBUTED=1
python benchmark/distributed/ipc_impls/benchmark_unrolledcp_p2p.py
```

## Results on Hopper connected by NVLink
|   Size (Bytes) | NVSHMEM Push BW (GB/s) | NVSHMEM Pull BW (GB/s) | TileScale Push BW (GB/s) | TileScale Pull BW (GB/s) |
|---------------:|----------------------:|-----------------------:|-------------------------:|--------------------------:|
|          2,048 |                0.1680 |                 0.1755 |                  0.0632 |                  0.0628  |
|          4,096 |                0.3415 |                 0.4082 |                  0.1316 |                  0.1284  |
|          8,192 |                0.6836 |                 0.8497 |                  0.2601 |                  0.2628  |
|         16,384 |                1.4119 |                 1.6178 |                  0.5241 |                  0.5232  |
|         32,768 |                2.4592 |                 1.8878 |                  1.0178 |                  1.1283  |
|         65,536 |                4.9380 |                 2.0408 |                  2.0380 |                  1.9723  |
|        131,072 |                8.7134 |                 2.1465 |                  3.9668 |                  2.1001  |
|        262,144 |                9.0743 |                 2.1935 |                  8.0200 |                  2.1920  |
|        524,288 |               10.0191 |                 2.2156 |                 10.7943 |                  2.2509  |
|      1,048,576 |               10.4359 |                 2.2352 |                 11.4781 |                  2.2648  |
|      2,097,152 |               10.5573 |                 2.2456 |                 11.7712 |                  2.2796  |
|      4,194,304 |               10.6560 |                 2.2474 |                 11.9145 |                  2.2845  |

> **Note:** All data presented above are unidirectional bandwidth.
