# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from . import CUDA

class V100(CUDA):
    def __init__(self, target):
        super().__init__(target)
        self.core = "V100"
        self.sm_count = 80
        self.base_freq = 1.24 * 1e9
        self.max_freq = 1.53 * 1e9
        self.tensor_cores_per_sm = 8
        self.tensor_core_shape = (4, 4, 4)
        self.tensor_core_flops = 128
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 900 * 1e9
        self.ddr_capacity = 32 * (1024**3)
        self.l2_bandwidth = 2155 * 1e9
        self.l2_capacity = 6 * (1024**2)
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 96 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.fp16_mixed_precision_tflops = 125.34 * 1e12
        self.fp32_cuda_core_tflops = 15.67 * 1e12
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
