# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from . import CUDA

class RTX4090(CUDA):
    def __init__(self, target):
        super().__init__(target)
        self.core = "RTX4090"
        self.sm_count = 128
        self.base_freq = 2.53 * 1e9
        self.max_freq = 2.53 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)
        self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 128
        self.ddr_bandwidth = 1008 * 1e9
        self.ddr_capacity = 24 * (1024**3)
        self.l2_bandwidth = 4510 * 1e9 *self.base_freq/2.23
        self.l2_capacity = 72 * (1024**2) # 
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 100 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        self.fp32_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 / 64
        self.sfu_tflops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9