# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from . import CUDA


class H100(CUDA):

    def __init__(self, target):
        super().__init__(target)
        self.core = "H100"
        self.sm_count = 132
        self.base_freq = 1.41 * 1e9
        self.max_freq = 1.41 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 16)
        self.tensor_core_flops = 1024
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 3352.32 * 1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 8748e9
        self.l2_capacity = 50 * (1024**2)
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm = 16
        self.fp16_mixed_precision_tflops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tflops = self.fp16_mixed_precision_tflops * 2
        self.fp32_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.int32_cuda_core_tflops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_tflops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9
