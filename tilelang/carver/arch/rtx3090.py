# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from . import CUDA

class RTX3090(CUDA):
    def __init__(self, target):
        super().__init__(target)
        self.core = "RTX3090"
        self.sm_count = 82
        self.base_freq = 1.695 * 1e9
        self.max_freq = 1.695 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)
        self.tensor_core_flops =256
        self.fp32_cores_per_sm = 128
        self.ddr_bandwidth = 936 * 1e9
        self.ddr_capacity = 24 * (1024**3)
        self.l2_bandwidth = 2722.7 * 1e9
        self.l2_capacity = 6 * (1024**2) # 
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 100 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        self.fp16_mixed_precision_tflops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tflops = self.fp16_mixed_precision_tflops * 2 
        self.int4_tflops = self.fp16_mixed_precision_tflops * 4 
        self.int8_int2_tflops = self.fp16_mixed_precision_tflops * 1
        self.int8_int1_tflops = self.fp16_mixed_precision_tflops * 2
        self.fp32_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1/64
        self.fp64_divide_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1/64 /8
        self.sfu_tflops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4
        self.ddr_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9
        self.l2_max_util=2345.755191/2722.7
