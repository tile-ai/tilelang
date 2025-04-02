# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import numpy as np
from dataclasses import dataclass
from tilelang import tvm
from tvm.tir.stmt_functor import ir_transform

ARCH_CONFIGS = {"80": (128, 1.41, 2, 108), "86": (128, 1.70, 2, 84), "89": (128, 2.52, 2, 128)}


@dataclass(frozen=True)
class AnalysisResult:
    total_flops: int
    total_global_bytes: int
    estimated_time: float
    tflops: float
    bandwidth_GBps: float


class Analyzer:

    def __init__(self, fn, device):
        if isinstance(fn, tvm.tir.function.PrimFunc):
            self.fn = tvm.IRModule({"main": fn})
        else:
            self.fn = fn
        self.device = device
        self.total_flops = 0
        self.total_global_bytes = 0
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1}
        self.loop_stack = []
        self.global_buffers = set()

    def _analyze_copy(self, call):
        src_buffer = call.args[0].args[0].buffer
        dst_buffer = call.args[1].args[0].buffer

        if src_buffer in self.global_buffers:
            buffer_region = call.args[0]
        elif dst_buffer in self.global_buffers:
            buffer_region = call.args[1]
        else:
            return

        elements = 1
        for r in range(2, len(buffer_region.args)):
            elements *= buffer_region.args[r]
        dtype_size = np.dtype(buffer_region.args[0].buffer.dtype).itemsize
        bytes_transferred = elements * dtype_size

        loop_product = 1
        for extent in self.loop_stack:
            loop_product *= extent.value if hasattr(extent, 'value') else extent
        total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
        total_bytes = bytes_transferred * loop_product * total_blocks
        self.total_global_bytes += total_bytes

    def _analyze_gemm(self, call):
        M = call.args[5].value
        N = call.args[6].value
        K = call.args[7].value
        flops_per_call = 2 * M * N * K

        loop_product = 1
        for extent in self.loop_stack:
            loop_product *= extent.value if hasattr(extent, 'value') else extent
        total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
        self.total_flops += flops_per_call * loop_product * total_blocks

    def ir_pass(self):

        def _ftransform(f, mod, ctx):
            self.global_buffers = set(f.buffer_map.values())

            def _pre_visit(stmt):
                # print(f"Pre Visiting node of type: {type(stmt)}")
                if isinstance(stmt, tvm.tir.AttrStmt):
                    if stmt.attr_key == "thread_extent":
                        iter_var = stmt.node
                        thread_tag = iter_var.thread_tag
                        if thread_tag in self.block_counts:
                            extent = stmt.value.value if hasattr(stmt.value,
                                                                 'value') else stmt.value
                            self.block_counts[thread_tag] = extent
                elif isinstance(stmt, tvm.tir.For):
                    self.loop_stack.append(stmt.extent)
                elif isinstance(stmt, tvm.tir.Evaluate):
                    value = stmt.value
                    if isinstance(value, tvm.tir.Call):
                        if value.op.name == "tl.copy":
                            self._analyze_copy(value)
                        elif value.op.name == "tl.gemm":
                            self._analyze_gemm(value)
                return None

            def _post_visit(stmt):
                if isinstance(stmt, tvm.tir.For) and self.loop_stack:
                    self.loop_stack.pop()
                return None

            new_body = ir_transform(f.body, _pre_visit, _post_visit)
            return f.with_body(new_body)

        tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.fn)
        return self

    def calculate(self) -> AnalysisResult:

        def get_peak_tflops(device) -> float:
            arch_key = device.compute_capability[:2]
            if arch_key not in ARCH_CONFIGS:
                raise ValueError(f"Unsupported compute capability: {device.compute_capability}")

            cores_per_sm, default_clock, flops_per_cycle, compute_max_core = ARCH_CONFIGS[arch_key]
            total_cores = compute_max_core * cores_per_sm
            tflops = (total_cores * default_clock * flops_per_cycle) / 1e3
            return round(tflops, 1)

        bandwidth_GBps = self.device.bandwidth[1] / 1000
        peak_tflops = get_peak_tflops(self.device)
        mem_time = self.total_global_bytes / (bandwidth_GBps * 1e9)
        compute_time = self.total_flops / (peak_tflops * 1e12)
        estimated_time = max(mem_time, compute_time)

        return AnalysisResult(
            total_flops=self.total_flops,
            total_global_bytes=self.total_global_bytes,
            estimated_time=float(estimated_time),
            tflops=float(self.total_flops / estimated_time / 1e12),
            bandwidth_GBps=bandwidth_GBps)

    @classmethod
    def analysis(cls, fn, device):
        return cls(fn, device).ir_pass().calculate()
