from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from tilelang import tvm
from tvm.tir.stmt_functor import ir_transform
import logging

# Configuration for different hardware architectures.
# Each entry contains: (cores per SM, default clock (GHz), FLOPs per cycle, max SM count)
ARCH_CONFIGS = {"80": (128, 1.41, 2, 108), "86": (128, 1.70, 2, 84), "89": (128, 2.52, 2, 128)}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """
    A data class to store the results of the analysis.
    Attributes:
        total_flops: Total floating-point operations.
        total_global_bytes: Total bytes transferred to/from global memory.
        estimated_time: Estimated execution time (seconds).
        tflops: Achieved TFLOPS (trillions of FLOPs per second).
        bandwidth_GBps: Achieved memory bandwidth in GB/s.
    """

    total_flops: int
    total_global_bytes: int
    estimated_time: float
    expected_tflops: float
    expected_bandwidth_GBps: float


class Analyzer:
    """
    A class to analyze the performance of a TVM IR module.
    It calculates metrics such as FLOPs, memory bandwidth, and estimated execution time.
    """

    def __init__(self, fn, device):
        """
        Initialize the Analyzer.
        Args:
            fn: A TVM IRModule or PrimFunc to analyze.
            device: The target device information.
        """
        if isinstance(fn, tvm.tir.function.PrimFunc):
            self.fn = tvm.IRModule({"main": fn})
        else:
            self.fn = fn
        self.device = device
        self.total_flops = 0  # Total floating-point operations
        self.total_global_bytes = 0  # Total global memory bytes
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1}  # Block dimensions
        self.loop_stack = []  # Stack to track nested loops
        self.global_buffers = set()  # Set of global memory buffers

    def _analyze_copy(self, call):
        """
        Analyze memory copy operations (e.g., tl.copy).
        Args:
            call: A TVM Call node representing the copy operation.
        """
        src_buffer = call.args[0].args[0].buffer
        dst_buffer = call.args[1].args[0].buffer

        # Determine if the source or destination is a global buffer
        if src_buffer in self.global_buffers:
            buffer_region = call.args[0]
        elif dst_buffer in self.global_buffers:
            buffer_region = call.args[1]
        else:
            return

        # Calculate the number of elements being copied
        elements = 1
        for r in range(2, len(buffer_region.args)):
            elements *= buffer_region.args[r]
        dtype_size = np.dtype(buffer_region.args[0].buffer.dtype).itemsize  # Size of the data type
        bytes_transferred = elements * dtype_size  # Total bytes transferred

        # Account for loop and block dimensions
        loop_product = 1
        for extent in self.loop_stack:
            loop_product *= extent.value if hasattr(extent, "value") else extent
        total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
        total_bytes = bytes_transferred * loop_product * total_blocks
        self.total_global_bytes += total_bytes

    def _analyze_gemm(self, call):
        """
        Analyze matrix multiplication (GEMM) operations (e.g., tl.gemm).
        Args:
            call: A TVM Call node representing the GEMM operation.
        """
        M = call.args[5].value
        N = call.args[6].value
        K = call.args[7].value
        flops_per_call = 2 * M * N * K  # FLOPs for one GEMM operation

        # Account for loop and block dimensions
        loop_product = 1
        for extent in self.loop_stack:
            loop_product *= extent.value if hasattr(extent, "value") else extent
        total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
        self.total_flops += flops_per_call * loop_product * total_blocks

    def ir_pass(self):
        """
        Traverse and transform the IR module to extract performance-related information.
        Returns:
            self: The Analyzer instance.
        """

        def _ftransform(f, mod, ctx):
            # Initialize the set of global buffers
            self.global_buffers = set(f.buffer_map.values())

            def _pre_visit(stmt):
                """
                Pre-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                if isinstance(stmt, tvm.tir.AttrStmt):
                    # Handle thread extent attributes
                    if stmt.attr_key == "thread_extent":
                        iter_var = stmt.node
                        thread_tag = iter_var.thread_tag
                        if thread_tag in self.block_counts:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.block_counts[thread_tag] = extent
                elif isinstance(stmt, tvm.tir.For):
                    # Push loop extent onto the stack
                    self.loop_stack.append(stmt.extent)
                elif isinstance(stmt, tvm.tir.Evaluate):
                    # Handle Evaluate nodes containing calls
                    value = stmt.value
                    if isinstance(value, tvm.tir.Call):
                        if value.op.name == "tl.copy":
                            self._analyze_copy(value)
                        elif value.op.name == "tl.gemm":
                            self._analyze_gemm(value)
                return None

            def _post_visit(stmt):
                """
                Post-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                if isinstance(stmt, tvm.tir.For) and self.loop_stack:
                    self.loop_stack.pop()
                return None

            # Use IR transformation to traverse and modify the function body
            new_body = ir_transform(f.body, _pre_visit, _post_visit)
            return f.with_body(new_body)

        # Apply the custom PrimFunc pass
        tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.fn)
        return self

    def calculate(self) -> AnalysisResult:
        """
        Calculate performance metrics based on the analysis.
        Returns:
            AnalysisResult: The calculated performance metrics.
        """

        def get_peak_tflops(device) -> float | None:
            """
            Get the peak TFLOPS for the target device.
            Args:
                device: The target device information.
            Returns:
                float: The peak TFLOPS.
            """
            arch_key = device.compute_capability[:2]
            if arch_key not in ARCH_CONFIGS:
                logger.info(f"Unsupported compute capability: {device.compute_capability}, theoretical peak tflops will be None")
                return None

            cores_per_sm, default_clock, flops_per_cycle, compute_max_core = ARCH_CONFIGS[arch_key]
            total_cores = compute_max_core * cores_per_sm
            tflops = (total_cores * default_clock * flops_per_cycle) / 1e3
            return round(tflops, 1)

        # Calculate memory bandwidth and peak TFLOPS
        bandwidth_GBps = self.device.bandwidth[1] / 1000
        peak_tflops = get_peak_tflops(self.device)

        # Estimate memory and compute times
        mem_time = self.total_global_bytes / (bandwidth_GBps * 1e9)
        compute_time = self.total_flops / (peak_tflops * 1e12) if peak_tflops else None
        estimated_time = max(mem_time, compute_time) if peak_tflops else mem_time

        # Return the analysis results
        return AnalysisResult(
            total_flops=self.total_flops,
            total_global_bytes=self.total_global_bytes,
            estimated_time=estimated_time,
            expected_tflops=peak_tflops,
            expected_bandwidth_GBps=bandwidth_GBps,
        )

    @classmethod
    def analysis(cls, fn, device):
        """
        Perform a full analysis of the given IR module or PrimFunc.
        Args:
            fn: A TVM IRModule or PrimFunc to analyze.
            device: The target device information.
        Returns:
            AnalysisResult: The calculated performance metrics.
        """
        return cls(fn, device).ir_pass().calculate()
