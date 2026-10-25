from __future__ import annotations
import math
import tvm_ffi
from tvm_tirx import ir, tirx
from dataclasses import dataclass
from tilelang._compat.tensor import tensor_parameter_map
from tilelang._compat.stmt_functor import ir_transform
import logging

# Configuration for different hardware architectures.
# Each entry contains: (cores per SM, default clock (GHz), FLOPs per cycle, max SM count)
ARCH_CONFIGS = {"80": (128, 1.41, 2, 108), "86": (128, 1.70, 2, 84), "89": (128, 2.52, 2, 128)}

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
        if isinstance(fn, tirx.Function):
            self.fn = ir.IRModule({"main": fn})
        else:
            self.fn = fn
        self.device = device
        self.total_flops = 0  # Total floating-point operations
        self.total_global_bytes = 0  # Total global memory bytes
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}  # Block dimensions
        self.loop_stack = []  # Stack to track nested loops
        self.global_buffers = set()  # Set of global memory buffers

    def _analyze_copy(self, call):
        """Count global traffic from the native source and destination regions."""
        normalize = tvm_ffi.get_global_func("tl.NormalizeToBufferRegion")
        repetitions = math.prod(self.loop_stack) * math.prod(self.block_counts.values())
        for operand in call.args[:2]:
            region = normalize(operand)
            if region.source not in self.global_buffers:
                continue
            elements = math.prod(int(bound.extent) for bound in region.region)
            dtype = region.source.dtype
            self.total_global_bytes += ((elements * dtype.bits * dtype.lanes + 7) // 8) * repetitions

    def _analyze_gemm(self, call):
        """Count multiply-adds using the tile operation's explicit M/N/K operands."""
        dimensions = [int(dimension) for dimension in call.args[5:8]]
        repetitions = math.prod(self.loop_stack) * math.prod(self.block_counts.values())
        self.total_flops += 2 * math.prod(dimensions) * repetitions

    def ir_pass(self):
        """Traverse native functions, launch regions, and stepped loops."""
        self.total_flops = 0
        self.total_global_bytes = 0
        for function in self.fn.functions.values():
            if not isinstance(function, tirx.Function) or function.body is None:
                continue
            self.global_buffers = set(tensor_parameter_map(function).values())
            self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
            self.loop_stack.clear()
            block_stack = []

            def launch_axis(statement):
                if isinstance(statement, ir.RegionStmt) and statement.op.name in ("tirx.launch_thread", "tl.launch_grid"):
                    axis = statement.args[0].value
                    if axis in self.block_counts:
                        return axis
                return None

            def pre_visit(statement, block_stack=block_stack):
                axis = launch_axis(statement)
                if axis is not None:
                    block_stack.append((axis, self.block_counts[axis]))
                    self.block_counts[axis] = int(statement.args[1])
                elif isinstance(statement, ir.For):
                    extent = len(range(int(statement.begin), int(statement.end), int(statement.step)))
                    self.loop_stack.append(extent)
                elif isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call):
                    call = statement.value
                    if isinstance(call.op, ir.Op):
                        if call.op.name in ("tl.copy", "tl.tileop.copy"):
                            self._analyze_copy(call)
                        elif call.op.name in ("tl.gemm", "tl.tileop.gemm"):
                            self._analyze_gemm(call)
                return None

            def post_visit(statement, block_stack=block_stack):
                if launch_axis(statement) is not None:
                    axis, previous = block_stack.pop()
                    self.block_counts[axis] = previous
                elif isinstance(statement, ir.For):
                    self.loop_stack.pop()
                return None

            ir_transform(function.body, pre_visit, post_visit)
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
