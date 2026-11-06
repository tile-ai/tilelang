"""Static tile-operation work estimates over native Function and TensorRegion IR."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import tvm_ffi
from tvm_tirx import ir, tirx

from tvm_tirx import sym
from tilelang.analysis._visitor import tensor_scope
from tilelang.language._construction import dtype_of


# (cores per SM, default clock in GHz, FLOPs per cycle, maximum SM count).
ARCH_CONFIGS = {"80": (128, 1.41, 2, 108), "86": (128, 1.70, 2, 84), "89": (128, 2.52, 2, 128)}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Static arithmetic/transfer counts and the existing roofline estimate."""

    total_flops: int
    total_global_bytes: int
    estimated_time: float
    expected_tflops: float | None
    expected_bandwidth_GBps: float


class Analyzer:
    """Estimate dense GEMM work and copy traffic without changing native IR.

    The counts describe tile operators and their enclosing static loop/grid
    multiplicities. They do not measure runtime branch frequencies or achieved
    bandwidth. Calls after lowering have no tile-op cost model in this helper.
    """

    def __init__(self, fn, device):
        self.fn = ir.IRModule.from_expr(fn) if isinstance(fn, tirx.Function) else fn
        if not isinstance(self.fn, ir.IRModule):
            raise TypeError("Analyzer expects a native Function or IRModule")
        self.device = device
        self.total_flops = 0
        self.total_global_bytes = 0
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        self.loop_stack = []
        self.global_buffers = set()
        self.symbolic = sym.Analyzer()

    def _static_int(self, value, description):
        if isinstance(value, int):
            return value
        simplified = self.symbolic.simplify(value)
        if not isinstance(simplified, ir.IntImm):
            raise ValueError(f"Static performance analysis needs a concrete {description}, got {simplified}")
        return int(simplified)

    def _trip_count(self, begin, end, step):
        stride = self._static_int(step, "loop step")
        if stride == 0:
            raise ValueError("Static performance analysis cannot use a zero loop step")
        difference = self._static_int(end - begin, "loop extent")
        # Native For always tests var < end, independently of the step sign.
        # An initially false condition performs no work. A negative step from
        # an initially true condition has no finite mathematical trip count.
        if difference <= 0:
            return 0
        if stride < 0:
            raise ValueError("Static performance analysis requires a finite increasing loop")
        return (difference + stride - 1) // stride

    def _multiplicity(self):
        return math.prod(self.loop_stack) * math.prod(self.block_counts.values())

    def _analyze_copy(self, call):
        if len(call.args) < 2 or not all(isinstance(value, ir.TensorRegion) for value in call.args[:2]):
            raise TypeError("Native copy analysis requires source and destination TensorRegion operands")
        source, destination = call.args[:2]
        # Retain the existing transfer convention: when both ends are global,
        # count the source edge once. Shared/global copies select their actual
        # global edge, whose element type may differ from the shared tensor.
        if tensor_scope(source.source) == "global":
            region = source
        elif tensor_scope(destination.source) == "global":
            region = destination
        else:
            return
        elements = math.prod(self._static_int(axis.extent, "copy extent") for axis in region.region)
        dtype = dtype_of(region.source)
        byte_count = (elements * dtype.bits * dtype.lanes + 7) // 8
        self.total_global_bytes += byte_count * self._multiplicity()

    def _analyze_gemm(self, call):
        # The shared dense GEMM contract keeps M/N/K in slots 5/6/7. Tensor
        # operands are native regions; the final annotation Tuple is excluded.
        dimensions = [self._static_int(value, "GEMM tile dimension") for value in call.args[5:8]]
        if len(dimensions) != 3:
            raise ValueError("Native dense GEMM call is missing its M/N/K operands")
        self.total_flops += 2 * math.prod(dimensions) * self._multiplicity()

    def _visit_for(self, node, visitor):
        self.loop_stack.append(self._trip_count(node.begin, node.end, node.step))
        try:
            visitor.visit(node.body)
        finally:
            self.loop_stack.pop()

    def _visit_region(self, node, visitor):
        previous = self.block_counts
        self.block_counts = previous.copy()
        try:
            if node.op.name == "tl.launch_kernel":
                begins, ends, steps = [value.fields for value in node.args[:3]]
                tags = node.attrs.get("grid_tags", [f"blockIdx.{axis}" for axis in "xyz"[: len(ends)]])
                for tag, begin, end, step in zip(tags, begins, ends, steps):
                    self.block_counts[str(tag)] = self._trip_count(begin, end, step)
            elif node.op.name == "tirx.launch_thread":
                tag = node.args[0].value
                if tag.startswith("blockIdx."):
                    self.block_counts[tag] = self._static_int(node.args[1], "grid extent")
            visitor.visit(node.body)
        finally:
            self.block_counts = previous

    def _visit_call(self, node, visitor):
        if isinstance(node.op, ir.Op) and node.op.name in {"tl.copy", "tl.tileop.copy"}:
            self._analyze_copy(node)
        elif isinstance(node.op, ir.Op) and node.op.name in {"tl.gemm", "tl.tileop.gemm"}:
            self._analyze_gemm(node)
        else:
            visitor.default_visit(node)

    def ir_pass(self):
        """Recompute totals from each function with fresh loop and grid state."""
        self.total_flops = 0
        self.total_global_bytes = 0
        for function in self.fn.functions.values():
            if not isinstance(function, tirx.Function) or function.attrs.get("declaration", False):
                continue
            self.global_buffers = {
                parameter
                for parameter in function.params
                if isinstance(parameter.ty, tirx.TensorType) and tensor_scope(parameter) == "global"
            }
            self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
            self.loop_stack.clear()
            tvm_ffi.structural_visit(
                function.body,
                [(ir.For, self._visit_for), (ir.RegionStmt, self._visit_region), (ir.Call, self._visit_call)],
            )
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
        Perform a full analysis of the given IR module or Function.
        Args:
            fn: A native IRModule or Function to analyze.
            device: The target device information.
        Returns:
            AnalysisResult: The calculated performance metrics.
        """
        return cls(fn, device).ir_pass().calculate()
