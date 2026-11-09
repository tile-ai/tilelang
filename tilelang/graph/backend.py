"""Torch Dynamo entry point for the experimental Ascend graph compiler."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
import logging
import operator
import inspect
import threading

import torch
from torch import fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

from .config import GraphConfig
from .frontend import UnsupportedGraphError, fx_to_relax, validate_inputs
from .runtime import _resolve_graph_context, _resolved_config, build_runner

logger = logging.getLogger(__name__)


@dataclass
class PartitionReport:
    """Observable compilation status; eager fallback is never counted as a kernel."""

    name: str
    compiled: bool = False
    kernel_count: int = 0
    compiled_calls: int = 0
    fallback_calls: int = 0
    fallback_reason: str | None = None
    fusion_decision: object | None = None
    target_backend: str | None = None
    execution_backend: str | None = None


@dataclass
class GraphReport:
    partitions: list[PartitionReport] = field(default_factory=list)
    fallback_reason: str | None = None


_FUNCTION_NAMES = frozenset(
    {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "true_divide",
        "truediv",
        "relu",
        "sum",
        "mean",
        "amax",
        "softmax",
        "matmul",
        "mm",
        "rsqrt",
        "sqrt",
        "exp",
        "square",
        "pow",
    }
)
_METHOD_NAMES = _FUNCTION_NAMES | {"to", "float", "half", "bfloat16", "type_as"}
_MODULE_TYPES = (torch.nn.ReLU, torch.nn.Softmax)
_ALIAS_METHODS = frozenset(
    {"view", "reshape", "flatten", "transpose", "permute", "expand", "as_strided", "detach", "contiguous", "narrow", "select"}
)
_INPLACE_OPERATORS = frozenset(
    {
        operator.iadd,
        operator.isub,
        operator.imul,
        operator.itruediv,
        operator.ifloordiv,
        operator.imod,
        operator.ipow,
        operator.iand,
        operator.ior,
        operator.ixor,
        operator.ilshift,
        operator.irshift,
        operator.imatmul,
        operator.setitem,
    }
)
_PURE_TORCH_BOUNDARIES = frozenset({torch.sin, torch.cos, torch.tanh, torch.sigmoid, torch.nn.functional.gelu, torch.nn.functional.silu})
_PURE_METHOD_BOUNDARIES = frozenset({"sin", "cos", "tanh", "sigmoid"})


def _function_name(target) -> str:
    return getattr(target, "__name__", "").split(".", 1)[0]


def _is_dtype_query(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target is getattr and len(node.args) == 2 and node.args[1] == "dtype"


def _has_unsafe_effects(gm: fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        if node.op == "call_method":
            name = str(node.target)
            if (name.endswith("_") and not name.startswith("__")) or name in _ALIAS_METHODS or name == "__setitem__":
                return True
            if name not in _METHOD_NAMES and name not in _PURE_METHOD_BOUNDARIES:
                return True
        elif node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            # An opaque module may mutate its inputs or internal state. Keep
            # it in its original graph instead of moving partitions across it.
            if type(module) not in _MODULE_TYPES or getattr(module, "inplace", False):
                return True
        elif node.op == "call_function":
            if _is_dtype_query(node):
                continue
            target = node.target
            schema = getattr(target, "_schema", None)
            if schema is not None:
                if schema.is_mutable:
                    return True
                if any(argument.alias_info is not None for argument in schema.returns):
                    return True
                random_tag = getattr(torch.Tag, "nondeterministic_seeded", None)
                if random_tag is not None and random_tag in getattr(target, "tags", ()):
                    return True
            name = _function_name(target)
            if (name.endswith("_") and not name.startswith("__")) or node.kwargs.get("out") is not None:
                return True
            if target in _INPLACE_OPERATORS:
                return True
            if node.kwargs.get("inplace", False):
                return True
            try:
                bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            except (TypeError, ValueError):
                pass
            else:
                if bound.arguments.get("inplace", False):
                    return True
            module = getattr(target, "__module__", "")
            if schema is None and not module.startswith(("torch", "_operator", "operator", "math")):
                return True
            if schema is None and module.startswith("torch") and name not in _FUNCTION_NAMES and target not in _PURE_TORCH_BOUNDARIES:
                # A Torch Python function is not necessarily functional (for
                # example training batch_norm updates running statistics).
                return True
    return False


class _SupportedOperators(OperatorSupportBase):
    def is_node_supported(self, submodules, node: fx.Node) -> bool:
        name = str(node.target) if node.op == "call_method" else _function_name(node.target)
        if name in ("add", "sub", "subtract") and node.kwargs.get("alpha", 1) != 1:
            return False
        if (
            node.op == "call_method"
            and node.target == "to"
            and (
                len(node.args) > 2
                or node.kwargs.get("device") is not None
                or node.kwargs.get("copy", False)
                or node.kwargs.get("memory_format") not in (None, torch.preserve_format)
            )
        ):
            return False
        if node.op == "call_method":
            if node.target in ("pow", "__pow__"):
                return len(node.args) == 2 and isinstance(node.args[1], int | float) and node.args[1] == 2
            return node.target in _METHOD_NAMES
        if node.op == "call_module":
            return type(submodules[node.target]) in _MODULE_TYPES
        if node.op != "call_function":
            return False
        if _is_dtype_query(node):
            return True
        if node.kwargs.get("out") is not None:
            return False
        if _function_name(node.target) == "pow":
            return len(node.args) == 2 and isinstance(node.args[1], int | float) and node.args[1] == 2
        if isinstance(node.target, torch._ops.OpOverload):
            return _function_name(node.target) in _FUNCTION_NAMES
        module = getattr(node.target, "__module__", "")
        return module.startswith(("torch", "_operator", "operator")) and _function_name(node.target) in _FUNCTION_NAMES


def _signature(args):
    result = [torch.is_grad_enabled()]
    for value in args:
        if isinstance(value, torch.Tensor):
            result.append(
                (tuple(value.shape), value.dtype, value.device, tuple(value.stride()), value.storage_offset(), value.requires_grad)
            )
        else:
            result.append((type(value), repr(value)))
    return tuple(result)


class _CompiledPartition(torch.nn.Module):
    def __init__(self, original: fx.GraphModule, config: GraphConfig, report: PartitionReport):
        super().__init__()
        self.original = original
        self.config = config
        self.report = report
        self._signature = None
        self._runner = None
        self._compile_lock = threading.Lock()

    def _prepare(self, args):
        self.report.compiled = False
        self.report.kernel_count = 0
        self.report.fallback_reason = None
        self.report.fusion_decision = None
        try:
            validate_inputs(args)
            mod = fx_to_relax(self.original, args)
            baseline = build_runner(mod, self.config)
        except Exception as error:
            # This region only constructs and compiles IR. Kernel execution,
            # including validation/profiling below, is outside this catch.
            if self.config.strict:
                raise
            self.report.fallback_reason = f"{type(error).__name__}: {error}"
            logger.debug("Ascend partition %s uses Torch: %s", self.report.name, error, exc_info=True)
            return self.original.forward

        from .fusion import GraphValidationError, validate_runner

        try:
            validate_runner(baseline, self.original, args, config=self.config)
        except GraphValidationError as error:
            if self.config.strict:
                raise
            self.report.fallback_reason = str(error)
            return self.original.forward
        selected = baseline
        if self.config.fusion == "profile":
            from .fusion import FusionDecision, SelectedRunner, make_fusion_candidate, select_fusion

            try:
                fused_mod = make_fusion_candidate(mod)
                candidate = build_runner(fused_mod, self.config)
            except Exception as error:
                logger.debug("Ascend fusion candidate compilation failed", exc_info=True)
                selected = SelectedRunner(baseline, FusionDecision(False, f"candidate compilation failed: {type(error).__name__}: {error}"))
            else:
                # Device execution errors deliberately propagate from the
                # selector. A numerical candidate rejection returns baseline.
                selected = select_fusion(baseline, candidate, args, self.config, reference=self.original)
        self.report.kernel_count = selected.kernel_count
        self.report.compiled = self.report.kernel_count > 0
        self.report.fusion_decision = getattr(selected, "fusion_decision", None)
        self.report.target_backend = getattr(selected, "target_backend", None) or self.report.target_backend
        self.report.execution_backend = getattr(selected, "execution_backend", None) or self.report.execution_backend
        return selected

    def forward(self, *args):
        signature = _signature(args)
        with self._compile_lock:
            if signature != self._signature:
                runner = self._prepare(args)
                self._runner = runner
                self._signature = signature
            else:
                runner = self._runner
        if self.report.compiled:
            self.report.compiled_calls += 1
        else:
            self.report.fallback_calls += 1
        return runner(*args)


def _fallback(gm: fx.GraphModule, reason: str):
    report = GraphReport(fallback_reason=reason)

    def run(*args):
        return gm(*args)

    run.graph_report = report
    return run


def make_backend(config: GraphConfig | None = None) -> Callable:
    """Create a Dynamo backend with an immutable compilation configuration."""
    config = config or GraphConfig()

    def backend(gm: fx.GraphModule, example_inputs, **kwargs):
        options = kwargs.pop("options", None)
        if kwargs:
            raise ValueError(f"Unsupported graph backend arguments: {sorted(kwargs)}")
        local_config = replace(config, **options) if options else config
        context = _resolve_graph_context(local_config)
        local_config = _resolved_config(local_config, context)
        if _has_unsafe_effects(gm):
            if local_config.strict:
                raise UnsupportedGraphError("Graph contains mutation, aliases, or opaque effects")
            return _fallback(gm, "Graph contains mutation, aliases, or opaque effects")
        if any(not isinstance(value, torch.Tensor) for value in example_inputs):
            if local_config.strict:
                raise UnsupportedGraphError("Graph requires tensor-only static inputs")
            return _fallback(gm, "Graph requires tensor-only static inputs")
        # Copy graph structure while retaining the user's modules and tensors.
        # Deep-copying the entire module could duplicate NPU parameters.
        partitioned = fx.GraphModule(gm, deepcopy(gm.graph))
        partitioner = CapabilityBasedPartitioner(partitioned, _SupportedOperators(), allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()
        if not partitions:
            if local_config.strict:
                raise UnsupportedGraphError("Graph has no supported Ascend partitions")
            return _fallback(gm, "Graph has no supported Ascend partitions")
        partitioned = partitioner.fuse_partitions(partitions, prefix="tilelang_partition_")
        report = GraphReport()
        for node in partitioned.graph.nodes:
            if node.op == "call_module" and str(node.target).startswith("tilelang_partition_"):
                name = str(node.target)
                status = PartitionReport(name, target_backend=context.module.name, execution_backend=context.execution_backend.name)
                report.partitions.append(status)
                partitioned.set_submodule(name, _CompiledPartition(partitioned.get_submodule(name), local_config, status))

        def run(*args):
            return partitioned(*args)

        run.graph_report = report
        return run

    return backend


tilelang_backend = make_backend()


def _require_ascend_backend(config: GraphConfig | None = None) -> None:
    _resolve_graph_context(config or GraphConfig())
