"""A small Relax executor whose GM tensor allocations belong to Torch.

This executes a statically legalized graph. It deliberately does not use a VM
storage allocator or convert live NPU tensors through TVM's DeviceAPI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from tilelang import tvm
from tvm import relax, tirx

from .config import GraphConfig
from .frontend import UnsupportedGraphError


def _resolve_graph_context(config: GraphConfig):
    """Resolve the target variant and its existing execution backend together.

    PTO is an alias/variant of the Ascend target, so resolution must precede
    target inspection. This preflight performs no device or toolchain probing.
    Kernel construction still goes through TileLang's shared compiler/cache.
    """
    from tvm.target import Target
    from tilelang.backend.module import create_backend_context

    if "ascend" not in Target.list_kinds() or tvm.get_global_func("tl.TargetIsAscend", allow_missing=True) is None:
        raise UnsupportedGraphError("Ascend graph compilation is unavailable in this TileLang build; rebuild with USE_ASCEND=ON")
    context = create_backend_context(config.target, execution_backend=config.execution_backend)
    pair = (context.module.name, context.execution_backend.name)
    codegen = {
        ("ascend", "tvm_ffi"): "target.build.tilelang_ascend",
        ("pto", "pto"): "target.build.tilelang_pto_without_compile",
    }.get(pair)
    if context.target.kind.name != "ascend" or codegen is None:
        raise UnsupportedGraphError("The graph backend supports AscendC with tvm_ffi or PTO with pto execution")
    if tvm.get_global_func(codegen, allow_missing=True) is None:
        raise UnsupportedGraphError(
            f"Graph backend {context.module.name!r} requires native codegen {codegen!r}; rebuild TileLang with USE_ASCEND=ON"
        )
    return context


def _resolved_config(config: GraphConfig, context) -> GraphConfig:
    """Freeze preflight selection while preserving the user's other options."""
    return replace(config, target=str(context.target), execution_backend=context.execution_backend.name)


def _shape(shape) -> tuple[int, ...]:
    if not isinstance(shape, relax.ShapeExpr):
        raise UnsupportedGraphError("Graph tensors require static shapes")
    result = []
    for extent in shape.values:
        if not isinstance(extent, tirx.IntImm) or extent.value <= 0:
            raise UnsupportedGraphError("Graph tensors require positive static dimensions")
        result.append(int(extent.value))
    return tuple(result)


def _torch_dtype(dtype: str) -> torch.dtype:
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype not in dtype_map:
        raise UnsupportedGraphError(f"Unsupported graph tensor dtype {dtype}")
    return dtype_map[dtype]


@dataclass(frozen=True)
class _TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_struct_info(cls, info):
        if not isinstance(info, relax.TensorStructInfo):
            raise UnsupportedGraphError("Graph kernel arguments and outputs must be tensors")
        return cls(_shape(info.shape), _torch_dtype(info.dtype))


def _compile_kernel(func: tirx.PrimFunc, config: GraphConfig) -> Callable:
    import tilelang
    from tilelang.ascend.graph.schedule import schedule_primfunc

    scheduled = schedule_primfunc(func, config)
    return tilelang.compile(scheduled, target=config.target, execution_backend=config.execution_backend)


class TorchGraphRunner:
    """Execute precompiled TIR kernels with fresh Torch-owned outputs.

    Compilation is eager when constructing the runner. Runtime failures from a
    kernel are propagated; the executor never retries after a device error.
    """

    def __init__(self, mod: tvm.IRModule, config: GraphConfig, kernel_compiler: Callable | None = None):
        self.mod = mod
        # An injected compiler also supports pure CPU executor tests on builds
        # without Ascend. Native compilation always resolves and validates the
        # existing target/execution pair before dispatching any kernel.
        self.backend_context = _resolve_graph_context(config) if kernel_compiler is None else None
        self.config = _resolved_config(config, self.backend_context) if self.backend_context is not None else config
        self.target_backend = self.backend_context.module.name if self.backend_context is not None else None
        self.execution_backend = self.backend_context.execution_backend.name if self.backend_context is not None else None
        self.main = mod["main"]
        if not isinstance(self.main, relax.Function):
            raise UnsupportedGraphError("A graph requires a Relax main function")

        def check_return(info):
            if isinstance(info, relax.TupleStructInfo):
                for field in info.fields:
                    check_return(field)
            else:
                _TensorSpec.from_struct_info(info)

        check_return(self.main.ret_struct_info)
        self.input_specs = tuple(_TensorSpec.from_struct_info(param.struct_info) for param in self.main.params)
        self.kernels = {}
        self.output_specs = {}
        self.constants = {}
        from .fusion import has_reduction, kernel_count, output_reduction_flags

        self.has_reduction = has_reduction(mod)
        self.kernel_count = kernel_count(mod)
        self.output_reductions = output_reduction_flags(mod)
        compiler = kernel_compiler or _compile_kernel
        pending_kernels = {}
        sequences = []

        def check(expr):
            if isinstance(expr, relax.Call):
                if not isinstance(expr.op, tvm.ir.Op) or expr.op.name != "relax.call_tir":
                    raise UnsupportedGraphError(f"Unsupported graph runtime call {expr.op}")
                if len(expr.args) != 2 or not isinstance(expr.args[0], tvm.ir.GlobalVar):
                    raise UnsupportedGraphError("Graph call_tir requires tensor-only arguments")
                if not isinstance(expr.args[1], relax.Tuple):
                    raise UnsupportedGraphError("Graph call_tir arguments must be a tuple")
                name = expr.args[0].name_hint
                func = mod[expr.args[0]]
                if not isinstance(func, tirx.PrimFunc):
                    raise UnsupportedGraphError("Graph call_tir must reference a PrimFunc")
                info = expr.struct_info
                infos = list(info.fields) if isinstance(info, relax.TupleStructInfo) else [info]
                specs = tuple(_TensorSpec.from_struct_info(item) for item in infos)
                if len(func.params) != len(expr.args[1].fields) + len(specs):
                    raise UnsupportedGraphError("Graph kernel ABI must contain input buffers followed by output buffers")
                if any(param not in func.buffer_map for param in func.params):
                    raise UnsupportedGraphError("Graph kernels with scalar parameters are not supported")
                self.output_specs[expr] = specs
                if name not in pending_kernels:
                    if func.attrs is None or "global_symbol" not in func.attrs:
                        func = func.with_attr("global_symbol", name)
                    pending_kernels[name] = func
            elif isinstance(expr, relax.Constant):
                _TensorSpec.from_struct_info(expr.struct_info)
                self.constants[expr] = torch.from_dlpack(expr.data)
            elif isinstance(expr, relax.ShapeExpr):
                _shape(expr)
            elif isinstance(expr, relax.SeqExpr):
                sequences.append(expr)
                for block in expr.blocks:
                    for binding in block.bindings:
                        if not isinstance(binding, relax.VarBinding):
                            raise UnsupportedGraphError("Graph runtime does not support MatchCast bindings")
            elif not isinstance(expr, relax.Var | relax.Tuple | relax.TupleGetItem | relax.Function | tvm.ir.Op | tvm.ir.GlobalVar):
                raise UnsupportedGraphError(f"Unsupported graph runtime expression {type(expr).__name__}")

        relax.analysis.post_order_visit(self.main.body, check)
        # Compile outside the FFI visitor callback: this preserves the Python
        # exception type and keeps toolchain invocation outside IR traversal.
        for name, func in pending_kernels.items():
            kernel = compiler(func, self.config)
            actual_context = getattr(kernel, "backend_context", None)
            if actual_context is not None:
                actual_pair = (actual_context.module.name, actual_context.execution_backend.name)
                if self.backend_context is not None and actual_pair != (self.target_backend, self.execution_backend):
                    raise UnsupportedGraphError("Graph kernels resolved inconsistent target and execution backends")
                self.backend_context = actual_context
                self.target_backend, self.execution_backend = actual_pair
            self.kernels[name] = kernel
        self._use_counts = {}
        self._binding_dependencies = {}
        self._return_dependencies = {}
        for sequence in sequences:
            for block in sequence.blocks:
                for binding in block.bindings:
                    dependencies = tuple(relax.analysis.free_vars(binding.value))
                    self._binding_dependencies[binding.var] = dependencies
                    for var in dependencies:
                        self._use_counts[var] = self._use_counts.get(var, 0) + 1
            dependencies = tuple(relax.analysis.free_vars(sequence.body))
            self._return_dependencies[sequence] = dependencies
            for var in dependencies:
                self._use_counts[var] = self._use_counts.get(var, 0) + 1

    def __call__(self, *args):
        if len(args) != len(self.input_specs):
            raise ValueError(f"Graph expects {len(self.input_specs)} inputs, got {len(args)}")
        if not args:
            raise UnsupportedGraphError("Graph execution requires at least one tensor input")
        if not isinstance(args[0], torch.Tensor):
            raise ValueError("Graph inputs must be tensors")
        device = args[0].device
        for value, spec in zip(args, self.input_specs):
            if (
                not isinstance(value, torch.Tensor)
                or value.device != device
                or tuple(value.shape) != spec.shape
                or value.dtype != spec.dtype
                or not value.is_contiguous()
                or value.storage_offset() != 0
                or (value.requires_grad and torch.is_grad_enabled())
            ):
                raise ValueError("Graph input does not match its compiled shape, dtype, device, or layout")
        env = dict(zip(self.main.params, args))
        constants = {}
        remaining_uses = dict(self._use_counts)

        def release(dependencies):
            for var in dependencies:
                remaining_uses[var] -= 1
                if remaining_uses[var] == 0:
                    env.pop(var, None)

        def evaluate(expr):
            if isinstance(expr, relax.Var):
                return env[expr]
            if isinstance(expr, relax.Constant):
                if expr not in constants:
                    constants[expr] = self.constants[expr].to(device=device)
                return constants[expr]
            if isinstance(expr, relax.Tuple):
                return tuple(evaluate(field) for field in expr.fields)
            if isinstance(expr, relax.TupleGetItem):
                return evaluate(expr.tuple_value)[expr.index]
            if isinstance(expr, relax.Call):
                inputs = evaluate(expr.args[1])
                outputs = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in self.output_specs[expr])
                self.kernels[expr.args[0].name_hint](*inputs, *outputs)
                return outputs if isinstance(expr.struct_info, relax.TupleStructInfo) else outputs[0]
            if isinstance(expr, relax.SeqExpr):
                for block in expr.blocks:
                    for binding in block.bindings:
                        env[binding.var] = evaluate(binding.value)
                        release(self._binding_dependencies[binding.var])
                        if not remaining_uses.get(binding.var, 0):
                            env.pop(binding.var, None)
                result = evaluate(expr.body)
                release(self._return_dependencies[expr])
                return result
            raise UnsupportedGraphError(f"Unsupported graph expression {type(expr).__name__}")

        # The existing kernel adapter supplies Torch's current NPU stream to
        # TVM-FFI. Keep raw Torch tensors at this boundary instead of wrapping
        # them in TVM tensors, which would bypass that exchange hook.
        context = torch.npu.device(device) if device.type == "npu" else torch.device(device)
        with context:
            if device.type == "npu":
                stream = torch.npu.current_stream(device)
                for value in args:
                    # A custom FFI launch bypasses Torch's dispatcher. Record
                    # consumption on this stream so an external input can be
                    # freed after graph return without premature allocator reuse.
                    # Producer/consumer stream ordering still follows Torch's
                    # usual explicit wait_stream/event protocol.
                    value.record_stream(stream)
            return evaluate(self.main.body)


def build_runner(
    mod: tvm.IRModule,
    config: GraphConfig | None = None,
    *,
    kernel_compiler: Callable[[tirx.PrimFunc, GraphConfig], Callable] | None = None,
) -> TorchGraphRunner:
    """Compile a legalized Relax graph for Torch-owned tensor execution."""
    return TorchGraphRunner(mod, config or GraphConfig(), kernel_compiler)
