"""Metal execution through ``torch.mps.compile_shader``.

The launch plan is derived from the lowered host module rather than from the
public signature. Host/device splitting orders device parameters by its own
rules and drops parameters the kernel never reads, so the MSL buffer order is
not the declared parameter order. Every kernel call site of the host program
is launched in program order, each MSL buffer slot is bound to the public
parameter the host passes there, and runtime scalar parameters are packed into
the kernel's argument struct exactly as the Metal code generator lays it out.

The user-facing call keeps the declared parameter order; parameters listed in
``out_idx`` are allocated by the adapter and returned, as with other backends.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tvm import tirx

from tilelang import tvm as tvm
from tilelang.engine.param import KernelParam

from ..base import BaseKernelAdapter

_CALL_PACKED = "tirx.tvm_call_packed"
_STRUCT_GET = "tirx.tvm_struct_get"
_ADD_BYTE_OFFSET = "tirx.handle_add_byte_offset"
_FFI_VALUE_FIELD = 15
_DATA_FIELD = 1
_DYN_SHARED_TAG = "tirx.use_dyn_shared_memory"
_LAUNCH_AXES = {
    "blockIdx.x": ("grid", 0),
    "blockIdx.y": ("grid", 1),
    "blockIdx.z": ("grid", 2),
    "threadIdx.x": ("block", 0),
    "threadIdx.y": ("block", 1),
    "threadIdx.z": ("block", 2),
}
# One eight-byte struct member per runtime scalar, matching codegen_metal.cc:
# 32-bit values occupy the low four bytes of a two-element array, 64-bit
# values the whole member.
_SCALAR_FORMATS = {
    "int32": "<i4x",
    "uint32": "<I4x",
    "float32": "<f4x",
    "int64": "<q",
    "uint64": "<Q",
    "float64": "<d",
}
_SCALAR_BYTES = 8


class MetalLaunchPlanError(ValueError):
    """The lowered program cannot be launched through the torch Metal adapter."""


@dataclass(frozen=True)
class MetalLaunch:
    """One device kernel call site of the host program.

    ``buffers`` lists, per MSL buffer slot, the index of the public parameter
    bound there. ``scalars`` lists the public parameter index of every runtime
    scalar in the order of the kernel's argument struct.
    """

    symbol: str
    buffers: tuple[int, ...]
    scalars: tuple[int, ...]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]

    @property
    def threads(self) -> tuple[int, int, int]:
        return tuple(g * b for g, b in zip(self.grid, self.block))


def _is_call_to(expr: Any, op_name: str) -> bool:
    return isinstance(expr, tirx.Call) and expr.op == tvm.ir.Op.get(op_name)


def _packed_slot(expr: Any, bindings: dict[Any, Any], args_var: tirx.Var, depth: int = 0) -> int | None:
    """Trace a host call-site argument back to its packed-ABI argument slot.

    The host entry unpacks ``args`` as ``X_handle = tvm_struct_get(args, i, 15)``
    (optionally byte-offset adjusted through a ``Select``) and dereferences the
    data pointer as ``X = tvm_struct_get(X_handle, 0, 1)``; scalars are read
    through a ``Cast`` of the same slot. Returns ``None`` when the argument is
    not a public parameter of the function.
    """
    if depth > 32:
        return None
    if isinstance(expr, tirx.Var):
        if expr.same_as(args_var):
            return None
        value = bindings.get(expr)
        return None if value is None else _packed_slot(value, bindings, args_var, depth + 1)
    if isinstance(expr, tirx.Cast):
        return _packed_slot(expr.value, bindings, args_var, depth + 1)
    if isinstance(expr, tirx.Select):
        first = _packed_slot(expr.true_value, bindings, args_var, depth + 1)
        second = _packed_slot(expr.false_value, bindings, args_var, depth + 1)
        return first if first is not None and first == second else None
    if _is_call_to(expr, _STRUCT_GET) and len(expr.args) == 3:
        source, index, field = expr.args
        if not isinstance(index, tirx.IntImm) or not isinstance(field, tirx.IntImm):
            return None
        if isinstance(source, tirx.Var) and source.same_as(args_var):
            return int(index) if int(field) == _FFI_VALUE_FIELD else None
        if int(index) == 0 and int(field) == _DATA_FIELD:
            return _packed_slot(source, bindings, args_var, depth + 1)
        return None
    if _is_call_to(expr, _ADD_BYTE_OFFSET) and len(expr.args) == 2:
        return _packed_slot(expr.args[0], bindings, args_var, depth + 1)
    return None


def _collect_call_sites(stmt: Any, bindings: dict[Any, Any], sites: list[tuple[str, list[Any], dict[Any, Any]]]) -> None:
    """Record packed calls in program order; reject host control flow."""
    if isinstance(stmt, tirx.SeqStmt):
        for item in stmt.seq:
            _collect_call_sites(item, bindings, sites)
    elif isinstance(stmt, tirx.AttrStmt):
        _collect_call_sites(stmt.body, bindings, sites)
    elif isinstance(stmt, tirx.Bind):
        bindings[stmt.var] = stmt.value
    elif isinstance(stmt, tirx.Evaluate):
        call = stmt.value
        if _is_call_to(call, _CALL_PACKED) and call.args and isinstance(call.args[0], tirx.StringImm):
            sites.append((call.args[0].value, list(call.args[1:]), dict(bindings)))
    elif isinstance(stmt, (tirx.AssertStmt, tirx.DeclBuffer)):
        return
    else:
        raise MetalLaunchPlanError(
            f"host statement {type(stmt).__name__} is not supported by the torch Metal adapter; "
            "kernel launches must form a straight-line host program"
        )


def _host_entry(host_mod: tvm.IRModule) -> tirx.PrimFunc:
    functions = list(host_mod.functions.values())
    entries = [f for f in functions if f.attrs is not None and f.attrs.get("tirx.is_entry_func")]
    if len(entries) == 1:
        return entries[0]
    if not entries and len(functions) == 1:
        return functions[0]
    raise MetalLaunchPlanError(f"expected one host entry function, found {len(entries) or len(functions)}")


def plan_metal_launches(
    params: Sequence[KernelParam],
    host_mod: tvm.IRModule,
    device_mod: tvm.IRModule,
) -> tuple[MetalLaunch, ...]:
    """Derive the ordered kernel launches and their bindings from the host program."""
    entry = _host_entry(host_mod)
    args_vars = [param for param in entry.params if param.name == "args"]
    if len(args_vars) != 1:
        raise MetalLaunchPlanError("host entry function has no packed 'args' parameter")
    args_var = args_vars[0]
    sites: list[tuple[str, list[Any], dict[Any, Any]]] = []
    _collect_call_sites(entry.body, {}, sites)

    device_functions = {gv.name_hint: func for gv, func in device_mod.functions.items()}
    launches: list[MetalLaunch] = []
    for symbol, call_args, bindings in sites:
        func = device_functions.get(symbol)
        if func is None:
            continue
        count = len(func.params)
        if len(call_args) < count:
            raise MetalLaunchPlanError(f"call site of '{symbol}' passes {len(call_args)} arguments for {count} device parameters")
        buffers: list[int] = []
        scalars: list[int] = []
        for device_param, call_arg in zip(func.params, call_args[:count]):
            slot = _packed_slot(call_arg, bindings, args_var)
            if slot is None or slot >= len(params):
                raise MetalLaunchPlanError(
                    f"device parameter '{device_param.name}' of '{symbol}' is bound to '{call_arg}', "
                    "which is not a public parameter of the function; generated buffers and "
                    "runtime symbols are not supported by the torch Metal adapter"
                )
            public = params[slot]
            if device_param.dtype == "handle":
                if public.is_scalar():
                    raise MetalLaunchPlanError(f"device buffer '{device_param.name}' is bound to scalar parameter {slot}")
                if scalars:
                    raise MetalLaunchPlanError(f"device buffer '{device_param.name}' follows scalar parameters in '{symbol}'")
                buffers.append(slot)
            else:
                if not public.is_scalar() or str(public.dtype) != str(device_param.dtype):
                    raise MetalLaunchPlanError(
                        f"device scalar '{device_param.name}' ({device_param.dtype}) is bound to parameter {slot} ({public.dtype})"
                    )
                if str(device_param.dtype) not in _SCALAR_FORMATS:
                    raise MetalLaunchPlanError(f"scalar parameter dtype {device_param.dtype} cannot be packed for Metal")
                scalars.append(slot)

        tags = func.attrs.get("tirx.kernel_launch_params") if func.attrs is not None else None
        if tags is None:
            raise MetalLaunchPlanError(f"device function '{symbol}' declares no launch parameters")
        tags = [str(tag) for tag in tags]
        launch_args = call_args[count:]
        if len(launch_args) != len(tags):
            raise MetalLaunchPlanError(
                f"call site of '{symbol}' passes {len(launch_args)} launch arguments for {len(tags)} launch parameters"
            )
        extents = {"grid": [1, 1, 1], "block": [1, 1, 1]}
        for tag, value in zip(tags, launch_args):
            if tag == _DYN_SHARED_TAG:
                continue
            if tag not in _LAUNCH_AXES:
                raise MetalLaunchPlanError(f"launch parameter '{tag}' of '{symbol}' is not a Metal grid or threadgroup axis")
            if not isinstance(value, tirx.IntImm):
                raise MetalLaunchPlanError(f"launch extent '{tag}' of '{symbol}' is '{value}'; only static launch geometry is supported")
            kind, axis = _LAUNCH_AXES[tag]
            extents[kind][axis] = int(value)
        launches.append(
            MetalLaunch(
                symbol=symbol,
                buffers=tuple(buffers),
                scalars=tuple(scalars),
                grid=tuple(extents["grid"]),
                block=tuple(extents["block"]),
            )
        )
    if not launches:
        raise MetalLaunchPlanError("host program launches no device kernel")
    return tuple(launches)


def _static_shape(param: KernelParam) -> tuple[int, ...] | None:
    shape = []
    for dim in param.shape:
        if isinstance(dim, tirx.IntImm):
            shape.append(int(dim))
        elif isinstance(dim, int):
            shape.append(dim)
        else:
            return None
    return tuple(shape)


def _describe_shape(param: KernelParam) -> str:
    return "(" + ", ".join(str(int(d)) if isinstance(d, (int, tirx.IntImm)) else str(d) for d in param.shape) + ")"


@dataclass(frozen=True)
class _TensorContract:
    """Per-parameter facts checked on every launch, resolved once at adapter creation."""

    index: int
    dtype: Any
    shape: tuple[int, ...] | None
    static_dims: tuple[tuple[int, int], ...]
    rank: int
    described: str

    @classmethod
    def of(cls, index: int, param: KernelParam) -> _TensorContract:
        static_dims = []
        for axis, dim in enumerate(param.shape):
            if isinstance(dim, tirx.IntImm):
                static_dims.append((axis, int(dim)))
            elif isinstance(dim, int):
                static_dims.append((axis, dim))
        return cls(
            index=index,
            dtype=param.torch_dtype(),
            shape=_static_shape(param),
            static_dims=tuple(static_dims),
            rank=len(param.shape),
            described=_describe_shape(param),
        )

    def check(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"argument {self.index} must be a tensor, got {type(value).__name__}")
        if value.device.type != "mps":
            raise ValueError(f"argument {self.index} must be on the mps device, got {value.device}")
        if value.dtype != self.dtype:
            raise TypeError(f"argument {self.index} has dtype {value.dtype}, expected {self.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"argument {self.index} must be contiguous")
        shape = value.shape
        if self.shape is not None:
            if tuple(shape) != self.shape:
                raise ValueError(f"argument {self.index} has shape {tuple(shape)}, expected {self.described}")
            return
        if len(shape) != self.rank or any(shape[axis] != dim for axis, dim in self.static_dims):
            raise ValueError(f"argument {self.index} has shape {tuple(shape)}, expected {self.described}")


def _pack_scalars(values: Sequence[tuple[str, Any]]) -> torch.Tensor:
    payload = bytearray(_SCALAR_BYTES * len(values))
    for index, (dtype, value) in enumerate(values):
        struct.pack_into(_SCALAR_FORMATS[dtype], payload, index * _SCALAR_BYTES, value)
    return torch.frombuffer(payload, dtype=torch.uint8).to("mps")


class MetalKernelAdapter(BaseKernelAdapter):
    """Launch a lowered Metal module through ``torch.mps.compile_shader``."""

    launches: tuple[MetalLaunch, ...]

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        kernel_global_source: str | None = None,
        verbose: bool = False,
    ):
        if host_mod is None or device_mod is None:
            raise MetalLaunchPlanError("the torch Metal adapter needs the lowered host and device modules")
        if kernel_global_source is None:
            raise MetalLaunchPlanError("the torch Metal adapter needs the generated Metal source")
        self.kernel_global_source = kernel_global_source
        self.verbose = verbose
        self.launches = plan_metal_launches(params, host_mod, device_mod)
        super().__init__(func_or_mod, result_idx=result_idx, params=params)

    _shader_library: Any = None
    _shader_kernels: dict[str, Any] | None = None

    @property
    def thread_execution_width(self) -> int:
        """SIMD-group width of the compiled pipelines."""
        widths = {int(kernel.thread_execution_width) for kernel in self._kernels().values()}
        if len(widths) != 1:
            raise RuntimeError(f"compiled Metal pipelines report differing execution widths {sorted(widths)}")
        return widths.pop()

    @property
    def max_total_threads_per_threadgroup(self) -> int:
        """Smallest threadgroup limit across the compiled pipelines."""
        return min(int(kernel.max_threads_per_threadgroup) for kernel in self._kernels().values())

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        if kernel_only:
            # Return just the kernel function body, stripping Metal
            # module-level boilerplate (includes, structs, etc.).
            idx = self.kernel_global_source.find("kernel void ")
            if idx >= 0:
                return self.kernel_global_source[idx:]
        return self.kernel_global_source

    def _kernels(self) -> dict[str, Any]:
        if self._shader_kernels is None:
            self._shader_library = torch.mps.compile_shader(self.kernel_global_source)
            kernels = {}
            for launch in self.launches:
                kernel = getattr(self._shader_library, launch.symbol)
                if math.prod(launch.block) > int(kernel.max_threads_per_threadgroup):
                    raise MetalLaunchPlanError(
                        f"'{launch.symbol}' launches {math.prod(launch.block)} threads per threadgroup; "
                        f"the compiled pipeline allows {int(kernel.max_threads_per_threadgroup)}"
                    )
                kernels[launch.symbol] = kernel
            self._shader_kernels = kernels
        return self._shader_kernels

    def _convert_torch_func(self) -> Callable:
        """Build the launcher; everything derivable from the plan is resolved here, not per call."""
        kernels = self._kernels()
        params = list(self.params)
        count = len(params)
        result_idx = list(self.result_idx)
        inputs = [index for index in range(count) if index not in result_idx]
        expected = len(inputs)
        # Argument slot i of a call is public parameter inputs[i].
        contracts: list[tuple[int, _TensorContract | None]] = [
            (index, None if params[index].is_scalar() else _TensorContract.of(index, params[index])) for index in inputs
        ]
        outputs_plan = []
        for index in result_idx:
            shape = _static_shape(params[index])
            if shape is None:
                raise MetalLaunchPlanError(f"output parameter {index} has a dynamic shape {_describe_shape(params[index])}")
            outputs_plan.append((index, shape, params[index].torch_dtype()))
        launch_plan = [
            (
                kernels[launch.symbol],
                launch.buffers,
                tuple((index, str(params[index].dtype)) for index in launch.scalars),
                list(launch.threads),
                list(launch.block),
            )
            for launch in self.launches
        ]

        def launcher(*args: Any) -> Any:
            if len(args) != expected:
                raise TypeError(f"kernel expects {expected} arguments, got {len(args)}")
            values: list[Any] = [None] * count
            for (index, contract), value in zip(contracts, args):
                if contract is None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError(f"argument {index} must be a scalar, got {type(value).__name__}")
                else:
                    contract.check(value)
                values[index] = value
            outputs = []
            for index, shape, dtype in outputs_plan:
                tensor = torch.empty(shape, dtype=dtype, device="mps")
                values[index] = tensor
                outputs.append(tensor)
            for kernel, buffers, scalars, threads, block in launch_plan:
                bound = [values[index] for index in buffers]
                if scalars:
                    bound.append(_pack_scalars([(dtype, values[index]) for index, dtype in scalars]))
                kernel(*bound, threads=threads, group_size=block)
            if not outputs:
                return None
            return outputs[0] if len(outputs) == 1 else outputs

        return launcher
