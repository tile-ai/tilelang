"""Shared helpers for preparing TVM-FFI callable ABIs."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index

import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir import prim
from tvm_tirx.tirx.layout import Axis

from tilelang._analyzer import create_analyzer

# Declarative attribute: which parameters are outputs (mirrors out_idx).
OUT_IDX_ATTR = "tilelang_out_idx"
# ABI decision attribute: set only when the resolved execution backend
# declares supports_callee_allocated_outputs. MakePackedAPI and the TVM-FFI
# adapter both read this attribute, so the compile-time decision has a single
# source of truth instead of each side re-deriving it from the target.
CALLEE_ALLOCATED_OUTPUTS_ATTR = "tilelang_callee_allocated_outputs"


def _normalize_output_indices(output_indices: list[int], num_params: int) -> list[int]:
    normalized = []
    for raw_index in output_indices:
        index = int(raw_index)
        if index < 0:
            index += num_params
        if index < 0 or index >= num_params:
            raise ValueError(f"out_idx index {raw_index} is out of range for a function with {num_params} parameters")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"out_idx contains duplicate tensor indices: {output_indices}")
    return normalized


def _stamp_callee_allocated_decision(func: tirx.Function) -> tirx.Function:
    if func.attrs is not None and CALLEE_ALLOCATED_OUTPUTS_ATTR in func.attrs:
        return func
    return func.with_attr(CALLEE_ALLOCATED_OUTPUTS_ATTR, 1)


def prepare_tvm_ffi_callee_allocated_outputs(
    func: tirx.Function,
    out_idx: list[int] | int | None,
    *,
    supports_callee_allocated_outputs: bool = False,
) -> tuple[tirx.Function, list[int] | None]:
    """Resolve output indices and expose them to TVM-FFI lowering.

    When the resolved execution backend declares
    ``supports_callee_allocated_outputs``, the ABI decision is recorded on the
    derived Function so lowering and the runtime adapter both honor it.
    Otherwise ``tilelang_out_idx`` stays purely declarative and the kernel
    keeps the caller-preallocated output path.
    """
    requested_indices = None if out_idx is None else ([out_idx] if isinstance(out_idx, int) else list(out_idx))
    attr_indices = None
    if func.attrs is not None and OUT_IDX_ATTR in func.attrs:
        attr_indices = [int(index) for index in func.attrs[OUT_IDX_ATTR]]

    if attr_indices is not None:
        if requested_indices is not None:
            num_params = len(func.params)
            if _normalize_output_indices(requested_indices, num_params) != _normalize_output_indices(attr_indices, num_params):
                raise ValueError("out_idx does not match the Function's tilelang_out_idx attribute")
        if attr_indices and supports_callee_allocated_outputs:
            func = _stamp_callee_allocated_decision(func)
        return func, attr_indices

    output_indices = requested_indices or []
    if not output_indices:
        return func, None
    _normalize_output_indices(output_indices, len(func.params))
    func = func.with_attr(OUT_IDX_ATTR, output_indices)
    if supports_callee_allocated_outputs:
        func = _stamp_callee_allocated_decision(func)
    return func, output_indices


@dataclass(frozen=True)
class TensorParameter:
    """A public tensor's logical shape and affine storage contract.

    These expressions refer to the original Function variables by identity.
    They are recomputed from the serialized Function when a cache is loaded;
    names and independently serialized metadata never join two shape graphs.
    """

    index: int
    variable: ir.Var
    shape: tuple[ir.Expr, ...]
    strides: tuple[ir.Expr, ...]
    element_offset: ir.Expr

    @property
    def dtype(self):
        return self.variable.ty.dtype.dtype


def _tensor_parameter(index: int, variable: ir.Var) -> TensorParameter:
    ty = variable.ty
    layout = ty.effective_layout()
    if (
        not isinstance(layout, tirx.TileLayout)
        or layout.replica
        or len(layout.shard) != len(ty.shape)
        or any(item.axis.name != "m" for item in layout.shard)
        or any(axis.name != "m" for axis in layout.offset)
    ):
        raise ValueError(f"Public tensor {variable.name} requires an affine memory layout")
    return TensorParameter(
        index,
        variable,
        tuple(ty.shape),
        tuple(item.stride for item in layout.shard),
        layout.offset.get(Axis.get("m"), ir.const(0, "int64")),
    )


class ShapeBindings:
    """Per-call integer bindings, with no facts retained between invocations."""

    def __init__(self, *, tensor_error_type: type[Exception] = ValueError, size_var_defaults=()):
        self.values: dict[ir.Var, int] = {}
        self.analyzer = create_analyzer()
        self.tensor_error_type = tensor_error_type
        self.size_var_defaults = tuple(size_var_defaults)
        self.context = sym.ConstraintContext()
        self.analyzer.sync_constraint_context(self.context)

    def bind(self, variable: ir.Var, value: int, source: str, *, from_tensor: bool = False) -> None:
        value = integer_index(value)
        if not isinstance(variable.ty, ir.PrimType):
            raise TypeError(f"{source} cannot bind a non-primitive shape variable")
        dtype = variable.ty.dtype
        if not dtype.is_integer or dtype.lanes != 1:
            raise TypeError(f"{source} cannot bind a non-integer shape variable")
        unsigned = dtype.type_code == tvm_ffi.dtype("uint8").type_code
        lower = 0 if unsigned else -(1 << (dtype.bits - 1))
        upper = (1 << (dtype.bits if unsigned else dtype.bits - 1)) - 1
        error_type = self.tensor_error_type if from_tensor else ValueError
        if not lower <= value <= upper:
            raise error_type(f"{source} value {value} is outside {dtype}")
        if variable in self.values:
            if self.values[variable] != value:
                raise error_type(f"{source} conflicts with another binding of {variable.name}")
            return
        self.values[variable] = value
        self.context.append(sym.BindingConstraint(variable, ir.IntImm(variable.ty, value)))
        self.analyzer.sync_constraint_context(self.context)

    def try_evaluate(self, expression: ir.Expr) -> int | None:
        value = self.analyzer.simplify(expression)
        return int(value.value) if isinstance(value, ir.IntImm) else None

    def evaluate(self, expression: ir.Expr, source: str) -> int:
        value = self.try_evaluate(expression)
        if value is None:
            raise ValueError(f"Cannot resolve {source} from the supplied input shapes, strides and scalar parameters")
        return value

    def bind_linear(self, expression: ir.Expr, actual: int, source: str) -> None:
        """Use the native packed binder's factor*var+offset decoding contract.

        This is an exact integer quotient, never a solver-selected witness.
        Narrowing casts and multi-variable equations are not affine decoders.
        Already bound variables are checked by the caller's final equality.
        """
        # Defaults only choose the exact normalized carrier for this attempt.
        # Actual scalar/bare-tensor bindings have priority, and final shape
        # equalities below use the unmodified expression in the real context.
        temporary = self.context.snapshot()
        for candidate in self.size_var_defaults:
            if candidate not in self.values:
                temporary.append(sym.PredicateConstraint(candidate >= ir.const(0, candidate.ty.dtype)))
        self.analyzer.sync_constraint_context(temporary)
        try:
            variable = self.analyzer.simplify(expression)
        finally:
            self.analyzer.sync_constraint_context(self.context)
        factor, offset = 1, 0
        if isinstance(variable, prim.Add):
            if isinstance(variable.b, ir.IntImm):
                variable, offset = variable.a, int(variable.b.value)
            elif isinstance(variable.a, ir.IntImm):
                variable, offset = variable.b, int(variable.a.value)
            else:
                return
        elif isinstance(variable, prim.Sub):
            if not isinstance(variable.b, ir.IntImm):
                return
            variable, offset = variable.a, -int(variable.b.value)
        if isinstance(variable, prim.Mul):
            if isinstance(variable.b, ir.IntImm):
                variable, factor = variable.a, int(variable.b.value)
            elif isinstance(variable.a, ir.IntImm):
                variable, factor = variable.b, int(variable.a.value)
            else:
                return
        if not isinstance(variable, ir.Var) or variable in self.values or factor == 0:
            return
        delta = actual - offset
        if delta % factor:
            raise self.tensor_error_type(f"{source} minus {offset} must be divisible by {factor}")
        self.bind(variable, delta // factor, source, from_tensor=True)


class FunctionABIPlan:
    """Resolve native public parameters before allocating caller-owned outputs.

    The packed native function remains responsible for DLPack dtype, device,
    byte-offset and pointer lifetime checks. This plan resolves output sizes
    and checks equalities against all input dimensions, including parameters
    that appear later in the signature. It does not create a legacy buffer_map.
    """

    def __init__(
        self,
        function: tirx.Function,
        output_indices: list[int] | tuple[int, ...] = (),
        *,
        tensor_error_type: type[Exception] = ValueError,
    ):
        if not isinstance(function, tirx.Function):
            raise TypeError("FunctionABIPlan requires a native tirx.Function")
        self.function = function
        self.size_var_defaults = tuple(tvm_ffi.get_global_func("tl.CollectSizeVarDefaults")(function))
        # Adapters may resolve tensor metadata before entering native code.
        # Keep that boundary's established error class without converting
        # unrelated argument-count, output-selection or user exceptions.
        self.tensor_error_type = tensor_error_type
        self.output_indices = tuple(_normalize_output_indices(list(output_indices), len(function.params)))
        self.input_indices = tuple(index for index in range(len(function.params)) if index not in self.output_indices)
        self.tensors = {
            index: _tensor_parameter(index, variable)
            for index, variable in enumerate(function.params)
            if isinstance(variable.ty, tirx.TensorType)
        }
        for index in self.output_indices:
            if index not in self.tensors:
                raise ValueError(f"Output parameter {index} is not a tensor")

    def bind_inputs(self, inputs) -> tuple[list, ShapeBindings]:
        if len(inputs) != len(self.input_indices):
            raise ValueError(f"Kernel expected {len(self.input_indices)} inputs, but {len(inputs)} are provided.")
        arguments = [None] * len(self.function.params)
        bindings = ShapeBindings(tensor_error_type=self.tensor_error_type, size_var_defaults=self.size_var_defaults)
        constraints = []
        for index, value in zip(self.input_indices, inputs):
            arguments[index] = value
            variable = self.function.params[index]
            tensor = self.tensors.get(index)
            if tensor is None:
                if isinstance(variable.ty, ir.PrimType):
                    dtype = variable.ty.dtype
                    if dtype.is_integer and dtype.lanes == 1:
                        bindings.bind(variable, value, f"scalar parameter {index}")
                continue
            if value is None:
                # Native MakePackedAPI distinguishes data use from shape-only
                # carriers and owns the nullable contract. An absent carrier
                # contributes no shape/stride facts to caller allocation.
                continue
            shape = tuple(value.shape)
            if len(shape) != len(tensor.shape):
                raise self.tensor_error_type(f"Tensor parameter {index} ndim mismatch: expected rank {len(tensor.shape)}, got {len(shape)}")
            element_bits = tensor.dtype.bits * tensor.dtype.lanes
            scale = 1
            if element_bits < 8 and str(tensor.dtype) != "bool":
                if isinstance(value, tvm_ffi.Tensor):
                    storage_bits = value.dtype.bits * value.dtype.lanes
                elif hasattr(value, "element_size"):
                    storage_bits = int(value.element_size()) * 8
                else:
                    storage_bits = int(value.itemsize) * 8
                if storage_bits % element_bits:
                    raise self.tensor_error_type(f"Tensor parameter {index} has incompatible sub-byte storage")
                scale = storage_bits // element_bits
                shape = (*shape[:-1], shape[-1] * scale) if shape else shape
            for axis, (expression, actual) in enumerate(zip(tensor.shape, shape)):
                constraints.append((expression, int(actual), f"parameter {index} shape[{axis}]"))
            if isinstance(value, tvm_ffi.Tensor):
                # FFI exposes DLPack strides in elements, including compact
                # tensors whose native strides pointer is null.
                strides = value.strides
            elif hasattr(value, "stride"):
                strides = value.stride()
            else:
                # numpy exposes byte strides; the DLPack element contract does not.
                strides = tuple(stride // value.itemsize for stride in value.strides)
            compact = tensor.variable.ty.layout is None
            empty = any(dimension == 0 for dimension in shape)
            if scale != 1 and strides and strides[-1] != 1 and not (compact and empty):
                raise self.tensor_error_type(f"Tensor parameter {index} requires contiguous strides in its packed sub-byte dimension")
            for axis, (expression, actual) in enumerate(zip(tensor.strides, strides)):
                # Match native BindCompactStrides: empty tensors address no
                # elements, and unit axes do not constrain compactness. The
                # shape constraints above still bind every dynamic dimension.
                # Explicit layouts use BindRegularStrides, including symbolic
                # stride providers, and must retain all their checks.
                if compact and (empty or shape[axis] == 1):
                    continue
                # A storage cell packs successive elements of the innermost
                # logical dimension. Outer strides are measured in logical
                # elements; its unit inner stride remains one.
                stride_scale = scale if axis + 1 < len(strides) else 1
                constraints.append((expression, int(actual) * stride_scale, f"parameter {index} strides[{axis}]"))

        # First bind direct facts from every input. A composite dimension such
        # as 2*n+1 can then be checked even when n occurs in a later parameter.
        for expression, actual, source in constraints:
            if isinstance(expression, ir.Var):
                bindings.bind(expression, actual, source, from_tensor=True)
        for expression, actual, source in constraints:
            if bindings.try_evaluate(expression) is None:
                bindings.bind_linear(expression, actual, source)
        for expression, actual, source in constraints:
            expected = bindings.evaluate(expression, source)
            if expected != actual:
                raise self.tensor_error_type(f"{source} expected {expected}, got {actual}")
        return arguments, bindings

    def output_shape(self, index: int, bindings: ShapeBindings, *, storage_bits: int | None = None) -> tuple[int, ...]:
        tensor = self.tensors[index]
        shape = tuple(bindings.evaluate(dim, f"output parameter {index} shape[{axis}]") for axis, dim in enumerate(tensor.shape))
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Output parameter {index} has a negative dimension: {shape}")
        bits = tensor.dtype.bits * tensor.dtype.lanes
        if storage_bits is not None and bits < storage_bits and str(tensor.dtype) != "bool" and shape:
            last_bits = shape[-1] * bits
            if last_bits % storage_bits:
                raise ValueError(f"Output parameter {index} sub-byte shape is not divisible by its storage width")
            shape = (*shape[:-1], last_bits // storage_bits)
        return shape
