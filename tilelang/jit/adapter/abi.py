"""One native signature graph for source adapters and generated host wrappers.

Tensor storage is projected from TensorType and its affine memory layout.
Parameter and shape identities belong to the original Function; separately
serialized KernelParam objects never supply variable identity to this plan.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass

import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.engine.param import KernelParam
from tilelang.jit.abi import FunctionABIPlan, ShapeBindings, TensorParameter


@dataclass(frozen=True)
class SymbolSource:
    """Carrier: shape (0), stride (1), scalar (2), or raw pointer origin (3)."""

    kind: int
    parameter_index: int
    axis: int
    scale: int = 1

    def tuple(self):
        return self.kind, self.parameter_index, self.axis, self.scale


class SourceABIPlan:
    """Project a Function to the tensor/pointer/scalar ABI of source launchers.

    ``implicit_symbols`` has one deterministic order shared by the wrapper and
    runtime: shape variables in parameter/axis order, followed by stride-only
    variables, then offset-only variables. Explicit scalar parameters are
    never appended a second time.
    Names are presentation only. Same-name definitions have distinct entries
    and receive distinct emitted identifiers.
    """

    def __init__(self, function: tirx.Function):
        self.function = function
        self.function_plan = FunctionABIPlan(function)
        self.size_var_defaults = self.function_plan.size_var_defaults
        self.tensors: dict[int, TensorParameter] = self.function_plan.tensors
        self.parameters = tuple(KernelParam.from_var(variable) for variable in function.params)
        self.tensor_parameters = {parameter.variable: parameter for parameter in self.tensors.values()}
        self.explicit_indices = {variable: index for index, variable in enumerate(function.params)}
        self.implicit_symbols: list[ir.Var] = []
        self.sources: dict[ir.Var, list[SymbolSource]] = {}
        self._compound_symbols: set[ir.Var] = set()
        self._body_symbols: set[ir.Var] = set()
        # A source default is not a runtime body use. Nullable carrier policy
        # must inspect the executable body, as the native packed binder does.
        executable = tvm_ffi.get_global_func("tl.StripSizeVarDefaults")(function)
        for value in (executable.body, function.ret_type, function.attrs):
            if value is not None:
                tvm_ffi.structural_walk(value, (ir.Var, self._body_symbols.add), order="pre")
        self.names: dict[ir.Var, str] = {}
        self._used_names = {
            "stream",
            "device_id",
            "kernels",
            "error_buf",
            "int",
            "long",
            "short",
            "float",
            "double",
            "unsigned",
            "signed",
            "bool",
            "void",
            "auto",
            "struct",
            "class",
            "template",
            "return",
            "const",
            "volatile",
            "restrict",
            "namespace",
            "new",
            "delete",
            "sizeof",
            "nullptr",
        }
        seen_implicit = set()
        for parameter in function.params:
            self.name(parameter)
        for index, parameter in enumerate(function.params):
            if isinstance(parameter.ty, ir.PrimType):
                self.sources.setdefault(parameter, []).append(SymbolSource(2, index, -1))

        # Record ABI order independently from the preferred nullable carrier.
        for field in ("shape", "strides", "element_offset"):
            for tensor in self.tensors.values():
                expressions = (tensor.element_offset,) if field == "element_offset" else getattr(tensor, field)
                for expression in expressions:
                    variables = []
                    tvm_ffi.structural_walk(expression, (ir.Var, variables.append), order="pre")
                    if not isinstance(expression, ir.Var):
                        self._compound_symbols.update(variables)
                    for variable in variables:
                        if not ir.is_prim_expr(variable):
                            raise TypeError("Tensor shape and stride variables must have primitive types")
                        if variable not in self.explicit_indices and variable not in seen_implicit:
                            self.implicit_symbols.append(variable)
                            seen_implicit.add(variable)
                            self.name(variable)

        # Keep the original parameter order for fallback carrier selection,
        # with shape before stride inside each parameter.
        for index, tensor in self.tensors.items():
            bits = tensor.dtype.bits * tensor.dtype.lanes
            stride_scale = 8 // bits if bits < 8 else 1
            for kind, expressions in enumerate((tensor.shape, tensor.strides)):
                for axis, expression in enumerate(expressions):
                    if isinstance(expression, ir.Var):
                        self.sources.setdefault(expression, []).append(SymbolSource(kind, index, axis, stride_scale if kind == 1 else 1))
            if isinstance(tensor.element_offset, ir.Var):
                self.sources.setdefault(tensor.element_offset, []).append(SymbolSource(3, index, -1))

    def name(self, variable: ir.Var) -> str:
        if variable in self.names:
            return self.names[variable]
        stem = re.sub(r"[^a-zA-Z0-9_]", "_", variable.name) or "arg"
        if stem[0].isdigit() or keyword.iskeyword(stem):
            stem = "arg_" + stem
        name, suffix = stem, 1
        while name in self._used_names:
            name = f"{stem}_{suffix}"
            suffix += 1
        self._used_names.add(name)
        self.names[variable] = name
        return name

    def primary_sources(self, output_indices=()) -> dict[ir.Var, SymbolSource]:
        """Prefer input shape carriers, then input strides, for allocation.

        The returned dictionary still follows ABI order. The separate ordered
        ``sources`` lists preserve nullable fallback order.
        """
        outputs = set(output_indices)
        result = {}
        for variable in self.implicit_symbols:
            candidates = self.sources.get(variable, ())
            if candidates:
                result[variable] = min(
                    candidates, key=lambda source: (source.kind, source.parameter_index in outputs, source.parameter_index, source.axis)
                )
        return result

    def validate_parameter_metadata(self, parameters):
        """Check the cache's primitive metadata and use this Function's graph."""
        if len(parameters) != len(self.parameters):
            raise ValueError("Kernel parameter metadata does not match the Function signature")
        for index, (actual, expected) in enumerate(zip(parameters, self.parameters)):
            if actual.kind != expected.kind or actual.dtype != expected.dtype or len(actual.shape) != len(expected.shape):
                raise ValueError(f"Kernel parameter {index} metadata does not match the Function signature")
        actual_shapes = tvm_ffi.convert([list(parameter.shape) for parameter in parameters])
        expected_shapes = tvm_ffi.convert([list(parameter.shape) for parameter in self.parameters])
        if not tvm_ffi.structural_equal(actual_shapes, expected_shapes, map_free_vars=True):
            raise ValueError("Kernel parameter shape graph does not match the Function signature")
        return list(self.parameters)

    def bind_arguments(self, arguments, *, packed_storage=False, validate=True, nvrtc_storage=False) -> ShapeBindings:
        """Resolve scalars from all live inputs before any output allocation.

        Raw source adapters may receive integer pointers and nullable tensor
        arguments. Such values supply no shape facts. Composite dimensions use
        the same exact affine decoder as native packed calls; a solver never
        chooses a witness for a missing runtime value.

        NVRTC's byte-carrier ABI predates the native packed descriptor ABI:
        implicit shape symbols observe the supplied storage tensor without expansion,
        while explicit sub-byte strides count logical elements on every axis.
        A symbol passed as a scalar uses that explicit value instead of the
        carrier's metadata. Fixed and composite sub-byte layouts describe
        accesses through the raw pointer. They are evaluated after their scalar
        or bare metadata sources bind, then checked against actual byte capacity.
        """
        if nvrtc_storage and packed_storage:
            raise ValueError("NVRTC byte carriers and native packed storage are distinct ABIs")
        if len(arguments) != len(self.function.params):
            raise ValueError("Argument slots do not match the Function signature")
        bindings = ShapeBindings(size_var_defaults=self.size_var_defaults)
        constraints = []
        empty_stride_sources = []

        def nvrtc_layout_value(expression):
            # Explicit scalar arguments precede tensor metadata as runtime
            # sources in the NVRTC ABI. Keep the same Var identity here: an
            # int4 extent N=14 can be backed by seven int8 storage elements.
            # Only a bare Var observes carrier metadata. A compound extent
            # such as 2*N is a logical layout, not an equation against byte
            # storage shape. Its variables must have their own live sources;
            # capacity validation below rejects any that remain unbound.
            return not isinstance(expression, ir.Var) or (
                expression in self.explicit_indices and arguments[self.explicit_indices[expression]] is not None
            )

        for index, variable in enumerate(self.function.params):
            value = arguments[index]
            if index not in self.tensors:
                if (
                    isinstance(variable.ty, ir.PrimType)
                    and variable.ty.dtype.is_integer
                    and variable.ty.dtype.lanes == 1
                    and value is not None
                ):
                    bindings.bind(variable, value, f"scalar parameter {index}")
                continue
            if value is None or not hasattr(value, "shape"):
                continue
            tensor = self.tensors[index]
            shape = tuple(value.shape)
            if len(shape) != len(tensor.shape):
                raise ValueError(f"Tensor parameter {index} expected rank {len(tensor.shape)}, got {len(shape)}")
            bits = tensor.dtype.bits * tensor.dtype.lanes
            factor = 8 // bits if bits < 8 and str(tensor.dtype) != "bool" else 1
            for axis, (expression, extent) in enumerate(zip(tensor.shape, shape)):
                if nvrtc_storage and factor > 1 and nvrtc_layout_value(expression):
                    # A logical int4(8,) may be backed by int8(4,), or
                    # by a larger allocation. Bare implicit dimensions read
                    # the exact supplied shape, as the source launcher did.
                    continue
                scale = factor if packed_storage and axis + 1 == len(shape) else 1
                constraints.append((expression, int(extent) * scale, f"parameter {index} shape[{axis}]"))
            strides = value.stride() if hasattr(value, "stride") else tuple(s // value.itemsize for s in value.strides)
            for axis, (expression, stride) in enumerate(zip(tensor.strides, strides)):
                if nvrtc_storage and factor > 1 and (tensor.variable.ty.layout is None or nvrtc_layout_value(expression)):
                    # A compact layout has no runtime stride operands. A
                    # fixed or composite stride is part of the raw layout,
                    # rather than a request to bind storage metadata.
                    continue
                if shape[axis] == 1 and not isinstance(expression, ir.Var):
                    continue
                # The last packed storage cell has one logical unit stride;
                # outer strides count all logical sub-byte elements.
                scale = factor if nvrtc_storage or axis + 1 < len(strides) else 1
                if any(int(extent) == 0 for extent in shape):
                    # Empty tensors have no addressable elements. Torch's
                    # arbitrary compact stride for a zero-sized dimension
                    # must not contradict a shape symbol already bound to
                    # zero. A stride-only symbol can still be supplied by
                    # this carrier when no nonempty input constrains it.
                    if isinstance(expression, ir.Var):
                        empty_stride_sources.append((expression, int(stride) * scale, f"parameter {index} empty stride[{axis}]"))
                    continue
                constraints.append((expression, int(stride) * scale, f"parameter {index} stride[{axis}]"))
            # Source adapters receive data_ptr(), the logical first element,
            # including any Torch storage offset. Its normalized descriptor
            # origin is zero, exactly as Torch's DLPack export. Passing a
            # storage_offset again here would apply it twice. Offset symbols
            # still belong to the graph and must agree with all other uses.
            offset = tensor.element_offset
            if not nvrtc_storage or (isinstance(offset, ir.Var) and all(source.kind == 3 for source in self.sources.get(offset, ()))):
                constraints.append((offset, 0, f"parameter {index} byte_offset"))

        for expression, actual, source in constraints:
            if isinstance(expression, ir.Var) and (validate or expression not in bindings.values):
                bindings.bind(expression, actual, source)

        # Solving one affine dimension may expose a second one. A pass that
        # binds no new variable terminates this finite dependency walk.
        def solve_constraints():
            while True:
                count = len(bindings.values)
                for expression, actual, source in constraints:
                    if bindings.try_evaluate(expression) is None:
                        simplified = bindings.analyzer.simplify(expression)
                        bindings.bind_linear(simplified, actual, source)
                if len(bindings.values) == count:
                    break

        solve_constraints()
        for variable, actual, source in empty_stride_sources:
            if variable not in bindings.values:
                bindings.bind(variable, actual, source)
        if empty_stride_sources:
            solve_constraints()
        if validate:
            for expression, actual, source in constraints:
                expected = bindings.try_evaluate(expression)
                if expected is not None and expected != actual:
                    raise ValueError(f"{source} expected {expected}, got {actual}")
            if nvrtc_storage:
                for index, tensor in self.tensors.items():
                    value = arguments[index]
                    if value is not None and hasattr(value, "shape"):
                        self._validate_nvrtc_storage(tensor, value, bindings)
        return bindings

    @staticmethod
    def _validate_nvrtc_storage(tensor, value, bindings):
        """Bound a sub-byte affine access against the real backing allocation.

        ``data_ptr`` already includes a Torch view's storage offset. Relative
        byte bounds therefore allow a layout offset exactly once, and retain
        the established packed static input and oversized output conventions.
        Raw integer pointers carry no allocation descriptor and do not enter
        this check.
        """
        bits = tensor.dtype.bits * tensor.dtype.lanes
        if bits >= 8 or str(tensor.dtype) == "bool":
            return
        shape = [bindings.evaluate(dim, f"parameter {tensor.index} shape") for dim in tensor.shape]
        if any(extent < 0 for extent in shape):
            raise ValueError(f"Tensor parameter {tensor.index} has a negative dimension")
        if any(extent == 0 for extent in shape):
            return
        strides = [bindings.evaluate(stride, f"parameter {tensor.index} stride") for stride in tensor.strides]
        low = high = bindings.evaluate(tensor.element_offset, f"parameter {tensor.index} element offset")
        for extent, stride in zip(shape, strides):
            displacement = (extent - 1) * stride
            low += min(displacement, 0)
            high += max(displacement, 0)
        needed_low, needed_high = low * bits // 8, ((high + 1) * bits + 7) // 8
        if hasattr(value, "untyped_storage"):
            origin = int(value.storage_offset()) * int(value.element_size())
            available_low = -origin
            available_high = int(value.untyped_storage().nbytes()) - origin
        elif hasattr(value, "__array_interface__"):
            owner = value
            while hasattr(getattr(owner, "base", None), "__array_interface__"):
                owner = owner.base
            origin = int(owner.__array_interface__["data"][0]) - int(value.__array_interface__["data"][0])
            available_low, available_high = origin, origin + int(owner.itemsize)
            for extent, stride in zip(owner.shape, owner.strides):
                displacement = (int(extent) - 1) * int(stride)
                available_low += min(displacement, 0)
                available_high += max(displacement, 0)
        else:
            raise TypeError(f"Tensor parameter {tensor.index} has no byte-storage capacity descriptor")
        if needed_low < available_low or needed_high > available_high:
            raise ValueError(
                f"Tensor parameter {tensor.index} needs byte range [{needed_low}, {needed_high}) "
                f"relative to data_ptr(), but its storage provides [{available_low}, {available_high})"
            )

    def resolve_implicit_value(self, variable: ir.Var, bindings: ShapeBindings, *, require_live_shape=True):
        """Resolve a native ABI scalar without inventing live tensor metadata.

        An absent optional tensor can leave a shape or stride in the raw host
        signature even when the function body never uses it. Preserve that
        established zero placeholder only for direct shape/stride carriers.
        Output allocation remains strict, as do body/type uses, compound
        dimensions, offsets and explicit scalar parameters.
        """
        value = bindings.try_evaluate(variable)
        if value is not None:
            return value
        candidates = self.sources.get(variable, ())
        kinds = {candidate.kind for candidate in candidates}
        if (
            variable in self.implicit_symbols
            and variable not in self._body_symbols
            and variable not in self._compound_symbols
            and kinds
            and kinds <= {0, 1}
            and (0 not in kinds or not require_live_shape)
        ):
            return 0
        raise TypeError(f"Dynamic symbolic var {variable.name} has no live tensor source for its required ABI value")

    def output_shape(self, index: int, bindings: ShapeBindings, *, packed_storage=False):
        tensor = self.tensors[index]
        shape = tuple(bindings.evaluate(dim, f"output parameter {index} shape[{axis}]") for axis, dim in enumerate(tensor.shape))
        if any(extent < 0 for extent in shape):
            raise ValueError(f"Output parameter {index} has a negative dimension")
        bits = tensor.dtype.bits * tensor.dtype.lanes
        factor = 8 // bits if bits < 8 and str(tensor.dtype) != "bool" else 1
        if packed_storage and factor > 1 and shape:
            if shape[-1] % factor:
                raise ValueError(f"Output parameter {index} is not aligned to its storage packing factor")
            shape = (*shape[:-1], shape[-1] // factor)
        return shape


def source_abi_plan(adapter) -> SourceABIPlan:
    """Cache a plan only while the adapter retains the same Function identity."""
    function = adapter.prim_func
    plan = getattr(adapter, "_source_abi_plan", None)
    if plan is None or not plan.function.same_as(function):
        plan = SourceABIPlan(function)
        adapter._source_abi_plan = plan
    return plan
