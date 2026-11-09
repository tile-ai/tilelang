"""FX to Relax conversion for statically shaped Ascend inference graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tilelang import tvm
from tvm import relax
from tvm.relax.frontend.torch.fx_translator import TorchFXImporter


class UnsupportedGraphError(ValueError):
    """A graph is outside the supported graph compiler capability contract."""


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_inputs(inputs: Sequence[object], *, require_npu: bool = True) -> torch.device:
    """Validate an inference partition without probing or changing devices."""
    if not inputs or not all(isinstance(value, torch.Tensor) for value in inputs):
        raise UnsupportedGraphError("Ascend graph inputs must be tensors")
    device = inputs[0].device
    if require_npu and device.type != "npu":
        raise UnsupportedGraphError("Ascend graph compilation requires NPU tensors")
    for value in inputs:
        if value.device != device:
            raise UnsupportedGraphError("Ascend graph inputs must use the same device")
        if value.dtype not in _FLOAT_DTYPES:
            raise UnsupportedGraphError("Ascend graph inputs support float16, bfloat16, and float32")
        if value.requires_grad and torch.is_grad_enabled():
            raise UnsupportedGraphError("Ascend graph compilation currently supports inference tensors only")
        if not value.is_contiguous() or value.storage_offset() != 0:
            raise UnsupportedGraphError("Ascend graph inputs must be contiguous with zero storage offset")
        if any(not isinstance(extent, int) or extent <= 0 for extent in value.shape):
            raise UnsupportedGraphError("Ascend graph inputs require positive static dimensions")
    return device


class _AscendFXImporter(TorchFXImporter):
    """Keep Torch output dtypes while accumulating reduced values in fp32."""

    def create_convert_map(self):
        mapping = super().create_convert_map()
        mapping.update(
            {
                "matmul": self._matmul,
                "mm": self._matmul,
                "sum": self._sum,
                "mean": self._mean,
                "amax": self._amax,
                "softmax": self._softmax,
                "bfloat16": self._bfloat16,
                "add": self._add,
                "sub": self._sub,
                "subtract": self._sub,
                "multiply": mapping["mul"],
                "divide": mapping["div"],
                "true_divide": mapping["truediv"],
                "pow": self._square_power,
                "__pow__": self._square_power,
                "to": self._to,
            }
        )
        return mapping

    def _matmul(self, node):
        lhs, rhs = self.retrieve_args(node)[:2]
        if lhs.struct_info.dtype != rhs.struct_info.dtype:
            raise UnsupportedGraphError("GEMM operands must have the same dtype")
        output_dtype = lhs.struct_info.dtype
        value = self.block_builder.emit(relax.op.matmul(lhs, rhs, out_dtype="float32"))
        if output_dtype != "float32":
            value = self.block_builder.emit(relax.op.astype(value, output_dtype))
        return value

    def _reduction(self, node, op):
        args = self.retrieve_args(node)
        value = args[0]
        dim = args[1] if len(args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
        output_dtype = self._convert_data_type(node.kwargs.get("dtype") or value.struct_info.dtype)
        if output_dtype not in ("float16", "bfloat16", "float32"):
            raise UnsupportedGraphError("Ascend graph reductions support floating point dtypes only")
        if value.struct_info.dtype != "float32":
            value = self.block_builder.emit(relax.op.astype(value, "float32"))
        value = self.block_builder.emit(op(value, axis=dim, keepdims=keepdim))
        if output_dtype != "float32":
            value = self.block_builder.emit(relax.op.astype(value, output_dtype))
        return value

    def _sum(self, node):
        return self._reduction(node, relax.op.sum)

    def _mean(self, node):
        return self._reduction(node, relax.op.mean)

    def _amax(self, node):
        return self._reduction(node, relax.op.max)

    def _softmax(self, node):
        args = self.retrieve_args(node)
        value = args[0]
        dim = args[1] if len(args) > 1 else node.kwargs.get("dim", -1)
        output_dtype = self._convert_data_type(node.kwargs.get("dtype") or value.struct_info.dtype)
        return self._softmax_value(value, dim, output_dtype)

    def _softmax_module(self, node):
        value = self.retrieve_args(node)[0]
        dtype = value.struct_info.dtype
        dim = self.named_modules[node.target].dim
        if dim is None:
            raise UnsupportedGraphError("Softmax requires an explicit dimension")
        return self._softmax_value(value, dim, dtype)

    def _softmax_value(self, value, dim, output_dtype):
        # Keep reductions in separate call_tir functions for the unfused
        # baseline. The target scheduler may reject a fused multi-reduction
        # candidate without losing an executable softmax graph.
        if value.struct_info.dtype != "float32":
            value = self.block_builder.emit(relax.op.astype(value, "float32"))
        maximum = self.block_builder.emit(relax.op.max(value, axis=dim, keepdims=True))
        shifted = self.block_builder.emit(relax.op.subtract(value, maximum))
        exponentials = self.block_builder.emit(relax.op.exp(shifted))
        total = self.block_builder.emit(relax.op.sum(exponentials, axis=dim, keepdims=True))
        value = self.block_builder.emit(relax.op.divide(exponentials, total))
        if output_dtype != "float32":
            value = self.block_builder.emit(relax.op.astype(value, output_dtype))
        return value

    def _bfloat16(self, node):
        return self.block_builder.emit(relax.op.astype(self.retrieve_args(node)[0], "bfloat16"))

    def _to(self, node):
        args = self.retrieve_args(node)
        value = args[0]
        if len(args) > 2:
            raise UnsupportedGraphError("Graph to() supports a single dtype argument")
        if (
            node.kwargs.get("device") is not None
            or node.kwargs.get("copy", False)
            or node.kwargs.get("memory_format") not in (None, torch.preserve_format)
        ):
            raise UnsupportedGraphError("Graph to() supports dtype conversion without device, copy, or layout changes")
        dtype = node.kwargs.get("dtype")
        if isinstance(dtype, torch.fx.Node):
            dtype = self.env[dtype]
        if dtype is None and len(args) > 1:
            other = args[1]
            dtype = other.struct_info.dtype if isinstance(other, relax.Expr) else other
        if dtype is None:
            return value
        dtype = self._convert_data_type(dtype)
        if dtype not in ("float16", "bfloat16", "float32"):
            raise UnsupportedGraphError("Graph to() supports float16, bfloat16, and float32")
        return self.block_builder.emit(relax.op.astype(value, dtype))

    def _add(self, node):
        return self._add_sub(node, relax.op.add)

    def _sub(self, node):
        return self._add_sub(node, relax.op.subtract)

    def _square_power(self, node):
        args = self.retrieve_args(node)
        if len(args) != 2 or not isinstance(args[1], int | float) or args[1] != 2:
            raise UnsupportedGraphError("Ascend graph pow currently supports only the constant exponent 2")
        return self.block_builder.emit(relax.op.multiply(args[0], args[0]))

    def _add_sub(self, node, op):
        args = self.retrieve_args(node)
        lhs, rhs = args[:2]
        alpha = node.kwargs.get("alpha", 1)
        if not isinstance(alpha, int | float) or alpha != 1:
            # Splitting a scaled add into multiply followed by add introduces
            # an intermediate rounding/overflow that Torch's opmath need not
            # have. Keep this atomic operation in Torch until scheduled as one.
            raise UnsupportedGraphError("Graph add/sub currently requires alpha=1")
        dtype = lhs.struct_info.dtype if isinstance(lhs, relax.Expr) else rhs.struct_info.dtype
        if not isinstance(lhs, relax.Expr):
            lhs = relax.const(lhs, dtype)
        if not isinstance(rhs, relax.Expr):
            rhs = relax.const(rhs, dtype)
        return self.block_builder.emit(op(lhs, rhs))


def fx_to_relax(gm: torch.fx.GraphModule, example_inputs: Sequence[torch.Tensor]) -> tvm.IRModule:
    """Convert one pure FX partition to unscheduled ``relax.call_tir`` IR.

    Tensor values are not executed by conversion. Device validation belongs to
    the backend, so CPU tensors can also be used for hardware-free IR tests.
    """
    validate_inputs(example_inputs, require_npu=False)
    input_info = [(tuple(value.shape), str(value.dtype)) for value in example_inputs]
    importer = _AscendFXImporter()
    # Dynamo commonly emits aten overloads; the FX importer dispatches using
    # __name__, so map each overload to the corresponding supported converter.
    custom_map = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
            name = node.target.__name__
            base = name.split(".", 1)[0]
            if base in importer.convert_map:
                converter = importer.convert_map[base]
                custom_map[name] = lambda current, _importer, convert=converter: convert(current)
    mod = importer.from_fx(
        gm,
        input_info,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False,
        custom_convert_map=custom_map,
    )
    return tvm.transform.Sequential(
        [relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(), relax.transform.DeadCodeElimination()]
    )(mod)
