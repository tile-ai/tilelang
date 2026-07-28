"""Convert PyTorch FX graph to Relax IR with automatic dtype cast insertion."""

import logging
import operator
import re
from collections.abc import Callable

import torch
from torch import fx
from torch.fx.experimental.symbolic_shapes import is_symbolic

import tilelang.language as T
from tilelang import tvm as tvm
from tvm import relax, tirx as tir
from tvm.relax.frontend.torch.fx_translator import TorchFXImporter

from tilelang.graph.utils import torch_dtype_to_tvm

logger = logging.getLogger(__name__)

_K_OPAQUE = 8

# Inductor's `register_pointwise_numeric*` list — pointwise transcendentals
# whose Triton codegen computes at opmath (fp32) for fp16/bf16 inputs.  Source:
# ``torch/_inductor/lowering.py`` ``register_pointwise_numeric*`` calls.
_FP32_POINTWISE_OPS = frozenset(
    {
        "rsqrt",
        "exp",
        "exp2",
        "expm1",
        "sigmoid",
        "sqrt",
        "cos",
        "sin",
        "log",
        "log1p",
        "tan",
        "tanh",
        "reciprocal",
        "lgamma",
        "erf",
    }
)

# Composite ops that decompose to transcendentals.  Promoted explicitly in case
# torch.compile does not expand them before they reach this importer.
_FP32_COMPOSITE_OPS = frozenset(
    {
        "gelu",
        "silu",
        "softplus",
        "softmax",
        "log_softmax",
    }
)

# Matmul family: Inductor decomposes these with ``@pw_cast_for_opmath`` which
# casts inputs to fp32.  We keep inputs in their original dtype (so tensor
# cores are used) and force the matmul's ``out_dtype`` to fp32, which tensor
# cores already compute internally via fp32 accumulation.  Numerically
# equivalent to Inductor; faster in TileLang's CUDA codegen.
_FP32_MATMUL_OPS = frozenset({"linear", "matmul", "mm", "bmm", "addmm"})

_LOW_PRECISION_DTYPES = frozenset({"float16", "bfloat16"})

_INDEX_COPY_KERNEL_CACHE: dict = {}


def _build_index_copy_kernel(
    x_shape: tuple,
    src_shape: tuple,
    dim: int,
    dtype: str,
    idx_dtype: str,
) -> tir.PrimFunc:
    """Build an in-place ``index_copy_`` kernel for the 1-element index case.

    The output buffer is aliased to ``data`` at the call site via
    ``call_tir_inplace``, so the kernel only writes the single row at
    axis ``dim`` position ``index[0]`` — no full-tensor copy.
    """
    key = (tuple(x_shape), tuple(src_shape), dim, dtype, idx_dtype)
    cached = _INDEX_COPY_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    M = 1
    for s in x_shape[:dim]:
        M *= s
    tail = 1
    for s in x_shape[dim + 1 :]:
        tail *= s
    S = x_shape[dim]
    threads = min(1024, max(32, tail))

    @T.prim_func
    def kernel(
        data: T.Tensor(x_shape, dtype),
        index: T.Tensor((1,), idx_dtype),
        source: T.Tensor(src_shape, dtype),
    ):
        src_flat = T.Tensor((M, tail), dtype, source.data)
        data_flat = T.Tensor((M, S, tail), dtype, data.data)
        with T.Kernel(M, threads=threads) as bm:
            for t in T.Parallel(tail):
                data_flat[bm, T.cast(index[0], "int32"), t] = src_flat[bm, t]

    func = kernel.with_attr("tirx.is_scheduled", True)
    func = func.with_attr("tirx.is_tilelang_kernel", True)
    func = func.with_attr("op_pattern", _K_OPAQUE)
    _INDEX_COPY_KERNEL_CACHE[key] = func
    return func


class SymbolicShapeEnv:
    """Maps SymInt dimension names to shared ``tir.Var`` instances.

    One instance per ``TileLangFXImporter`` — avoids process-global state.
    """

    def __init__(self):
        self._cache: dict[str, tir.Var] = {}

    def resolve(self, s) -> "int | tir.Var":
        """Convert a shape dimension to int (static) or tir.Var (symbolic)."""
        if is_symbolic(s):
            name = str(s)
            if name not in self._cache:
                self._cache[name] = tir.Var(name, "int32")
            return self._cache[name]
        return int(s)

    def resolve_shape(self, shape) -> list:
        """Convert each dimension of a shape."""
        return [self.resolve(s) for s in shape]

    def struct_info_from_fake(self, val) -> relax.TensorStructInfo:
        """Build Relax TensorStructInfo from a FakeTensor."""
        shape = self.resolve_shape(val.shape)
        dtype = torch_dtype_to_tvm(val.dtype)
        return relax.TensorStructInfo(shape, dtype)


def extract_input_info(gm, example_inputs, sym_env=None):
    """Extract (shape, dtype_str) pairs using FX graph metadata.

    Uses node.meta["val"] from placeholder nodes which preserves SymInt
    for dynamic dimensions. Falls back to example_inputs for shapes.
    """
    if sym_env is None:
        sym_env = SymbolicShapeEnv()

    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val", node.meta.get("example_value"))
            if val is not None and isinstance(val, torch.Tensor):
                placeholders.append(val)

    sources = placeholders if placeholders else [inp for inp in example_inputs if isinstance(inp, torch.Tensor)]

    input_info = []
    for tensor in sources:
        shape = tuple(sym_env.resolve_shape(tensor.shape))
        dtype_str = torch_dtype_to_tvm(tensor.dtype)
        input_info.append((shape, dtype_str))
    return input_info


class TileLangFXImporter(TorchFXImporter):
    """FX→Relax importer with dtype cast insertion and per-op fallback.

    - After each conversion, inserts R.astype if output dtype mismatches
      the FX expectation (fused as epilogue by FuseOps/FuseTIR).
    - Unsupported ops emit opaque ``R.call_dps_packed`` calls that act as
      fusion barriers; the codegen layer maps them back to torch ops.
    """

    def _retrieve_kwargs(self, node: fx.Node):
        return self._retrieve_args(node.kwargs)

    @staticmethod
    def _arg_or_kwarg(args, kwargs, index, name, default):
        return args[index] if len(args) > index else kwargs.get(name, default)

    def _conv_transpose1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        output_padding = self._arg_or_kwarg(args, kwargs, 5, "output_padding", 0)
        groups = self._arg_or_kwarg(args, kwargs, 6, "groups", 1)
        dilation = self._arg_or_kwarg(args, kwargs, 7, "dilation", 1)
        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

    def _conv_transpose2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        output_padding = self._arg_or_kwarg(args, kwargs, 5, "output_padding", 0)
        groups = self._arg_or_kwarg(args, kwargs, 6, "groups", 1)
        dilation = self._arg_or_kwarg(args, kwargs, 7, "dilation", 1)
        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )

    def _conv1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        dilation = self._arg_or_kwarg(args, kwargs, 5, "dilation", 1)
        groups = self._arg_or_kwarg(args, kwargs, 6, "groups", 1)
        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        dilation = self._arg_or_kwarg(args, kwargs, 5, "dilation", 1)
        groups = self._arg_or_kwarg(args, kwargs, 6, "groups", 1)
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        dilation = self._arg_or_kwarg(args, kwargs, 5, "dilation", 1)
        groups = self._arg_or_kwarg(args, kwargs, 6, "groups", 1)
        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _convolution(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        kwargs = self._retrieve_kwargs(node)
        x = args[0]
        weight = args[1]
        bias = self._arg_or_kwarg(args, kwargs, 2, "bias", None)
        stride = self._arg_or_kwarg(args, kwargs, 3, "stride", 1)
        padding = self._arg_or_kwarg(args, kwargs, 4, "padding", 0)
        dilation = self._arg_or_kwarg(args, kwargs, 5, "dilation", 1)
        transposed = self._arg_or_kwarg(args, kwargs, 6, "transposed", False)
        output_padding = self._arg_or_kwarg(args, kwargs, 7, "output_padding", 0)
        groups = self._arg_or_kwarg(args, kwargs, 8, "groups", 1)

        input_shape = self.shape_of(x)
        ndim = len(input_shape)

        if transposed:
            if ndim == 3:
                return self._conv_transpose1d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    output_padding=output_padding,
                )
            if ndim == 4:
                return self._conv_transpose2d_impl(
                    x,
                    weight,
                    bias=bias,
                    strides=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    output_padding=output_padding,
                )
            raise ValueError(f"Unsupported transposed convolution dimensionality: {ndim}")

        if ndim == 3:
            return self._conv1d_impl(
                x,
                weight,
                bias=bias,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ndim == 4:
            return self._conv2d_impl(
                x,
                weight,
                bias=bias,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if ndim == 5:
            return self._conv3d_impl(
                x,
                weight,
                bias=bias,
                strides=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        raise ValueError(f"Unsupported convolution dimensionality: {ndim}")

    def _get_expected_dtype(self, node: fx.Node):
        val = node.meta.get("val", node.meta.get("example_value"))
        if val is not None and isinstance(val, torch.Tensor):
            return torch_dtype_to_tvm(val.dtype)
        return None

    def _maybe_cast(self, node: fx.Node, expr: relax.Expr) -> relax.Expr:
        """Insert a cast if the Relax expr dtype differs from FX expectation."""
        expected = self._get_expected_dtype(node)
        if expected is None:
            return expr
        sinfo = expr.struct_info_ if hasattr(expr, "struct_info_") else None
        if sinfo is None or not isinstance(sinfo, relax.TensorStructInfo):
            return expr
        actual = sinfo.dtype
        if actual and expected and actual != expected:
            logger.debug("Inserting cast %s -> %s for %s", actual, expected, node.name)
            return self.block_builder.emit(relax.op.astype(expr, expected))
        return expr

    def _cat(self, node: fx.Node) -> relax.Expr:
        """Handle torch.cat, filtering out empty-tensor inputs (KV cache prefill)."""
        args = self.retrieve_args(node)
        if isinstance(args[0], (list, tuple)):
            tensors = list(args[0])
        else:
            tensors = [args[0]]

        # Filter out 0-element tensors (e.g. empty KV cache during prefill).
        # Only catches statically-known zero dims (IntImm); symbolic zeros
        # are not filtered and will go through the normal concat path.
        non_empty = []
        for t in tensors:
            sinfo = t.struct_info_ if hasattr(t, "struct_info_") else None
            if sinfo and isinstance(sinfo, relax.TensorStructInfo):
                shape = sinfo.shape
                if shape is not None and any(isinstance(d, tvm.tirx.IntImm) and int(d) == 0 for d in shape):
                    continue
            non_empty.append(t)

        if len(non_empty) == 0:
            return tensors[0]
        if len(non_empty) == 1:
            return non_empty[0]

        axis = node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(non_empty, axis=axis))

    def _sum(self, node: fx.Node) -> relax.Var:
        """Override upstream ``_sum`` to accept ``dim``/``keepdim`` as kwargs.

        ``BaseFXGraphImporter._sum`` only inspects positional ``args``, so the
        kwarg form (``torch.sum(x, dim=-1)`` / ``aten.sum.dim_IntList``) silently
        drops ``dim`` and reduces over all axes, producing a scalar.
        """
        args = self.retrieve_args(node)
        dim = node.kwargs.get("dim", None)
        keepdim = node.kwargs.get("keepdim", False)
        if len(args) >= 2:
            dim = args[1]
        if len(args) >= 3:
            keepdim = args[2]
        return self.block_builder.emit(relax.op.sum(args[0], dim, keepdims=keepdim))

    def _item(self, node: fx.Node) -> relax.Expr:
        """Handle .item() — extract scalar from 0-dim or 1-element tensor."""
        x = self.env[node.args[0]]
        sinfo = x.struct_info_ if hasattr(x, "struct_info_") else None
        if sinfo and isinstance(sinfo, relax.TensorStructInfo):
            shape = sinfo.shape
            if shape is not None and len(shape) == 0:
                # 0-dim tensor: .item() is identity in Relax terms
                return x
        return self.block_builder.emit(relax.op.take(x, relax.const(0, "int64"), axis=0))

    def _index_copy_(self, node: fx.Node) -> relax.Expr:
        """Handle ``tensor.index_copy_(dim, index, source)``.

        Specialised for the static single-index case (cache_position
        during decode); other cases fall back to torch.  Topi's
        ``scatter_elements`` builds with ``te.extern`` which produces
        raw TIR that no GPU schedule rule can handle, so we emit a
        TileLang DSL kernel directly via ``call_tir_inplace`` to avoid
        the full-tensor copy.
        """
        args = self.retrieve_args(node)
        x, dim, index, source = args[0], args[1], args[2], args[3]

        x_sinfo = x.struct_info_ if hasattr(x, "struct_info_") else None
        idx_sinfo = index.struct_info_ if hasattr(index, "struct_info_") else None
        src_sinfo = source.struct_info_ if hasattr(source, "struct_info_") else None

        if not (
            isinstance(x_sinfo, relax.TensorStructInfo)
            and isinstance(idx_sinfo, relax.TensorStructInfo)
            and isinstance(src_sinfo, relax.TensorStructInfo)
        ):
            return self._emit_torch_fallback(node)
        if x_sinfo.shape is None or src_sinfo.shape is None or idx_sinfo.shape is None:
            return self._emit_torch_fallback(node)

        try:
            x_shape = [int(s) for s in x_sinfo.shape]
            src_shape = [int(s) for s in src_sinfo.shape]
            idx_shape = [int(s) for s in idx_sinfo.shape]
        except (TypeError, ValueError):
            return self._emit_torch_fallback(node)

        if dim < 0:
            dim += len(x_shape)
        if not (0 <= dim < len(x_shape)):
            return self._emit_torch_fallback(node)
        if len(src_shape) != len(x_shape):
            return self._emit_torch_fallback(node)
        if idx_shape != [1] or src_shape[dim] != 1:
            return self._emit_torch_fallback(node)
        for i, (xs, ss) in enumerate(zip(x_shape, src_shape)):
            if i != dim and xs != ss:
                return self._emit_torch_fallback(node)

        dtype = x_sinfo.dtype
        prim_func = _build_index_copy_kernel(tuple(x_shape), tuple(src_shape), dim, dtype, idx_sinfo.dtype)

        gvar = self.block_builder.add_func(prim_func, "index_copy_tir")
        out_sinfo = relax.TensorStructInfo(x_shape, dtype)
        result = self.block_builder.emit(relax.op.call_tir_inplace(gvar, [x, index, source], inplace_indices=0, out_sinfo=out_sinfo))

        # Propagate in-place mutation: subsequent FX nodes may read the
        # original input var, which Dynamo has not rebound to result.
        self.env[node.args[0]] = result
        return result

    def __init__(self, *args, extern_dispatch=None, fallback_namespace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_calls: dict[str, tuple] = {}
        self.extern_dispatch = extern_dispatch or default_extern_dispatch
        self.fallback_namespace = re.sub(r"[^A-Za-z0-9_]", "_", str(fallback_namespace)) if fallback_namespace else None
        self.sym_env = SymbolicShapeEnv()

        self.convert_map["add_"] = self._binary_op(relax.op.add, operator.add)
        self.convert_map["index_copy_"] = self._index_copy_
        self.convert_map["linear"] = self._linear_fp32_output
        self.convert_map["matmul"] = self._matmul_fp32_output
        self.convert_map["mm"] = self._matmul_fp32_output
        self.convert_map["bmm"] = self._matmul_fp32_output
        self.convert_map["addmm"] = self._addmm_fp32_output
        self.convert_map["getitem"] = self._getitem_select

    # ----------------------------------------------------------------- getitem
    def _getitem_select(self, node: fx.Node) -> relax.Expr:
        """``tensor[i]`` with a plain int selects along dim 0 and drops it.

        The upstream ``_getitem`` returns the tensor unchanged for an int
        index on a TensorStructInfo var, silently turning e.g. the
        ``q, k, v = qkv[0], qkv[1], qkv[2]`` MHA split pattern into three
        aliases of the full packed tensor.
        """
        x = self.env[node.args[0]]
        idx = node.args[1]
        if (
            isinstance(x, relax.Expr)
            and not isinstance(x, relax.Tuple)
            and isinstance(getattr(x, "struct_info_", None), relax.TensorStructInfo)
            and isinstance(idx, int)
            and not isinstance(idx, bool)
        ):
            shape = self.shape_of(x)
            dim0 = shape[0]
            if idx < 0:
                if not isinstance(dim0, tir.IntImm):
                    return self._getitem(node)  # can't normalize negative index
                idx += int(dim0)
            sliced = self.block_builder.emit(relax.op.strided_slice(x, axes=[0], begin=[idx], end=[idx + 1]))
            new_shape = list(self.shape_of(sliced))[1:]
            return self.block_builder.emit(relax.op.reshape(sliced, new_shape))
        return self._getitem(node)

    # ------------------------------------------------------------------ matmul
    # Inductor-equivalent semantics for low-precision float inputs (fp16/bf16):
    # force ``out_dtype="float32"`` so downstream sees stable accumulation.
    # Inputs stay in their original dtype so tensor cores handle the multiply;
    # the fp32 accumulation happens inside the tensor core itself.
    # For any other input dtype (fp32, fp64, int*, bool), output dtype tracks
    # the inputs — an int32 matmul stays int32 end-to-end.

    def _cast_to(self, expr: relax.Expr, dtype: str) -> relax.Expr:
        sinfo = getattr(expr, "struct_info_", None)
        if isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype == dtype:
            return expr
        return self.block_builder.emit(relax.op.astype(expr, dtype))

    @staticmethod
    def _matmul_out_dtype(*inputs) -> str | None:
        """Return ``"float32"`` iff every tensor input is fp16 or bf16."""
        dtypes = []
        for x in inputs:
            sinfo = getattr(x, "struct_info_", None)
            if isinstance(sinfo, relax.TensorStructInfo):
                dtypes.append(sinfo.dtype)
        if not dtypes:
            return None
        if all(d in _LOW_PRECISION_DTYPES for d in dtypes):
            return "float32"
        return None

    def _linear_fp32_output(self, node):
        args = self.retrieve_args(node)
        x, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        out_dtype = self._matmul_out_dtype(x, weight)
        if out_dtype is not None and bias is not None:
            bias = self._cast_to(bias, out_dtype)
        return self.block_builder.emit(relax.op.linear(x, weight, bias, out_dtype))

    def _matmul_fp32_output(self, node):
        args = self.retrieve_args(node)
        out_dtype = self._matmul_out_dtype(args[0], args[1])
        return self.block_builder.emit(relax.op.matmul(args[0], args[1], out_dtype=out_dtype))

    def _addmm_fp32_output(self, node):
        args = self.retrieve_args(node)
        out_dtype = self._matmul_out_dtype(args[1], args[2])
        product = self.block_builder.emit(relax.op.matmul(args[1], args[2], out_dtype=out_dtype))
        bias = args[0]
        if out_dtype is not None:
            bias = self._cast_to(bias, out_dtype)
        return self.block_builder.emit(relax.op.add(bias, product))

    # -------------------------------------------------------- fp32 promotion
    def _should_promote_fp32(self, node: fx.Node, key) -> bool:
        from tilelang.graph.backend import backend_config

        if not getattr(backend_config, "auto_fp32_promote", True):
            return False
        if key in _FP32_MATMUL_OPS:
            # Matmul handled by dedicated overrides (fp32 output, low-prec
            # inputs preserved); do not additionally promote inputs.
            return False
        if key not in _FP32_POINTWISE_OPS and key not in _FP32_COMPOSITE_OPS:
            return False
        return self._inputs_are_low_precision(node)

    def _inputs_are_low_precision(self, node: fx.Node) -> bool:
        """True iff at least one tensor input has a low-precision dtype."""
        for src in self._tensor_fx_inputs(node):
            sinfo = getattr(self.env[src], "struct_info_", None)
            if isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype in _LOW_PRECISION_DTYPES:
                return True
        return False

    def _tensor_fx_inputs(self, node: fx.Node) -> list[fx.Node]:
        """FX nodes that (a) appear in args/kwargs and (b) are in env."""
        seen = []

        def _collect(v):
            if isinstance(v, fx.Node) and v in self.env and v not in seen:
                seen.append(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    _collect(item)

        for a in node.args:
            _collect(a)
        for v in node.kwargs.values():
            _collect(v)
        return seen

    def _call_with_fp32_promotion(self, converter, node: fx.Node) -> relax.Expr:
        """Cast low-precision tensor inputs to fp32, call the converter, restore env."""
        saved = {}
        for src in self._tensor_fx_inputs(node):
            expr = self.env[src]
            sinfo = getattr(expr, "struct_info_", None)
            if isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype in _LOW_PRECISION_DTYPES:
                saved[src] = expr
                self.env[src] = self.block_builder.emit(relax.op.astype(expr, "float32"))
        try:
            return converter(node)
        finally:
            for src, expr in saved.items():
                self.env[src] = expr

    def _convert_or_fallback(self, node: fx.Node, key):
        """Try the converter for key; extern-dispatch ops fallback to torch."""
        if self.extern_dispatch(node):
            return self._emit_torch_fallback(node)
        converter = self.convert_map.get(key)
        if converter is None:
            return self._emit_torch_fallback(node)
        try:
            if self._should_promote_fp32(node, key):
                result = self._call_with_fp32_promotion(converter, node)
            else:
                result = converter(node)
            return self._maybe_cast(node, result)
        except Exception as e:
            logger.debug("Converter failed for %s, falling back to torch: %s", key, e)
        return self._emit_torch_fallback(node)

    def _emit_torch_fallback(self, node: fx.Node) -> relax.Expr:
        """Emit an opaque packed call for an unsupported op.

        The call acts as a fusion barrier in FuseOps. At runtime the
        codegen layer dispatches it to the original torch op.
        """
        op_name = node.target.__name__ if node.op == "call_function" else node.target
        logger.info("Unsupported op '%s', emitting torch fallback", op_name)

        # Build arg template: track which args are tensors (looked up at
        # runtime from env) vs scalars (stored as-is).
        # For list/tuple args containing tensors (e.g. torch.cat([t1, t2])),
        # store as ("list", [var_name1, var_name2, ...]).
        raw_args = self.retrieve_args(node)
        arg_template = []
        flat_tensor_args = []
        for a in raw_args if isinstance(raw_args, (list, tuple)) else [raw_args]:
            if isinstance(a, relax.Expr):
                var_name = a.name_hint if hasattr(a, "name_hint") else None
                arg_template.append((True, var_name))
                flat_tensor_args.append(a)
            elif isinstance(a, (list, tuple)) and any(isinstance(e, relax.Expr) for e in a):
                # List/tuple of tensors (e.g., cat inputs)
                names = []
                for e in a:
                    if isinstance(e, relax.Expr):
                        var_name = e.name_hint if hasattr(e, "name_hint") else None
                        names.append(var_name)
                        flat_tensor_args.append(e)
                    else:
                        names.append(e)
                arg_template.append(("list", names))
            else:
                arg_template.append((False, a))

        # Resolve kwargs: FX Node references → Relax var names for tensors
        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node) and v in self.env:
                expr = self.env[v]
                if isinstance(expr, relax.Expr) and hasattr(expr, "name_hint"):
                    kwargs[k] = ("__tensor__", expr.name_hint)
                    flat_tensor_args.append(expr)
                else:
                    kwargs[k] = v
            else:
                kwargs[k] = v

        # Build output struct info from FakeTensor metadata
        val = node.meta.get("val", node.meta.get("example_value"))
        if val is not None and isinstance(val, torch.Tensor):
            out_sinfo = self.sym_env.struct_info_from_fake(val)
        else:
            out_sinfo = relax.ObjectStructInfo()

        # Store full call info under a key that is stable across pipeline
        # transforms AND unique per call spec: two fallbacks of the same op
        # with different scalar args (e.g. reshape(1,8,3,4,8) vs
        # reshape(1,8,32)) must not overwrite each other's arg template.
        spec = (node.target, arg_template, kwargs)
        symbol_prefix = "torch_fallback"
        if self.fallback_namespace is not None:
            symbol_prefix += f".{self.fallback_namespace}"
        key = f"{symbol_prefix}.{op_name}"
        suffix = 0
        while key in self.fallback_calls and self.fallback_calls[key] != spec:
            suffix += 1
            key = f"{symbol_prefix}.{op_name}_{suffix}"
        self.fallback_calls[key] = spec

        # Emit an opaque call (acts as fusion barrier).
        # Always use regular Call (not call_dps_packed) so the callee
        # allocates and returns the output directly — avoids the extra
        # pre-allocation + DtoD memcpy that call_dps_packed requires.
        extern = relax.ExternFunc(key)
        call = relax.Call(extern, flat_tensor_args, sinfo_args=[out_sinfo])
        result = self.block_builder.emit(call)

        return result

    def from_fx(
        self,
        model,
        input_info: list[tuple[tuple[int], str]],
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
        custom_convert_map: dict = None,
    ) -> tvm.IRModule:
        if custom_convert_map:
            custom_ops = set(custom_convert_map.keys())
            self.update_convert_map(custom_convert_map)
        else:
            custom_ops = set()
        self.named_modules = dict(model.named_modules())

        graph: fx.Graph = model.graph
        inputs = self.create_input_vars(input_info)

        func_name = "main"
        self.block_builder = relax.BlockBuilder()
        params = []
        if keep_params_as_input:
            func_attrs = {"num_input": len(inputs)}
            for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                shape = param.data.shape
                dtype = self._convert_data_type(str(param.data.dtype))
                inputs.append(relax.Var(name, relax.TensorStructInfo(shape, dtype)))
                self.params[param] = inputs[-1]
                params.append(tvm.runtime.tensor(param.data.cpu().numpy()))
        else:
            func_attrs = None

        with self.block_builder.function(name=func_name, params=inputs.copy(), attrs=func_attrs):
            output = None
            with self.block_builder.dataflow():
                for _, param in model.named_parameters():
                    shape = param.data.shape
                    dtype = self._convert_data_type(str(param.data.dtype))
                    if dtype in ("float32", "float16", "bfloat16"):
                        if not keep_params_as_input:
                            self.params[param] = self._convert_torch_tensor_to_relax(param)
                    else:
                        raise ValueError(f"Unsupported parameter dtype: {dtype}")

                for node in graph.nodes:
                    if node.op == "placeholder":
                        ev = node.meta.get("example_value")
                        # SymInt placeholder — store as tir.Var for dynamic shapes
                        if ev is not None and isinstance(ev, torch.SymInt):
                            resolved = self.sym_env.resolve(ev)
                            if isinstance(resolved, tir.Var):
                                self.env[node] = resolved
                            else:
                                self.env[node] = tir.const(resolved, "int32")
                            continue
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            continue
                        if ev is not None and not hasattr(ev, "shape"):
                            continue
                        assert len(inputs) > 0
                        self.env[node] = inputs.pop(0)

                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        if isinstance(args[0], (tuple, list, relax.Tuple)):
                            if unwrap_unit_return_tuple and len(args[0]) == 1:
                                output = self.block_builder.emit_output(args[0][0])
                            elif no_bind_return_tuple:
                                output = [self.block_builder.emit_output(r) for r in args[0]]
                        if output is None:
                            output = self.block_builder.emit_output(args[0])
                        break

                    elif node.op == "get_attr":
                        self.env[node] = self._fetch_attr(model, node.target)

                    elif node.op == "call_module":
                        module = self.named_modules[node.target]
                        key = type(module)
                        self.env[node] = self._convert_or_fallback(node, key)

                    elif node.op == "call_function":
                        fn = node.target.__name__
                        if fn in custom_ops:
                            result = self.convert_map[fn](node, self)
                            self.env[node] = self._maybe_cast(node, result)
                        else:
                            self.env[node] = self._convert_or_fallback(node, fn)

                    elif node.op == "call_method":
                        self.env[node] = self._convert_or_fallback(node, node.target)

                    else:
                        raise ValueError(f"Unsupported op {node.op}")

            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        if keep_params_as_input:
            mod["main"] = mod["main"].with_attr("params", params)
        return mod


def fx_to_relax(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    extern_dispatch: "Callable[[fx.Node], bool] | None" = None,
    fallback_namespace: str | None = None,
) -> tuple[tvm.IRModule, dict[str, tuple]]:
    """Convert an FX GraphModule to a Relax IRModule.

    Supported ops → Relax ops (compiled by TileLang).
    Unsupported ops → opaque calls (fusion barriers, run as torch ops).

    Parameters
    ----------
    extern_dispatch : (fx.Node) -> bool, optional
        Predicate that decides whether an FX node should be dispatched to
        torch instead of compiled by TileLang. Receives the full FX node
        so users can inspect op name, shapes, dtypes, etc.
        ``None`` uses :func:`default_extern_dispatch`.
    fallback_namespace : str, optional
        Namespace for process-global fallback symbols. The torch.compile
        backend passes the graph cache key so independently compiled graphs
        cannot overwrite each other's fallback closures.

    Returns (module, fallback_calls) where fallback_calls maps op names
    to (callable, arg_template, kwargs) for runtime dispatch.
    """
    importer = TileLangFXImporter(
        extern_dispatch=extern_dispatch,
        fallback_namespace=fallback_namespace,
    )
    input_info = extract_input_info(gm, example_inputs, sym_env=importer.sym_env)
    mod = importer.from_fx(
        gm,
        input_info,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False,
    )
    return mod, importer.fallback_calls


_supported_ops_cache = None


def _get_supported_ops() -> dict:
    """Lazily build and cache the convert_map keys to avoid instantiating
    TileLangFXImporter on every disk-cache hit."""
    global _supported_ops_cache
    if _supported_ops_cache is None:
        _supported_ops_cache = TileLangFXImporter().convert_map
    return _supported_ops_cache


def _get_matmul_mnk(node: fx.Node) -> tuple[int, int, int] | None:
    """Extract (M, N, K) from a matmul-family FX node, or None if not a matmul."""
    fn_name = node.target.__name__ if node.op == "call_function" else ""
    if fn_name not in ("linear", "mm", "addmm", "bmm", "matmul"):
        return None

    arg_vals = []
    for a in node.args:
        if isinstance(a, fx.Node):
            av = a.meta.get("val", a.meta.get("example_value"))
            if av is not None and hasattr(av, "shape"):
                arg_vals.append(av)

    if len(arg_vals) < 2:
        return None

    try:
        a_shape = list(arg_vals[0].shape)
        b_shape = list(arg_vals[1].shape)

        if fn_name == "linear":
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[0])
        elif fn_name in ("mm", "matmul"):
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[-1])
        elif fn_name == "bmm":
            M, K, N = int(a_shape[-2]), int(a_shape[-1]), int(b_shape[-1])
        elif fn_name == "addmm" and len(arg_vals) >= 3:
            a_shape = list(arg_vals[1].shape)
            b_shape = list(arg_vals[2].shape)
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[-1])
        else:
            return None

        return M, N, K
    except (TypeError, ValueError):
        return None  # symbolic (SymInt) shapes — can't determine statically


def default_extern_dispatch(node: fx.Node, *, gemv_threshold: int = 1) -> bool:
    """Default predicate for dispatching FX nodes to torch.

    Dispatches to torch (cuBLAS/FlashAttention) when:
    - The op is ``scaled_dot_product_attention``
    - The op is a matmul-family op that is NOT a GEMV (M >= ``gemv_threshold``)

    GEMV (M < ``gemv_threshold``) stays in TileLang — these are memory-bound
    and TileLang's GEMV schedule rule handles them well.

    Users can replace this with a custom callable to control dispatch::

        def my_dispatch(node):
            # Always use torch for LayerNorm
            if node.target.__name__ == "layer_norm":
                return True
            # Use default for everything else
            return default_extern_dispatch(node, gemv_threshold=8)

        compiled = torch.compile(model, backend="tilelang")
        # Or via fx_to_relax(gm, inputs, extern_dispatch=my_dispatch)
    """
    # fn_name = node.target.__name__ if node.op == "call_function" else ""
    # if fn_name in ("scaled_dot_product_attention", "embedding"):
    #     return True

    # mnk = _get_matmul_mnk(node)
    # if mnk is not None:
    #     M, N, K = mnk
    #     # GEMV (M <= gemv_threshold) → TileLang; GEMM → cuBLAS
    #     return M > gemv_threshold

    return False


def extract_fallback_calls(
    gm: fx.GraphModule,
    *,
    extern_dispatch: "Callable[[fx.Node], bool] | None" = None,
) -> dict[str, tuple]:
    """Scan FX graph for unsupported / extern-dispatched ops.

    Used when loading from disk cache — avoids re-running full conversion
    but still provides the callable references for fallback ops.
    """
    convert_map = _get_supported_ops()
    modules = dict(gm.named_modules())
    _dispatch = extern_dispatch or default_extern_dispatch
    fallback_calls = {}

    def _build_arg_template(node):
        tpl = []
        for a in node.args:
            if isinstance(a, fx.Node):
                tpl.append((True, a.name))
            elif isinstance(a, (list, tuple)) and any(isinstance(e, fx.Node) for e in a):
                tpl.append(("list", [e.name if isinstance(e, fx.Node) else e for e in a]))
            else:
                tpl.append((False, a))
        return tpl

    def _is_fallback(node, key):
        return _dispatch(node) or key not in convert_map

    for node in gm.graph.nodes:
        if node.op == "call_function":
            fn = node.target.__name__
            if _is_fallback(node, fn):
                fallback_calls[fn] = (node.target, _build_arg_template(node), dict(node.kwargs))
        elif node.op == "call_method":
            if node.target not in convert_map:
                fallback_calls[node.target] = (node.target, _build_arg_template(node), dict(node.kwargs))
        elif node.op == "call_module":
            module = modules.get(node.target)
            if module is not None and type(module) not in convert_map:
                fn = type(module).__name__
                fallback_calls[fn] = (type(module), _build_arg_template(node), dict(node.kwargs))
    return fallback_calls
