"""Utilities to adapt TVM FFI kernels to Torch tensors.

This adapter intentionally captures PyTorch's current CUDA stream and device
via light-weight callables so that, when the wrapped function is invoked,
the execution observes the same stream context as the active Torch code.
On non-CUDA builds, the stream/device fall back to 0/CPU semantics.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable
import sys
import threading

import torch
from tilelang import tvm
from tvm import runtime, tirx
from tvm.tirx.expr import PrimExpr
from tvm.tirx.stmt_functor import post_order_visit, substitute
from tvm.target import Target
from tvm.relax import TensorType
from tilelang.backend.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter, CachedTextSource
from tilelang.utils.language import retrieve_func_from_module
from tilelang.engine.param import KernelParam
from tilelang.language.dtypes import dtype


COMPILE_ARGS = {}

# Runtime source kinds stored in `_DynamicSymbolicSource` below.
_DYNAMIC_SOURCE_SHAPE = 0
_DYNAMIC_SOURCE_STRIDE = 1
_DYNAMIC_SOURCE_SCALAR = 2

# A dynamic-symbol source uses the tuple layout
# `(source_kind, param_idx, dim_idx, stride_scale)`:
# - `source_kind` selects tensor shape, tensor stride, or an explicit scalar.
# - `param_idx` indexes the same full PrimFunc ABI order used by `self.params`
#   and the per-call `param_values` list.
# - `dim_idx` selects the tensor dimension; it is `-1` for scalar sources.
# - `stride_scale` converts physical Torch strides to logical element strides
#   for sub-byte dtypes; it is `1` for shape and scalar sources.
_DynamicSymbolicSource = tuple[int, int, int, int]

if sys.platform == "darwin":
    from torch.utils import cpp_extension

    COMPILE_ARGS["options"] = ["-x", "objective-c++", "-g", "-std=gnu++17"] + ["-I" + i for i in cpp_extension.include_paths()]
elif sys.platform == "win32":
    from tilelang.contrib.msvc import create_shared as _msvc_create_shared

    COMPILE_ARGS["fcompile"] = _msvc_create_shared


class TVMFFIKernelAdapter(BaseKernelAdapter):
    """Adapter that runs a TVM runtime.Executable with Torch tensors.

    Notes
    - We capture the "current" PyTorch CUDA stream/device as thunks (callables)
      rather than materializing them at construction time. This ensures the
      actual stream/device is read just-in-time when the function runs, matching
      the user's current Torch context (e.g., after a stream guard/switch).
    - The stream pointer returned is a raw CUDA stream handle compatible with
      TVM's device API; on CPU or when CUDA is unavailable, we return 0.
    """

    # Class attributes to store compiled kernel information
    target: str | Target = "cuda"
    ir_module: tvm.IRModule | None = None
    # The global source code of the kernel -> global means the source code of the kernel
    # that is not wrapped by the wrapper code
    host_kernel_source: str | None = None
    device_kernel_source: str | None = None
    executable: tvm.runtime.Executable | None = None
    # Pass configs for the compiler
    pass_configs: dict[str, Any] | None = None
    # host_mod
    host_mod: tvm.IRModule | None = None
    # device_mod
    device_mod: tvm.IRModule | None = None
    # rt_mod
    rt_mod: tvm.runtime.Module | None = None
    # Compile-time binding plan from each dynamic TIR Var to its runtime source.
    # Concrete values are read from call arguments later, in `_convert_torch_func`.
    dynamic_symbolic_map: dict[tirx.Var, _DynamicSymbolicSource] | None = None

    # Stream/device functors are inherited from BaseKernelAdapter
    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_mod: tvm.IRModule | None = None,
        device_mod: tvm.IRModule | None = None,
        rt_mod: tvm.runtime.Module | None = None,
        host_kernel_source: str | None = None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        """Initialize the adapter with the given TIR function or module.

        Args:
            params: Metadata for every PrimFunc ABI parameter, including caller
                inputs, auto-allocated outputs, and explicit scalar parameters.
            result_idx: Positions in `params` that the adapter must allocate and
                return instead of consuming from the caller. Negative indices are
                normalized by `_legalize_result_idx`.
            target: Target platform (e.g., 'cuda')
            func_or_mod: TIR function or module to be compiled
            verbose: Enable verbose logging
        """
        # Both lists use the full PrimFunc ABI order. For example, an eager
        # `kernel(x) -> out` normally has params `[x, out]` and result_idx `[1]`.
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.host_kernel_source = host_kernel_source
        self.device_kernel_source = device_kernel_source

        if isinstance(func_or_mod, tirx.PrimFunc):
            self.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            self.ir_module = func_or_mod

        self.target = Target(determine_target(target))

        self.host_mod = host_mod
        self.device_mod = device_mod
        self.rt_mod = rt_mod
        self.verbose = verbose
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.dynamic_symbolic_map = self._process_dynamic_symbolic()
        self.kernel_global_source = self.device_kernel_source
        self.executable = None
        self._executable_lock = threading.Lock()

        self._post_init()

    def _make_executable(self) -> tvm.runtime.Executable:
        if self.rt_mod is None:
            raise RuntimeError("Cannot create TVM FFI executable without a runtime module.")
        executable = runtime.Executable(self.rt_mod)
        if COMPILE_ARGS:
            # Precompile jit module with extra arguments.
            executable.jit(**COMPILE_ARGS)
        return executable

    def _get_executable(self) -> tvm.runtime.Executable:
        executable = self.executable
        if executable is not None:
            return executable

        with self._executable_lock:
            executable = self.executable
            if executable is None:
                executable = self._make_executable()
                self.executable = executable
            return executable

    def get_exportable_executable(self) -> tvm.runtime.Executable:
        return self._get_executable()

    def _process_dynamic_symbolic(self) -> dict[tirx.Var, _DynamicSymbolicSource]:
        """Build the compile-time plan used to bind dynamic symbols at runtime.

        The returned map does not contain concrete sizes. It records where each
        directly anchored dynamic Var can be read on every invocation:

        - an explicit scalar PrimFunc parameter;
        - one dimension of a non-result input tensor's shape; or
        - one dimension of a non-result input tensor's stride.

        Auto-allocated results are intentionally excluded as sources because
        their shapes must be resolved before those tensors exist. Composite
        expressions such as ``B + 1`` are not source entries either; their leaf
        Vars are bound from this map and the whole expression is evaluated later
        by `_resolve_output_shape_dim`.

        If the same Var is directly exposed by multiple inputs, the first source
        in ABI order is retained. The generated native wrapper remains responsible
        for validating that all buffers sharing the symbol have compatible shapes.
        """
        func = self.prim_func
        params = func.params
        buffer_map = func.buffer_map
        dynamic_symbolic_map: dict[tirx.Var, _DynamicSymbolicSource] = {}

        # Non-buffer parameters are explicit runtime scalars. Buffer handles are
        # also represented by tirx.Var, so checking `param not in buffer_map` is
        # what distinguishes a real scalar from a tensor handle.
        for i, param in enumerate(params):
            if param not in buffer_map and (param not in dynamic_symbolic_map):
                dynamic_symbolic_map[param] = (_DYNAMIC_SOURCE_SCALAR, i, -1, 1)

        # A bare Var in an input shape directly anchors that symbol to a concrete
        # `torch.Tensor.shape[dim]` value on each call.
        for i, param in enumerate(params):
            if param in buffer_map and i not in self.result_idx:
                buffer = buffer_map[param]
                for j, shape in enumerate(buffer.shape):
                    if isinstance(shape, tirx.Var) and (shape not in dynamic_symbolic_map) and (shape not in params):
                        dynamic_symbolic_map[shape] = (_DYNAMIC_SOURCE_SHAPE, i, j, 1)

        # Dynamic strides use the same ABI parameter indexing. Torch reports
        # strides in storage units, so sub-byte types require `stride_scale`.
        for i, param in enumerate(params):
            if param in buffer_map and i not in self.result_idx:
                buffer = buffer_map[param]
                element_bits = buffer.dtype.bits * buffer.dtype.lanes
                stride_scale = 8 // element_bits if element_bits < 8 else 1
                for j, stride in enumerate(buffer.strides):
                    if isinstance(stride, tirx.Var) and (stride not in dynamic_symbolic_map) and (stride not in params):
                        dynamic_symbolic_map[stride] = (_DYNAMIC_SOURCE_STRIDE, i, j, stride_scale)
        return dynamic_symbolic_map

    @staticmethod
    def _lookup_dynamic_symbolic_source(
        symbol: tirx.Var,
        dynamic_symbolic_map: dict[tirx.Var, _DynamicSymbolicSource],
    ) -> _DynamicSymbolicSource:
        """Return the runtime-source tuple associated with `symbol`.

        TIR Var equality constructs an IR comparison expression, so handle
        identity must be checked with `same_as`. A unique name match is retained
        only as a compatibility fallback for cached artifacts whose parameter
        metadata may contain equivalent, but non-identical, Var handles.

        Raises:
            ValueError: If no source exists, or if the name fallback is ambiguous.
        """
        for candidate, source in dynamic_symbolic_map.items():
            if symbol.same_as(candidate):
                return source

        name_matches = [source for candidate, source in dynamic_symbolic_map.items() if symbol.name == candidate.name]
        if len(name_matches) == 1:
            return name_matches[0]
        if len(name_matches) > 1:
            raise ValueError(f"Dynamic symbolic variable '{symbol.name}' has ambiguous runtime sources")
        raise ValueError(f"Dynamic symbolic variable '{symbol.name}' has no runtime source")

    def _resolve_dynamic_symbolic_value(
        self,
        symbol: tirx.Var,
        param_values: list[Any],
        dynamic_symbolic_map: dict[tirx.Var, _DynamicSymbolicSource],
    ) -> Any:
        """Read one concrete symbol value from the current call's ABI arguments.

        `param_values` is aligned one-to-one with `self.params`. Caller-provided
        inputs have already been populated, while auto-allocated result slots may
        still be `None`. `_process_dynamic_symbolic` therefore records only scalar
        parameters and non-result tensors as valid sources.

        Returns:
            The explicit scalar value, tensor shape dimension, or logical stride
            that should replace `symbol` in an output-shape expression.
        """
        source_kind, param_idx, dim_idx, stride_scale = self._lookup_dynamic_symbolic_source(symbol, dynamic_symbolic_map)
        ref_value = param_values[param_idx]
        if source_kind == _DYNAMIC_SOURCE_SCALAR:
            if ref_value is None:
                raise ValueError(f"Dynamic symbolic variable '{symbol.name}' has no scalar runtime value")
            return ref_value
        if not isinstance(ref_value, torch.Tensor):
            raise ValueError(
                f"Dynamic symbolic variable '{symbol.name}' requires tensor parameter {param_idx}, but got {type(ref_value).__name__}"
            )
        if source_kind == _DYNAMIC_SOURCE_SHAPE:
            return ref_value.shape[dim_idx]
        if source_kind == _DYNAMIC_SOURCE_STRIDE:
            return ref_value.stride()[dim_idx] * stride_scale
        raise ValueError(f"Unknown dynamic symbolic reference kind: {source_kind}")

    def _resolve_output_shape_dim(
        self,
        dim: int | PrimExpr,
        param_values: list[Any],
        dynamic_symbolic_map: dict[tirx.Var, _DynamicSymbolicSource],
    ) -> int:
        """Resolve one output-shape dimension to the `int` required by Torch.

        Static integers and bare dynamic Vars use fast paths. For a composite
        PrimExpr, `post_order_visit` collects every nested Var by handle identity,
        each Var is replaced with an IntImm from the current call, and TVM's
        Analyzer folds the substituted expression. The final result must be an
        IntImm; otherwise the dimension still contains an unbound symbol or an
        operation that cannot represent a concrete tensor extent.

        Example: ``B + 1`` with ``B = 4`` becomes ``Add(4, 1)``, then ``IntImm(5)``.

        Raises:
            TypeError: If a dimension or resolved symbol is not integer-valued.
            ValueError: If the expression cannot be fully resolved to an IntImm.
        """
        if isinstance(dim, int):
            return dim
        if isinstance(dim, tirx.IntImm):
            return int(dim)
        if isinstance(dim, tirx.Var):
            return int(self._resolve_dynamic_symbolic_value(dim, param_values, dynamic_symbolic_map))
        if not isinstance(dim, PrimExpr):
            raise TypeError(f"Unsupported output shape dimension type: {type(dim).__name__}")

        symbols: list[tirx.Var] = []

        def collect_symbol(node: Any) -> None:
            if isinstance(node, tirx.Var) and not any(node.same_as(symbol) for symbol in symbols):
                symbols.append(node)

        post_order_visit(dim, collect_symbol)
        value_map = {}
        for symbol in symbols:
            runtime_value = self._resolve_dynamic_symbolic_value(symbol, param_values, dynamic_symbolic_map)
            try:
                runtime_value = int(runtime_value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"Dynamic symbolic variable '{symbol.name}' resolved to non-integer value {runtime_value!r}") from error
            value_map[symbol] = tirx.IntImm(symbol.dtype, runtime_value)

        resolved = tvm.arith.Analyzer().simplify(substitute(dim, value_map))
        if not isinstance(resolved, tirx.IntImm):
            raise ValueError(f"Output shape expression '{dim}' did not resolve to an integer; simplified to '{resolved}'")
        return int(resolved)

    def _convert_torch_func(self) -> Callable[..., Any]:
        """Create the PyTorch-facing callable for this compiled adapter.

        Adapter construction caches dtype/shape metadata and the symbolic-source
        plan in the returned closure. Each invocation then reconstructs the full
        PrimFunc ABI, resolves and allocates implicit outputs from current runtime
        inputs, calls the native executable, and returns the result positions.
        """
        # Capture thunks that reflect Torch's current stream and device.
        # These are evaluated at call time to align TVM execution with the
        # caller's active PyTorch stream/device.
        # current_stream_functor = self.get_current_stream_functor()
        current_device_functor = self.get_current_device_functor()

        # Convert TVM types to native Python types during initialization
        # Convert tvm.DataType to torch.dtype for tensor creation
        param_dtypes = [param.torch_dtype() for param in self.params]
        # Cache parameter shapes in full ABI order. Static IntImm dimensions are
        # converted eagerly; dynamic Vars and composite PrimExprs remain as IR
        # handles until a concrete call supplies their runtime values.
        param_shapes = []

        for param in self.params:
            native_shape = []
            for dim in param.shape:
                if isinstance(dim, tirx.IntImm):
                    native_shape.append(int(dim))
                elif isinstance(dim, tirx.Var):
                    native_shape.append(dim)  # Keep tirx.Var for dynamic dimensions
                else:
                    native_shape.append(dim)
            tl_dtype = param.dtype
            if tl_dtype.bits < 8:
                stroage_dtype: dtype = dtype(param.torch_dtype())
                # last dim divide by bits to get the actual shape
                native_shape[-1] = native_shape[-1] * tl_dtype.bits * tl_dtype.lanes // (stroage_dtype.bits * stroage_dtype.lanes)
            param_shapes.append(native_shape)

        # This is a source plan only. No dynamic value is captured here, which is
        # why one compiled adapter can be reused for inputs with different shapes.
        dynamic_symbolic_map = self._process_dynamic_symbolic()

        # Prepare helpers for friendly dtype error messages
        prim_func = self.prim_func
        buffer_map = prim_func.buffer_map
        params = prim_func.params
        # Expected dtype string per parameter index (for buffers only)
        expected_dtype_strs: list[str | None] = []
        # Track whether each param is a buffer (has dtype) vs scalar
        is_buffer_param: list[bool] = []
        for p in params:
            if p in buffer_map:
                expected_dtype_strs.append(str(buffer_map[p].dtype))
                is_buffer_param.append(True)
            else:
                expected_dtype_strs.append(None)
                is_buffer_param.append(False)

        def func(*inputs: torch.Tensor | Any):
            """Allocate implicit results and invoke the compiled full-ABI function."""

            # `inputs` contains only caller-supplied parameters. Positions listed
            # in `result_idx` are omitted because this adapter creates them.
            expected_inputs = len(self.params) - len(self.result_idx)
            if len(inputs) != expected_inputs:
                raise ValueError(f"Kernel expected {expected_inputs} inputs, but {len(inputs)} are provided.")

            # Resolve the device used for outputs. Prefer the first tensor input's device
            # if available, otherwise use PyTorch's current device.
            out_device: torch.device | None = next(
                (input.device for input in inputs if isinstance(input, torch.Tensor)),
                None,
            )

            # Reconstruct the full positional ABI list expected by the executable.
            # For params `[x, out]` with result_idx `[1]`, this first produces
            # `[x, None]`. Populating every input before any output allocation lets
            # output shapes reference an input regardless of signature position.
            ins_idx: int = 0
            param_values: list[Any] = [None] * len(self.params)
            for i in range(len(self.params)):
                if i not in self.result_idx:
                    param_values[i] = inputs[ins_idx]
                    ins_idx += 1

            # Resolve every implicit result shape from the current `param_values`,
            # allocate it with Torch, and fill its reserved ABI slot.
            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = param_dtypes[i]
                    shape = [self._resolve_output_shape_dim(s, param_values, dynamic_symbolic_map) for s in param_shapes[i]]

                    if out_device is None:
                        out_device = current_device_functor()

                    if len(shape) == 0:
                        param_name = self.params[i].name if hasattr(self.params[i], "name") else f"parameter_{i}"
                        raise ValueError(
                            f"Cannot create output tensor (name={param_name}) - 0-dimensional tensors are not supported. "
                            f"Expected shape: {shape}"
                        )
                    tensor = torch.empty(*shape, dtype=dtype, device=out_device)
                    param_values[i] = tensor

            # The native wrapper receives the complete PrimFunc ABI, including
            # both caller inputs and the tensors allocated above.
            executable = self._get_executable()
            executable(*param_values)

            # Return outputs in the requested form
            if len(self.result_idx) == 1:
                return param_values[self.result_idx[0]]
            return [param_values[i] for i in self.result_idx]

        return func

    @classmethod
    def from_database(
        cls,
        params: list[TensorType],
        result_idx: list[int],
        target: str,
        func_or_mod: tirx.PrimFunc | tvm.IRModule,
        host_kernel_source: CachedTextSource,
        device_kernel_source: CachedTextSource,
        kernel_lib_path: str,
        verbose: bool = False,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | None = None,
    ):
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        host_kernel_source = adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        device_kernel_source = adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        adapter.wrapped_source = (
            device_kernel_source.text + "\n\n" + host_kernel_source.text
            if device_kernel_source.text is not None and host_kernel_source.text is not None
            else None
        )
        adapter.pass_configs = pass_configs

        if isinstance(func_or_mod, tirx.PrimFunc):
            adapter.ir_module = tvm.IRModule({func_or_mod.attrs["global_symbol"]: func_or_mod})
        else:
            adapter.ir_module = func_or_mod

        target = determine_target(target, return_object=True)
        adapter.target = Target(determine_target(target))

        adapter.verbose = verbose
        adapter.libpath = kernel_lib_path
        adapter.kernel_global_source = device_kernel_source.text
        adapter.rt_mod = None
        adapter.executable = runtime.load_module(kernel_lib_path)
        adapter._executable_lock = threading.Lock()
        adapter._post_init()
        return adapter

    def get_host_source(self) -> str | None:
        """Returns the source code of the host module."""
        source = self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")
        if source is not None:
            return source
        rt_mod = getattr(self, "rt_mod", None)
        if rt_mod is None:
            return None
        return rt_mod.inspect_source()

    def get_device_source(self) -> str | None:
        """Returns the source code of the device module."""
        source = self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        if source is not None:
            self.kernel_global_source = source
            return source
        rt_mod = getattr(self, "rt_mod", None)
        if rt_mod is None:
            return None
        return rt_mod.imports[0].inspect_source()

    def get_kernel_source(self, kernel_only: bool = False):
        """Returns the source code of the compiled kernel."""
        device_source = self.get_device_source() or ""
        if kernel_only:
            return device_source

        host_source = self.get_host_source() or ""
        if device_source and host_source:
            return device_source + "\n\n" + host_source
        return device_source or host_source

    @property
    def prim_func(self) -> tirx.PrimFunc:
        """Returns the primary TIR function from the IR module."""
        return retrieve_func_from_module(self.ir_module)
