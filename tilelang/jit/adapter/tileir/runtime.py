"""Runtime bridge from cached TileIR cubins to the native cuTile dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from math import prod

import torch

from tvm import tirx


class PrecompiledTileIRDispatcher:
    """Thin adapter from a TileIR cubin to cuTile's native launch dispatcher.

    Parameters
    ----------
    cubin : bytes
        The compiled kernel binary.
    symbol : str
        Kernel entry-point symbol name.
    num_args : int
        Total number of kernel arguments (buffers + scalars).

    Notes
    -----
    All arguments remain runtime arguments. cuTile 1.5's ``TileDispatcher``
    accepts parameter-annotation nodes, so the precompiled adapter supplies one
    non-constant leaf annotation per entry. Scalar-vs-buffer metadata is applied
    separately when normalizing launch arguments.
    """

    def __init__(self, cubin: bytes, symbol: str, num_args: int):
        try:
            from cuda.tile._annotated_function import LeafAnnotationNode
            from cuda.tile._cext import TileDispatcher
        except ImportError as exc:
            raise RuntimeError("cuTile native dispatcher is unavailable.") from exc

        parameter_annotations = tuple(LeafAnnotationNode(constant=False) for _ in range(num_args))

        class Dispatcher(TileDispatcher):
            def __init__(self, image: bytes, kernel_symbol: str, annotations: tuple):
                super().__init__(annotations)
                self._image = image
                self._kernel_symbol = kernel_symbol

            def _compile(self, signature, context):
                del signature, context
                # cuTile 1.5 additionally expects an optional calling convention
                # and compiler-provided parameter constraints.
                return self._image, self._kernel_symbol, None, []

        self.dispatcher = Dispatcher(cubin, symbol, parameter_annotations)


def torch_dtype_from_tileir(dtype: str):
    dtype = str(dtype)
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    if dtype == "float64":
        return torch.float64
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "int8":
        return torch.int8
    if dtype == "uint8":
        return torch.uint8
    if dtype in {"int32", "uint32"}:
        return torch.int32
    if dtype in {"int64", "uint64"}:
        return torch.int64
    if dtype == "bool":
        return torch.bool
    if dtype in {"float8_e4m3fn", "float8_e4m3"}:
        return torch.float8_e4m3fn
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    if dtype == "float8_e8m0fnu":
        # torch 2.6 has no native float8_e8m0fnu dtype; use uint8 as a bit-pattern carrier.
        # torch >= 2.8 has torch.float8_e8m0fnu as a native dtype.
        return torch.uint8
    raise TypeError(f"Unsupported TileIR temporary dtype `{dtype}`.")


def load_native_dispatchers(adapter) -> tuple[PrecompiledTileIRDispatcher, list[PrecompiledTileIRDispatcher]]:
    artifact = adapter.tileir_artifact
    if artifact.kernels:
        dispatchers = [
            PrecompiledTileIRDispatcher(
                kernel.cubin,
                kernel.kernel_name,
                len(kernel.argument_names) + bool(kernel.scratch_bytes_per_block),
            )
            for kernel in artifact.kernels
        ]
        return dispatchers[0], dispatchers

    num_args = len(artifact.argument_names) or len(adapter.params)
    dispatcher = PrecompiledTileIRDispatcher(
        artifact.cubin,
        artifact.kernel_name,
        num_args + bool(artifact.scratch_bytes_per_block),
    )
    return dispatcher, [dispatcher]


def make_torch_func(adapter) -> Callable[..., Any]:
    try:
        from cuda.tile._cext import launch
    except ImportError as exc:
        raise RuntimeError("cuTile native dispatcher is unavailable.") from exc

    dispatcher = adapter.native_dispatcher.dispatcher
    params = adapter.params
    result_idx = tuple(adapter.result_idx)
    param_dtypes = tuple(adapter.param_dtypes)
    param_shapes = tuple(tuple(shape) for shape in adapter.param_shapes)
    current_device = adapter.get_current_device_functor()
    compiled_kernels = adapter.tileir_artifact.kernels
    param_argument_names = tuple(
        adapter.prim_func.buffer_map[param].name if param in adapter.prim_func.buffer_map else param.name
        for param in adapter.prim_func.params
    )
    scalar_param_indices = {
        param.name: index for index, param in enumerate(adapter.prim_func.params) if param not in adapter.prim_func.buffer_map
    }
    expected_inputs = len(params) - len(result_idx)
    result_idx_set = frozenset(result_idx)

    def resolve_launch_stream(stream: torch.cuda.Stream | int | None) -> torch.cuda.Stream:
        """Return the PyTorch stream object required by ``cuda.tile.launch``."""

        if stream is None:
            return torch.cuda.current_stream()
        if isinstance(stream, int):
            if stream == 0:
                return torch.cuda.default_stream()
            return torch.cuda.ExternalStream(stream)
        return stream

    # Symbolic output dims are resolved against INPUT tensor shapes at call time.
    # Two deliberate choices here:
    #   * Keyed by var NAME, not tirx.Var object: TVM Vars hash by identity, and
    #     the Vars in ``adapter.param_shapes`` (built from the original PrimFunc,
    #     or deserialized from the kernel cache) are generally *different objects*
    #     from same-named Vars elsewhere (e.g. ``adapter.dynamic_symbolic_map`` is
    #     keyed by the lowered module's Vars), so identity lookups KeyError.
    #   * Values index into the runtime ``inputs`` tuple (outputs excluded), not
    #     the full param list, so allocation stays correct even when an output
    #     param is not the last one.
    input_dim_by_name: dict[str, tuple[int, int]] = {}
    input_position = 0
    for param_index in range(len(params)):
        if param_index in result_idx_set:
            continue
        for dim_index, dim in enumerate(param_shapes[param_index]):
            if isinstance(dim, tirx.Var) and dim.name not in input_dim_by_name:
                input_dim_by_name[dim.name] = (input_position, dim_index)
        input_position += 1

    def _to_scalar(arg: Any) -> Any:
        """Convert a 0-d tensor to a Python scalar for cuTile scalar arg dispatch.

        When a scalar entry param (e.g. ``scale: T.float32``) is passed as a
        0-d CUDA tensor, cuTile's ``launch`` C extension reads the tensor's
        data pointer instead of its value.  Converting to a Python scalar
        (via ``.item()``) gives cuTile the actual value.
        """
        if isinstance(arg, torch.Tensor) and arg.dim() == 0:
            return arg.item()
        return arg

    def normalize_launch_args(args: tuple[Any, ...], scalar_flags: tuple[bool, ...]) -> tuple[Any, ...]:
        """Apply the artifact's ordered scalar ABI metadata to launch arguments."""

        if len(args) != len(scalar_flags):
            raise ValueError(f"TileIR launch metadata has {len(scalar_flags)} scalar flags for {len(args)} arguments.")
        return tuple(_to_scalar(arg) if is_scalar else arg for arg, is_scalar in zip(args, scalar_flags))

    def materialize_call_args(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(inputs) == len(params):
            return inputs
        if result_idx and len(inputs) == expected_inputs:
            first_tensor = next((arg for arg in inputs if isinstance(arg, torch.Tensor)), None)
            input_index = 0
            args_list: list[Any] = []
            for i, param in enumerate(params):
                if i in result_idx:
                    if param.is_scalar():
                        raise ValueError("TileIR backend cannot allocate scalar outputs automatically.")
                    shape = []
                    for dim in param_shapes[i]:
                        if isinstance(dim, tirx.Var):
                            ref = input_dim_by_name.get(dim.name)
                            if ref is None:
                                raise ValueError(
                                    f"TileIR backend cannot allocate output tensor `{param_argument_names[i]}`: "
                                    f"symbolic dim `{dim.name}` does not appear in any input tensor shape. "
                                    "Pass the output tensor explicitly instead of relying on out_idx allocation."
                                )
                            ref_tensor_idx, ref_shape_idx = ref
                            shape.append(inputs[ref_tensor_idx].shape[ref_shape_idx])
                        else:
                            shape.append(int(dim))
                    device = first_tensor.device if first_tensor is not None else current_device()
                    args_list.append(torch.empty(*shape, dtype=param_dtypes[i], device=device))
                else:
                    args_list.append(inputs[input_index])
                    input_index += 1
            return tuple(args_list)

        if result_idx:
            raise ValueError(
                f"Expected either {expected_inputs} inputs for automatic output allocation "
                f"or {len(params)} explicit arguments, got {len(inputs)}."
            )
        raise ValueError(f"TileIR kernel expected {len(params)} explicit arguments, got {len(inputs)}.")

    def return_results(args: tuple[Any, ...]):
        if not result_idx:
            return None
        if len(result_idx) == 1:
            return args[result_idx[0]]
        return [args[i] for i in result_idx]

    def eval_launch_extent(expr: Any, args: tuple[Any, ...]) -> int:
        if isinstance(expr, int):
            return expr
        if isinstance(expr, tirx.IntImm):
            return int(expr)
        if isinstance(expr, tirx.Var):
            if expr.name in scalar_param_indices:
                return int(args[scalar_param_indices[expr.name]])
            ref_tensor_idx, ref_shape_idx = adapter._lookup_dynamic_symbolic(expr)
            return int(args[ref_tensor_idx].shape[ref_shape_idx])
        if isinstance(expr, tirx.Cast):
            return eval_launch_extent(expr.value, args)
        if isinstance(expr, tirx.Add):
            return eval_launch_extent(expr.a, args) + eval_launch_extent(expr.b, args)
        if isinstance(expr, tirx.Sub):
            return eval_launch_extent(expr.a, args) - eval_launch_extent(expr.b, args)
        if isinstance(expr, tirx.Mul):
            return eval_launch_extent(expr.a, args) * eval_launch_extent(expr.b, args)
        if isinstance(expr, tirx.FloorDiv):
            return eval_launch_extent(expr.a, args) // eval_launch_extent(expr.b, args)
        if isinstance(expr, tirx.FloorMod):
            return eval_launch_extent(expr.a, args) % eval_launch_extent(expr.b, args)
        if isinstance(expr, tirx.Min):
            return min(eval_launch_extent(expr.a, args), eval_launch_extent(expr.b, args))
        if isinstance(expr, tirx.Max):
            return max(eval_launch_extent(expr.a, args), eval_launch_extent(expr.b, args))
        raise ValueError(f"TileIR launch grid expression `{expr}` cannot be evaluated from runtime tensor shapes.")

    def runtime_grid(launch_metadata, args: tuple[Any, ...]) -> tuple[int, int, int]:
        return tuple(eval_launch_extent(extent, args) for extent in launch_metadata.grid)

    def launch_kernel(kernel, native_dispatcher, kernel_args, program_args, launch_stream):
        grid = runtime_grid(kernel.launch_metadata, program_args)
        launch_args = normalize_launch_args(kernel_args, kernel.argument_scalar_flags)
        if kernel.scratch_bytes_per_block:
            first_tensor = next((arg for arg in kernel_args if isinstance(arg, torch.Tensor)), None)
            device = first_tensor.device if first_tensor is not None else current_device()
            # Allocate on the launch stream. The allocator then keeps this
            # invocation's workspace alive until its GPU work completes,
            # including graph capture and explicit non-current streams.
            with torch.cuda.stream(launch_stream):
                scratch = torch.empty(prod(grid) * kernel.scratch_bytes_per_block, dtype=torch.uint8, device=device)
                launch(launch_stream, grid, native_dispatcher, (*launch_args, scratch))
        else:
            launch(launch_stream, grid, native_dispatcher, launch_args)

    if compiled_kernels:
        dispatchers = tuple(dispatcher.dispatcher for dispatcher in adapter.native_dispatchers)
        temporary_buffers = adapter.tileir_artifact.temporary_buffers

        def multi_kernel_func(*inputs: Any, stream: torch.cuda.Stream | int | None = None):
            args = materialize_call_args(inputs)
            first_tensor = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
            device = first_tensor.device if first_tensor is not None else current_device()
            temporary_args = tuple(
                torch.empty(
                    *temporary.shape,
                    dtype=torch_dtype_from_tileir(temporary.dtype),
                    device=device,
                )
                for temporary in temporary_buffers
            )

            def resolve_argument(ref):
                values = args if ref.kind == "parameter" else temporary_args
                if ref.index >= len(values):
                    raise ValueError(f"TileIR {ref.kind} argument reference {ref.index} is out of range for {len(values)} runtime values.")
                return values[ref.index]

            launch_stream = resolve_launch_stream(stream)
            for kernel, dispatcher in zip(compiled_kernels, dispatchers):
                if len(kernel.argument_refs) != len(kernel.argument_names):
                    raise ValueError(f"TileIR kernel `{kernel.kernel_name}` is missing stable runtime argument references.")
                kernel_args = tuple(resolve_argument(ref) for ref in kernel.argument_refs)
                launch_kernel(kernel, dispatcher, kernel_args, args, launch_stream)
            return return_results(args)

        return multi_kernel_func

    if not result_idx:

        def explicit_func(*inputs: Any, stream: torch.cuda.Stream | int | None = None):
            args = materialize_call_args(inputs)
            launch_kernel(adapter.tileir_artifact, dispatcher, args, args, resolve_launch_stream(stream))

        return explicit_func

    def allocate_outputs_func(*inputs: Any, stream: torch.cuda.Stream | int | None = None):
        args = materialize_call_args(inputs)
        launch_kernel(adapter.tileir_artifact, dispatcher, args, args, resolve_launch_stream(stream))
        return return_results(args)

    return allocate_outputs_func
