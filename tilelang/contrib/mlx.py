"""MLX execution of static single-launch Metal kernels.

An optional native primitive owns stream ordering and explicit state aliases.
No new JIT execution backend is registered.
"""

from __future__ import annotations

import hashlib

from tilelang import tvm
from tvm import tirx


class MLXKernel:
    """A static TileLang kernel evaluated lazily by MLX.

    Use :func:`compile` to construct one. Calls accept input arrays in PrimFunc
    order, excluding ``out_idx``, and return one array or a tuple of arrays.
    MLX owns output allocation, stream ordering, and shader compilation.
    """

    def __init__(self, func, out_idx, inplace_idx):
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError("MLX execution requires mlx: install it with `pip install mlx`.") from exc
        if not mx.metal.is_available():
            raise RuntimeError("MLX execution requires an available Metal device.")
        if not isinstance(func, tirx.PrimFunc):
            raise TypeError("Expected a TileLang PrimFunc.")
        if mx.__version__ != "0.32.4":
            raise RuntimeError("The native MLX runtime currently requires mlx==0.32.4.")
        try:
            import _tilelang_mlx as runtime
        except ImportError as exc:
            raise ImportError("Install the optional runtime: pip install ./tilelang/contrib/mlx_runtime") from exc
        if runtime.mlx_version != mx.__version__:
            raise RuntimeError("Rebuild tilelang-mlx-runtime for the installed MLX version.")
        self.mx = mx
        self.runtime = runtime
        self.outputs = [out_idx] if isinstance(out_idx, int) else list(out_idx)
        self.inplace = [inplace_idx] if isinstance(inplace_idx, int) else list(inplace_idx)
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in self.outputs + self.inplace):
            raise ValueError("Output and state indices must be integers.")
        count = len(func.params)
        self.outputs = [i + count if i < 0 else i for i in self.outputs]
        self.inplace = [i + count if i < 0 else i for i in self.inplace]
        results = self.outputs + self.inplace
        if not results or len(set(results)) != len(results) or any(i < 0 or i >= count for i in results):
            raise ValueError("out_idx and inplace_idx must contain distinct, disjoint, valid parameter indices.")
        self.inputs = [i for i in range(count) if i not in self.outputs]
        self.shapes, self.dtypes, self.strides = [], [], []
        for slot, p in enumerate(func.params):
            if p not in func.buffer_map:
                raise ValueError("The MLX adapter supports tensor parameters only; specialize scalars before compiling.")
            b = func.buffer_map[p]
            if not isinstance(b.elem_offset, tirx.IntImm) or int(b.elem_offset) != 0:
                raise ValueError("The MLX adapter requires compact, zero-offset buffer declarations.")
            if not all(isinstance(s, tirx.IntImm) and int(s) > 0 for s in b.shape):
                raise ValueError("The MLX adapter requires positive static tensor shapes.")
            if b.dtype not in ("float16", "float32", "int32", "uint32"):
                raise ValueError(f"Unsupported MLX prototype dtype: {b.dtype}")
            shape = tuple(int(s) for s in b.shape)
            stride = 1
            expected_strides = []
            for extent in reversed(shape):
                expected_strides.insert(0, stride)
                stride *= extent
            strides = tuple(expected_strides)
            if b.strides:
                if not all(
                    isinstance(s, tirx.IntImm) and (int(s) > 0 or (extent == 1 and int(s) == 0)) for extent, s in zip(shape, b.strides)
                ):
                    raise ValueError("The MLX runtime requires positive static strides, except singleton axes may have stride zero.")
                strides = tuple(int(s) for s in b.strides)
            if (
                any(extent != 1 and actual != compact for extent, actual, compact in zip(shape, strides, expected_strides))
                and slot not in self.inplace
            ):
                raise ValueError("Only in-place state supports noncompact declared strides.")
            span = 1
            for extent, stride in zip(reversed(shape), reversed(strides)):
                if extent == 1:
                    continue
                if stride < span:
                    raise ValueError("In-place state requires non-overlapping row-major strides.")
                span += (extent - 1) * stride
            self.strides.append(strides)
            self.shapes.append(shape)
            self.dtypes.append(getattr(mx, b.dtype))

        from tilelang.backend import create_backend_context
        from tilelang.engine.lower import lower_with_context

        # Use ordinary Metal, independent of Metal 4 capability auto-detection.
        context = create_backend_context({"kind": "metal", "keys": ["metal", "gpu"]}, None, "tvm_ffi")
        with tvm.transform.PassContext(opt_level=3, config={"tirx.disable_vectorize": True}), context.target:
            artifact = lower_with_context(func, context, enable_host_codegen=False, enable_device_compile=False)
        if len(artifact.device_mod.functions) != 1 or len(artifact.host_mod.functions) != 1:
            raise ValueError("The MLX adapter supports exactly one device kernel and one host function.")
        device = next(iter(artifact.device_mod.functions.values()))
        host = next(iter(artifact.host_mod.functions.values()))
        symbol = next(iter(artifact.device_mod.functions)).name_hint
        bindings, launches = {}, []

        def visit(node):
            if isinstance(node, tirx.Bind):
                bindings[node.var] = node.value
            if isinstance(node, (tirx.For, tirx.IfThenElse, tirx.While)):
                raise ValueError("Host-side control flow is unsupported by the MLX adapter.")
            if isinstance(node, tirx.Call) and node.op == tvm.ir.Op.get("tirx.tvm_call_packed"):
                if node.args[0].value == symbol:
                    launches.append(node)
                elif node.args[0].value != "__tvm_set_device":
                    raise ValueError("Additional host runtime calls are unsupported by the MLX adapter.")

        tirx.stmt_functor.post_order_visit(host.body, visit)
        if len(launches) != 1:
            raise ValueError("The MLX adapter requires exactly one kernel launch.")

        def parameter_slot(expr):
            slots, visited = set(), set()

            def collect(node):
                if isinstance(node, tirx.Var) and node in bindings and node not in visited:
                    visited.add(node)
                    tirx.stmt_functor.post_order_visit(bindings[node], collect)
                if (
                    isinstance(node, tirx.Call)
                    and node.op == tvm.ir.Op.get("tirx.tvm_struct_get")
                    and node.args[0].same_as(host.params[1])
                    and int(node.args[2]) == 15
                ):
                    slots.add(int(node.args[1]))

            if not isinstance(expr, tirx.Var):
                raise ValueError("Offset or computed launch arguments are unsupported by the MLX adapter.")
            collect(expr)
            if len(slots) != 1:
                raise ValueError("Cannot bind the Metal argument to a unique PrimFunc parameter.")
            return slots.pop()

        slots = [parameter_slot(arg) for arg in launches[0].args[1 : 1 + len(device.params)]]
        readonly = {int(i) for i in device.attrs.get("tl.readonly_param_indices", [])}
        if any(slot in self.inputs and slot not in self.inplace and i not in readonly for i, slot in enumerate(slots)):
            raise ValueError("MLX inputs are immutable: declare writable inputs with inplace_idx or use a separate output buffer.")
        if any(slot not in slots for slot in results):
            raise ValueError("Every output must be used by the device kernel.")
        extents = device.attrs["thread_extent"]
        if not all(isinstance(v, tirx.IntImm) and int(v) > 0 for v in extents.values()):
            raise ValueError("The MLX adapter requires positive static launch dimensions.")
        self.threadgroup = tuple(int(extents.get(f"threadIdx.{axis}", 1)) for axis in "xyz")
        self.grid = tuple(int(extents.get(f"blockIdx.{axis}", 1)) for axis in "xyz")
        self.source = str(artifact.kernel_source)
        self.symbol = str(symbol)
        self.key = "tilelang_" + hashlib.sha256(self.source.encode()).hexdigest()
        self.results = results
        self.bindings = [-(results.index(slot) + 1) if slot in results else self.inputs.index(slot) for slot in slots]
        self.aliases = [-1] * len(self.outputs) + [self.inputs.index(slot) for slot in self.inplace]

    def __call__(self, *args, stream=None):
        if len(args) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} input arrays, got {len(args)}.")
        for arg, slot in zip(args, self.inputs):
            if not isinstance(arg, self.mx.array):
                raise TypeError("MLX kernels require mlx.core.array inputs.")
            if tuple(arg.shape) != self.shapes[slot] or arg.dtype != self.dtypes[slot]:
                raise ValueError(f"Parameter {slot} requires shape {self.shapes[slot]} and dtype {self.dtypes[slot]}.")
        # Only read-only inputs may be materialized. State views retain storage.
        inputs = [arg if slot in self.inplace else self.mx.contiguous(arg) for arg, slot in zip(args, self.inputs)]
        if stream is None:
            stream = self.mx.default_stream(self.mx.gpu)
        elif isinstance(stream, self.mx.Device):
            stream = self.mx.default_stream(stream)
        outputs = self.runtime.launch(
            inputs,
            [self.shapes[i] for i in self.results],
            [self.dtypes[i] for i in self.results],
            self.source,
            self.symbol,
            self.key,
            self.bindings,
            self.aliases,
            self.grid,
            self.threadgroup,
            [self.strides[i] for i in self.inputs],
            stream,
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def compile(func, *, out_idx=(), inplace_idx=()):
    """Compile a static single-launch kernel for MLX.

    ``out_idx`` parameters are allocated outputs and omitted from call arguments.
    ``inplace_idx`` parameters remain inputs and return aliases to their original
    storage, after the fresh outputs. Always rebind state to the returned alias:
    it carries the write dependency. Old aliases must not be used concurrently.

    State strides must match the static buffer declaration; it is never copied. Read-only
    inputs may be made contiguous. Shapes and scalar values must be specialized.
    Automatic differentiation, vmap, and TileLang cache/export are unsupported.
    """
    return MLXKernel(func, out_idx, inplace_idx)
