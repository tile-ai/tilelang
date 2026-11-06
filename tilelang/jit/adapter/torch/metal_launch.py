"""Read the native host/device contract of a direct Torch Metal launch."""

from __future__ import annotations

from tvm_tirx import sym

from tvm_tirx import ir, tirx

from ..source_host import SourceHostProgram


def _constant_extent(value, tag):
    value = sym.Analyzer().simplify(value)
    if not isinstance(value, ir.IntImm):
        raise ValueError(f"The Torch Metal adapter requires a static {tag} extent, got {value}")
    extent = int(value.value)
    if extent <= 0:
        raise ValueError(f"Metal {tag} extent must be positive, got {extent}")
    return extent


class MetalLaunchPlan:
    """Select the real device symbol and caller argument order.

    The established Torch adapter forwards preallocated arguments and the
    shader's return value for one static launch. It does not allocate outputs
    or execute host control flow. The retained native host graph supplies the
    argument projection instead of guessing names or parsing emitted source.
    """

    def __init__(self, function: tirx.Function, device_module: ir.IRModule, host_module: ir.IRModule | None = None):
        if not isinstance(function, tirx.Function) or not isinstance(device_module, ir.IRModule):
            raise TypeError("Metal launches require a native Function and device IRModule")
        devices = {}
        for key, value in device_module.functions.items():
            if not isinstance(value, tirx.Function):
                continue
            attrs = value.attrs or {}
            if attrs.get("tirx.is_global_func", False) or int(attrs.get("calling_conv", 0)) == 2:
                symbol = str(attrs.get("global_symbol", key.name_hint))
                if symbol in devices:
                    raise ValueError(f"Duplicate Metal device symbol {symbol!r}")
                devices[symbol] = value
        if not devices:
            raise ValueError("The native device module has no Metal kernel")

        if host_module is None:
            raise ValueError("Selecting Metal arguments requires the native host module")

        program = SourceHostProgram(function, host_module)
        definitions = {}
        launch = None

        def same_address_call(value):
            if not isinstance(value, ir.Call) or not isinstance(value.op, ir.Op) or len(value.args) != 1:
                return False
            source_type, destination_type = value.args[0].ty, value.ty
            if value.op.name == "tirx.reinterpret":
                # A scalar bitcast changes the Python value to pass to MPS.
                # Only pointer-to-pointer casts preserve direct forwarding.
                return isinstance(source_type, ir.PtrType) and isinstance(destination_type, ir.PtrType)
            if value.op.name == "tirx.decl_tensor":
                return isinstance(source_type, ir.PtrType) and isinstance(destination_type, tirx.TensorType)
            if value.op.name == "tirx.tensor_data_ptr":
                return isinstance(source_type, tirx.TensorType) and isinstance(destination_type, ir.PtrType)
            return False

        def is_projection(value):
            if isinstance(value, (ir.Var, ir.IntImm, ir.FloatImm)):
                return True
            return same_address_call(value) and is_projection(value.args[0])

        def visit(statement):
            nonlocal launch
            if isinstance(statement, ir.SeqStmt):
                for child in statement.seq:
                    visit(child)
            elif isinstance(statement, ir.Bind) and is_projection(statement.value):
                if launch is not None:
                    raise ValueError("The Torch Metal adapter requires a single direct host launch")
                definitions[statement.var] = statement.value
            elif (
                isinstance(statement, ir.Evaluate)
                and isinstance(statement.value, ir.Call)
                and isinstance(statement.value.op, ir.Op)
                and statement.value.op.name == "tirx.call_ffi_kernel"
            ):
                if launch is not None:
                    raise ValueError("The Torch Metal adapter requires a single direct host launch")
                launch = statement.value
            else:
                raise ValueError("The Torch Metal adapter requires a single direct host launch")

        visit(program.entry.body)
        if launch is None or not isinstance(launch.args[0], ir.StringImm):
            raise ValueError("The native host module has no direct Metal launch")
        self.kernel_name = str(launch.args[0].value)
        if self.kernel_name not in devices:
            raise ValueError(f"Missing Metal device function {self.kernel_name!r}")
        self.function = devices[self.kernel_name]
        count = len(self.function.params)
        tags = [str(tag) for tag in launch.attrs.launch_params]
        if len(launch.args) != 1 + count + len(tags):
            raise ValueError("Malformed native Metal launch operand count")
        if len(set(tags)) != len(tags):
            raise ValueError("Duplicate native Metal launch dimensions")
        extents = dict(zip(tags, launch.args[1 + count :]))
        parameter_indices = {value: index for index, value in enumerate(program.entry.params)}

        def parameter(expression, active=None):
            active = set() if active is None else active
            if isinstance(expression, ir.Var):
                if expression in parameter_indices:
                    return parameter_indices[expression]
                if expression in definitions and expression not in active:
                    return parameter(definitions[expression], active | {expression})
            if same_address_call(expression):
                return parameter(expression.args[0], active)
            raise ValueError("The Torch Metal adapter requires direct public kernel arguments")

        self.argument_order = [parameter(value) for value in launch.args[1 : 1 + count]]

        self.block_info, self.grid_info = [1, 1, 1], [1, 1, 1]
        for tag, extent in extents.items():
            if tag not in {f"{kind}.{axis}" for kind in ("threadIdx", "blockIdx") for axis in "xyz"}:
                raise ValueError(f"Unsupported Torch Metal launch field: {tag}")
            sizes = self.block_info if tag.startswith("threadIdx.") else self.grid_info
            sizes["xyz".index(tag[-1])] = _constant_extent(extent, tag)
        self.threads = [block * grid for block, grid in zip(self.block_info, self.grid_info)]
