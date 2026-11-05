"""CuTeDSL hosts generated from native Function and launch contracts."""

from __future__ import annotations

from tvm_tirx import ir, tirx

from tilelang.jit.adapter.device_py_host import NativePythonHost
from tilelang.jit.adapter.wrapper import _require_lowered_modules
from tilelang.utils.language import retrieve_func_from_module


def _parameter(ty):
    if isinstance(ty, ir.PtrType):
        if isinstance(ty.element_type, tirx.TensorMapType):
            return {"kind": "tensormap", "dtype": "uint8", "alignment": 64}
        if isinstance(ty.element_type, ir.TupleType) and not ty.element_type.fields:
            dtype = "uint8"
        elif isinstance(ty.element_type, ir.PrimType) and ty.element_type.dtype.lanes == 1:
            dtype = str(ty.element_type.dtype)
        else:
            raise TypeError(f"Unsupported CuTeDSL pointer element type: {ty.element_type}")
        # A raw pointer carries no tensor alignment guarantee. In particular,
        # a slice's logical pointer may be less aligned than its allocation.
        return {"kind": "pointer", "dtype": dtype, "alignment": 1}
    if not isinstance(ty, ir.PrimType) or ty.dtype.lanes != 1:
        raise TypeError(f"CuTeDSL kernel parameters require native pointers or scalar primitives, got {ty}")
    return {"kind": "scalar", "dtype": str(ty.dtype)}


def _launch_fields(spec):
    """Name CuTe compiler operands; actual CUDA decoding stays in the provider."""
    if spec["cuda_launch_fields"]:
        return [(field, index, None) for index, field in enumerate(spec["cuda_launch_fields"])]
    entries, position = [], 0
    flags = {
        "tirx.use_programtic_dependent_launch": "programmatic_stream_serialization",
        "tirx.use_cooperative_launch": "cooperative",
    }
    prefixes = {"blockIdx.": "grid.", "threadIdx.": "block.", "clusterCtaIdx.": "cluster.", "preferredClusterCtaIdx.": "preferred_cluster."}
    for tag in spec["launch_param_tags"]:
        if tag in flags:
            entries.append((flags[tag], None, 1))
            continue
        if tag == "tirx.use_required_block_dimension":
            continue
        field = "dynamic_smem_bytes" if tag == "tirx.use_dyn_shared_memory" else None
        for old, new in prefixes.items():
            if tag.startswith(old) and tag[len(old) :] in ("x", "y", "z"):
                field = new + tag[len(old) :]
        if field is None:
            raise ValueError(f"Unknown native CuTeDSL launch tag: {tag}")
        entries.append((field, position, None))
        position += 1
    return entries


class _CuTePythonHost(NativePythonHost):
    def __init__(self, *args):
        super().__init__(*args)
        self.compile_specs = {}

    def _launch(self, call):
        key = f"launch_{len(self.kernels)}"
        result = super()._launch(call)
        spec = self.kernels[key]
        function = self.device_functions[spec["symbol"]]
        operands = list(call.args[1 + len(function.params) :])
        fields = []
        for field, position, constant in _launch_fields(spec):
            operand = operands[position] if position is not None else None
            if isinstance(operand, ir.IntImm):
                constant = int(operand.value)
            elif isinstance(operand, ir.FloatImm):
                constant = float(operand.value)
            fields.append({"name": field, "position": position, "constant": constant})
        self.compile_specs[key] = {
            "symbol": spec["symbol"],
            "parameters": [_parameter(parameter.ty) for parameter in function.params],
            "launch_parameters": [_parameter(operand.ty) for operand in operands],
            "fields": fields,
            "kernel_attrs": spec["cuda_kernel_attrs"],
        }
        return result


class TLCuTeDSLSourceWrapper:
    """Preserve host control flow and compile each typed device entry lazily.

    Device parameter order comes from native Functions. Kernel argument values,
    launch options and TensorMap construction come from the retained native host
    graph. CuTeDSL source is an output of that graph, never an ABI parser input.
    """

    launcher_cpp_code = None
    launcher_lib_name = None

    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        self.mod = scheduled_ir_module
        self.target = target
        self.pass_configs = pass_configs
        self.device_mod, self.host_mod = _require_lowered_modules(device_mod, host_mod)
        self.prim_func = retrieve_func_from_module(scheduled_ir_module)
        self.update_lib_code(source)

    def update_lib_code(self, code):
        self.source = self.lib_code = code
        host = _CuTePythonHost(self.prim_func, self.host_mod, self.device_mod)
        source = host.emit()
        arch = str(self.target.attrs.get("arch", ""))
        if not arch:
            raise ValueError("CuTeDSL requires a concrete target architecture")
        self.native_host = host
        self.function_names = list(host.function_names)
        self.host_func = (
            source
            + "\n_native_cutedsl_schema = 1\n"
            + f"_cutedsl_device_source = {code!r}\n"
            + f"_cutedsl_compile_specs = {host.compile_specs!r}\n"
            + f"_cutedsl_arch = {arch!r}\n"
        )
        return code

    def get_launcher_cpp_code(self):
        """The provider owns the C++ launcher; no additional DSO is emitted."""
        return None
