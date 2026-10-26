"""Raw-pointer adapters constructed from native TIRx host IR."""

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat.stmt_functor import ir_transform, post_order_visit
from tilelang._compat.tensor import tensor_strides
from tilelang.transform import _tirx_passes


def dynamic_symbols(function):
    """Return the adapter ABI's shape-first, stride-second scalar parameters."""
    symbols = []
    known = set(function.params)
    tensors = [parameter for parameter in function.params if tirx.is_tensor_var(parameter)]
    for geometry in (lambda tensor: tensor.shape, tensor_strides):
        for tensor in tensors:
            for value in geometry(tensor):
                if isinstance(value, ir.Var) and value not in known:
                    symbols.append(value)
                    known.add(value)
    return symbols


def raw_host_function(function, host_module, target, extra_parameters=()):
    """Preserve the host's control flow while replacing packed arguments with pointers."""
    scopes = []
    symbol = function.attrs.get("global_symbol")
    entries = [
        host
        for global_var, host in host_module.functions.items()
        if str(global_var.name_hint) == symbol or host.attrs.get("global_symbol") == symbol
    ]
    if not entries:
        entries = [host for host in host_module.functions.values() if host.attrs.get("tirx.is_entry_func", False)]
    if not entries and len(host_module.functions) == 1:
        entries = list(host_module.functions.values())
    if len(entries) != 1:
        raise ValueError(f"A native adapter host must have one entry for {symbol!r}")

    def collect_scope(node):
        if isinstance(node, ir.RegionStmt) and node.op.name == "tirx.compute_scope":
            scopes.append(node)

    post_order_visit(entries[0], collect_scope)
    if len(scopes) != 1:
        raise ValueError("A native adapter host must contain exactly one compute scope")
    body = scopes[0].body
    variables = {}
    for variable in tirx.analysis.undefined_vars(body):
        if variable.name in variables and not variables[variable.name].same_as(variable):
            raise ValueError(f"Ambiguous native host argument: {variable.name}")
        variables[variable.name] = variable
    parameters = []
    declarations = []
    for formal in [*function.params, *dynamic_symbols(function)]:
        parameter = variables.pop(formal.name, formal)
        if tirx.is_tensor_var(parameter):
            pointer = ir.Var(parameter.name + "_ptr", ir.PtrType(ir.PrimType(parameter.dtype), parameter.scope()))
            parameters.append(pointer)
            declarations.append(ir.Bind(parameter, ir.Call("tirx.decl_tensor", [pointer], ty_args=[parameter.ty])))
        else:
            parameters.append(parameter)
    if variables:
        raise ValueError(f"Native host has unresolved arguments: {sorted(variables)}")
    return tirx.Function(
        [*parameters, *extra_parameters],
        [*declarations, body, ir.Return(ir.const(0, "int32"))],
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs(dict={"global_symbol": "call", "tirx.is_global_func": True, "target": target}),
    )


def _with_cuda_stream(node, stream):
    if not isinstance(node, ir.Call) or not isinstance(node.op, ir.Op):
        return None
    if (
        node.op.name == "tirx.call_packed"
        and node.args
        and isinstance(node.args[0], ir.StringImm)
        and node.args[0].value == "runtime.get_cuda_stream"
    ):
        return stream
    if node.op.name != "tirx.call_ffi_kernel":
        return None
    attributes = node.attrs
    fields = list(attributes.launch_fields)
    requirements = dict(attributes.kernel_attrs.items())
    if fields:
        kernel_count = len(node.args) - len(fields) - 1
        values = list(node.args[kernel_count + 1 :])
    else:
        tags = list(attributes.launch_params)
        flags = {
            "tirx.use_programtic_dependent_launch": "programmatic_stream_serialization",
            "tirx.use_cooperative_launch": "cooperative",
        }
        value_tags = [tag for tag in tags if tag not in flags and tag != "tirx.use_required_block_dimension"]
        kernel_count = len(node.args) - len(value_tags) - 1
        operands = iter(node.args[kernel_count + 1 :])
        values = []
        prefixes = {
            "blockIdx.": "grid.",
            "threadIdx.": "block.",
            "clusterCtaIdx.": "cluster.",
            "preferredClusterCtaIdx.": "preferred_cluster.",
        }
        for tag in tags:
            if tag == "tirx.use_required_block_dimension":
                requirements["required_block_size"] = 1
                continue
            if tag in flags:
                fields.append(flags[tag])
                values.append(ir.const(True))
                continue
            field = "dynamic_smem_bytes" if tag == "tirx.use_dyn_shared_memory" else None
            for old_prefix, new_prefix in prefixes.items():
                if tag.startswith(old_prefix) and tag[-1] in "xyz":
                    field = new_prefix + tag[-1]
            if field is None:
                raise ValueError(f"Unknown native CUDA launch tag: {tag}")
            fields.append(field)
            values.append(next(operands))
    if attributes.num_kernel_args >= 0 and attributes.num_kernel_args != kernel_count:
        raise ValueError("Native CUDA launch operand count does not match its signature")
    if "stream" not in fields:
        fields.append("stream")
        values.append(stream)
    return ir.Call(
        node.op,
        [*node.args[: kernel_count + 1], *values],
        attrs=tirx.CallFFIKernelAttr(launch_fields=fields, num_kernel_args=kernel_count, kernel_attrs=requirements),
        ty=node.ty,
        loc=node.loc,
    )


_ERROR_API = """
#include <string>
extern "C" TVM_FFI_DLL_EXPORT int init() { return 0; }
extern "C" TVM_FFI_DLL_EXPORT const char* get_last_error() {
  static thread_local std::string message;
  TVMFFIObjectHandle error = nullptr;
  TVMFFIErrorMoveFromRaised(&error);
  if (error != nullptr) {
    auto* cell = TVMFFIErrorGetCellPtr(error);
    message.assign(cell->message.data, cell->message.size);
    TVMFFIObjectDecRef(error);
  }
  return message.c_str();
}
"""


class NativeCUDASourceWrapper:
    """Use standalone CUDA launch codegen and embedded cubins for Cython's C ABI."""

    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        from tvm_tirx.backend.cuda.host import export_cuda_host

        from tilelang.engine.lower import _prepare_device_codegen_mod
        from tilelang.utils.language import retrieve_func_from_module

        if host_mod is None or device_mod is None:
            raise ValueError("Native CUDA adapters require lowered host and device modules")
        function = retrieve_func_from_module(scheduled_ir_module)
        host_target = Target("cuda_host")
        stream = ir.Var("stream", ir.PtrType(ir.PrimType("void")))
        raw = raw_host_function(function, host_mod, host_target, [stream])
        raw = ir_transform(raw, None, lambda node: _with_cuda_stream(node, stream))
        raw = raw.with_attr("tirx.is_host_func", True)
        with ir.transform.PassContext(config=pass_configs or {}):
            with target:
                device = tvm_ffi.get_global_func("tl.cuda.build_from_source")(_prepare_device_codegen_mod(device_mod), target, source)
            with host_target:
                host_ir = _tirx_passes.LowerTVMBuiltin()(ir.IRModule({"call": raw}))
                host = tvm_ffi.get_global_func("tl.codegen.build_cuda_host")(_prepare_device_codegen_mod(host_ir), host_target)
        host.import_module(device)
        self.lib_code = export_cuda_host(host) + _ERROR_API
