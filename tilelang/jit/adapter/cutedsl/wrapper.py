"""CuTe compilation and launch metadata derived from native TIRx modules."""

from tilelang._compat import ir
from tvm_tirx import tirx
from tilelang._compat import sym

from tilelang._compat.stmt_functor import ir_transform, post_order_visit
from tilelang.jit.adapter.native import _with_cuda_stream, raw_host_function
from tilelang.utils.language import retrieve_func_from_module


class TLCuTeDSLSourceWrapper:
    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        if device_mod is None or host_mod is None:
            raise ValueError("CuTeDSL requires native lowered host and device modules")
        function = retrieve_func_from_module(scheduled_ir_module)
        stream = ir.Var("stream", ir.PtrType(ir.PrimType("void")))
        host = raw_host_function(function, host_mod, target, [stream])
        host = ir_transform(host, None, lambda node: _with_cuda_stream(node, stream))
        launches = {}
        descriptors = {}

        def collect(node):
            if not isinstance(node, ir.Call) or not isinstance(node.op, ir.Op):
                return
            if node.op.name == "tirx.call_ffi_kernel":
                launches.setdefault(node.args[0].value, node)
            elif node.op.name == "tirx.cuda.tensormap_encode_tiled":
                descriptors[node.args[0]] = node.attrs

        post_order_visit(host, collect)
        functions = {
            str(function.attrs.get("global_symbol", global_var.name_hint)): function
            for global_var, function in device_mod.functions.items()
        }
        plans = []
        analyzer = sym.Analyzer()
        for name, launch in launches.items():
            if name not in functions:
                raise ValueError(f"CuTeDSL kernel {name} is missing native device metadata")
            device = functions[name]
            parameters = []
            count = launch.attrs.num_kernel_args
            if len(device.params) != count:
                raise ValueError(f"CuTeDSL kernel {name} has inconsistent native argument metadata")
            for parameter, operand in zip(device.params, launch.args[1 : 1 + count], strict=True):
                descriptor = descriptors.get(operand)
                if descriptor is not None:
                    parameters.append(
                        {
                            "kind": "tensormap",
                            "dtype": str(descriptor.descriptor_dtype),
                            "rank": int(descriptor.rank),
                            "swizzle": int(descriptor.swizzle),
                        }
                    )
                elif tirx.is_tensor_var(parameter):
                    parameters.append({"kind": "tensor", "dtype": str(parameter.dtype)})
                elif isinstance(parameter.ty, ir.PtrType):
                    element = parameter.ty.element_type
                    dtype = str(element.dtype) if isinstance(element, ir.PrimType) else "uint8"
                    parameters.append({"kind": "tensor", "dtype": "uint8" if dtype == "void" else dtype})
                elif isinstance(parameter.ty, ir.PrimType):
                    parameters.append({"kind": "scalar", "dtype": str(parameter.ty.dtype)})
                else:
                    raise TypeError(f"Unsupported native CuTeDSL parameter type: {parameter.ty}")
            fields = {}
            for field, value in zip(launch.attrs.launch_fields, launch.args[1 + count :], strict=True):
                if field == "stream" or field.startswith("grid."):
                    continue
                simplified = analyzer.simplify(value)
                fields[field] = int(simplified) if isinstance(simplified, ir.IntImm) else None
            plans.append({"name": name, "parameters": parameters, "fields": fields})
        self.function_names = list(launches)
        self.kernel_plans = plans
        self.lib_code = source
        architecture = str(target.attrs.get("arch", "sm_80"))
        self.architecture = architecture
        self.host_func = (
            "from tilelang.jit.adapter.cutedsl.runtime import CuTeHostProgram\n"
            f"_program = CuTeHostProgram({ir.save_json(host)!r}, {source!r}, {plans!r}, {architecture!r}, __file__)\n"
            "def call(*arguments, stream=0, device_id=0):\n"
            "    return _program(arguments, stream, device_id)\n"
        )
