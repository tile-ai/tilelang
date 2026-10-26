"""NVRTC launch programs serialized from native TIRx host IR."""

from tilelang._compat import ir

from tilelang._compat.stmt_functor import ir_transform, post_order_visit
from tilelang.jit.adapter.native import _with_cuda_stream, raw_host_function
from tilelang.utils.language import retrieve_func_from_module


class TLNVRTCSourceWrapper:
    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        if device_mod is None or host_mod is None:
            raise ValueError("NVRTC requires native lowered host and device modules")
        function = retrieve_func_from_module(scheduled_ir_module)
        stream = ir.Var("stream", ir.PtrType(ir.PrimType("void")))
        host = raw_host_function(function, host_mod, target, [stream])
        host = ir_transform(host, None, lambda node: _with_cuda_stream(node, stream))
        names = []

        def collect(node):
            if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tirx.call_ffi_kernel":
                name = node.args[0].value
                if name not in names:
                    names.append(name)

        post_order_visit(host, collect)
        self.function_names = names
        self.lib_code = source
        self.host_func = (
            "from tilelang.jit.adapter.nvrtc.runtime import NativeHostProgram\n"
            f"_function_names = {names!r}\n"
            f"_program = NativeHostProgram({ir.save_json(host)!r})\n"
            "def call(kernels, *arguments, stream=0):\n"
            "    return _program(kernels, arguments, stream)\n"
        )
