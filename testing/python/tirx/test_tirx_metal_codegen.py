import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_native_metal_source_adapter_honors_postprocessing_and_source_only():
    build = tvm_ffi.get_global_func("target.build.tilelang_metal_without_compile")
    names = ["tilelang_callback_metal_postproc", "tvm_callback_metal_compile"]
    previous = {name: tvm_ffi.get_global_func(name, allow_missing=True) for name in names}
    compiled = []
    tvm_ffi.register_global_func(names[0], lambda source, target: source + "\n// postprocessed\n", override=True)
    tvm_ffi.register_global_func(names[1], lambda *args: compiled.append(args), override=True)
    try:
        function = (
            tirx.Function([], [], ret_type=ir.TupleType([]))
            .with_attr("global_symbol", "metal_kernel")
            .with_attr("calling_conv", 2)
            .with_attr("tirx.kernel_launch_params", [])
        )
        module = build(ir.IRModule({"main": function}).with_attr("codegen.preamble", "// native adapter\n"), Target("metal"))
        source = module.inspect_source("metal")
        assert "metal_kernel" in source
        assert "// native adapter" in source
        assert "// postprocessed" in source
        assert not compiled
    finally:
        for name, callback in previous.items():
            if callback is None:
                tvm_ffi.remove_global_func(name)
            else:
                tvm_ffi.register_global_func(name, callback, override=True)
