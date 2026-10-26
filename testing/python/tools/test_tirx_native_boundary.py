import os
import subprocess
import sys
import textwrap


def test_import_does_not_replace_native_functions_or_python_types():
    script = textwrap.dedent(
        """
        import ctypes
        import os
        import sys
        import tvm_ffi
        import tvm_tirx
        from tvm_ffi.registry import list_global_func_names
        from tvm_tirx.tirx.script import ir_builder as native_builder

        functions = {name: tvm_ffi.get_global_func(name) for name in list_global_func_names()}
        analyzer_type = tvm_tirx.sym.Analyzer
        primitive_type = tvm_tirx.ir.PrimType
        cast_type = tvm_tirx.ir.prim.Cast
        native_launch = native_builder.launch_thread
        library = os.environ.get("TILELANG_TIRX_TEST_LIBRARY")
        if library and os.environ.get("SKIP_LOADING_TILELANG_SO") == "1":
            extension = ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        import tilelang

        replaced = [name for name, function in functions.items()
                    if not function.same_as(tvm_ffi.get_global_func(name))]
        assert not replaced, replaced
        assert tvm_tirx.sym.Analyzer is analyzer_type
        assert tvm_tirx.ir.PrimType is primitive_type
        assert tvm_tirx.ir.prim.Cast is cast_type
        assert native_builder.launch_thread is native_launch
        assert "tvm" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
