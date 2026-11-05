"""C-like emitters consume assumptions as hints, separately from runtime checks."""

import ctypes
import json
from pathlib import Path
import subprocess

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_ffi.libinfo import include_paths
from tvm_tirx.libinfo import find_include_path

I = tvm.ir


def _emit(backend, *, invalid=False, roundtrip=False):
    count = I.Var("count", "int32")
    pointer = I.Var("output", I.PtrType(I.PrimType("int32")))
    tensor = I.Var("output_view", tvm.tirx.TensorType((1,), "int32", layout="default"))
    condition = count if invalid else count > 0
    hint = I.Evaluate(I.Call("prim.assume", [condition]))
    body = [I.Bind(tensor, I.Call("tirx.decl_tensor", [pointer], ty_args=[tensor.ty])), hint, I.TensorStore(tensor, [0], count + 1)]
    # Metal's runtime binds leading buffers, then packs trailing scalars.
    # SplitHostDevice normally establishes this order before this emitter.
    params = [pointer, count] if backend == "metal" else [count, pointer]
    function = tvm.tirx.Function(params, body, ret_type=I.PrimType("void")).with_attr("global_symbol", "hint_kernel")
    if backend in ("cuda", "hip", "metal"):
        function = function.with_attr("calling_conv", 2)
    if roundtrip:
        function = I.load_json(I.save_json(function))
    suffix = "_without_compile" if backend in ("cuda", "hip", "metal") else ""
    target = "c" if backend == "c_host" else backend
    return tvm_ffi.get_global_func("target.build.tilelang_" + backend + suffix)(
        tvm.IRModule({"hint_kernel": function}), tvm.target.Target(target)
    ).inspect_source()


@pytest.mark.parametrize("backend", ["c", "c_host", "cuda", "hip", "metal"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_assume_does_not_emit_an_unknown_function(backend, roundtrip, tmp_path):
    code = _emit(backend, roundtrip=roundtrip)
    (tmp_path / "kernel.txt").write_text(code)
    assert "assume" not in code
    count = "arg.count[0]" if backend == "metal" else "count"
    assert f"{count} + 1" in code
    assert "hint_kernel" in code


@pytest.mark.parametrize("backend", ["c", "c_host", "cuda", "hip", "metal"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_assume_requires_a_boolean_condition(backend, roundtrip):
    with pytest.raises(TypeError, match="scalar boolean condition"):
        _emit(backend, invalid=True, roundtrip=roundtrip)


@pytest.mark.parametrize("roundtrip", [False, True])
def test_c_assume_compiles_and_preserves_the_continuation(tmp_path, roundtrip):
    source = tmp_path / "kernel.cc"
    output = tmp_path / "kernel.so"
    source.write_text(_emit("c", roundtrip=roundtrip))
    root = Path(__file__).resolve().parents[3]
    command = [
        "c++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-I{root / 'src'}",
        f"-I{find_include_path()}",
        *[f"-I{path}" for path in include_paths()],
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (tmp_path / "compile.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
    (tmp_path / "compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    library = ctypes.CDLL(str(output))
    function = library.hint_kernel
    function.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    function.restype = None
    answer = ctypes.c_int32(0)
    function(4, ctypes.byref(answer))
    assert answer.value == 5
