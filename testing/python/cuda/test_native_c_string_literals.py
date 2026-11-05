"""Check C/CUDA literal emission with actual compilers and byte oracles."""

import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx, target

import tilelang
import tilelang.language as T


MESSAGES = [
    pytest.param("", id="empty"),
    pytest.param("plain 100% %s %n %%", id="percent"),
    pytest.param('a "quoted" message', id="quote"),
    pytest.param("path\\to\\file\\", id="backslash"),
    pytest.param("line one\nline two\r\n", id="newline"),
    pytest.param("\a\b\f\t\v\r\n", id="controls"),
    pytest.param("\x00" + "01234567F", id="nul-digit"),
    pytest.param("\x01F\x1ba\x7f0", id="control-hex"),
    pytest.param("内存Fé9", id="utf8-hex"),
]


def _emitted_assert_source(message, kind):
    flag = ir.Var("flag", "bool")
    if kind == "assert_stmt":
        statement = ir.AssertStmt(flag, ir.StringImm("ValueError"), [ir.StringImm(message)])
    else:
        statement = ir.Evaluate(ir.Call("tl.device_assert_with_msg", [flag, ir.StringImm(message)], ty="void"))
    function = tirx.Function(
        [flag],
        [statement],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "literal_probe", "calling_conv": 2}),
    )
    module = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(
        ir.IRModule({"literal_probe": function}), target.Target({"kind": "cuda", "arch": "sm_80"})
    )
    return module.inspect_source("cuda")


@pytest.mark.parametrize("message", MESSAGES)
@pytest.mark.parametrize("kind", ["call", "assert_stmt"])
def test_emitted_c_literal_preserves_every_byte(message, kind, tmp_path):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A C++ compiler is required to interpret emitted string literals")
    generated = _emitted_assert_source(message, kind)
    (tmp_path / "kernel.cu").write_text(generated)
    literal = re.search(r'device_assert_with_msg\([^,]*,\s*("(?:\\.|[^"\\])*")\s*\)', generated)
    assert literal is not None, generated
    # The literal comes directly from the owning emitter. sizeof prevents a
    # bad escape from making the byte oracle read past the generated array.
    source = tmp_path / "literal.cc"
    source.write_text(
        "#include <cstring>\n#include <cstddef>\n"
        'extern "C" std::size_t read_literal(unsigned char* out, std::size_t capacity) {\n'
        f"  static const char value[] = {literal.group(1)};\n"
        "  if (sizeof(value) <= capacity) std::memcpy(out, value, sizeof(value));\n"
        "  return sizeof(value);\n}\n"
    )
    library = tmp_path / "literal.so"
    command = [*compiler, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (tmp_path / "compile.json").write_text(json.dumps({"command": command, "exit_code": result.returncode}, indent=2))
    (tmp_path / "compile.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    loaded = ctypes.CDLL(str(library))
    function = loaded.read_literal
    function.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t]
    function.restype = ctypes.c_size_t
    expected = message.encode("utf-8") + b"\0"
    storage = (ctypes.c_ubyte * (len(expected) + 8))(*([0xA5] * (len(expected) + 8)))
    size = function(storage, len(storage))
    actual = bytes(storage)
    (tmp_path / "expected.bin").write_bytes(expected)
    (tmp_path / "actual.bin").write_bytes(actual)
    assert size == len(expected)
    assert actual[:size] == expected
    assert actual[size:] == b"\xa5" * (len(storage) - size)


BYTE_READER = """
__device__ unsigned char tl_test_string_byte(const char* text, int index) {
  return static_cast<unsigned char>(text[index]);
}
"""


def _literal_kernel(message):
    size = len(message.encode("utf-8")) + 1

    @T.prim_func
    def main(Output: T.Tensor((size,), "uint8")):
        with T.Kernel(1, threads=128):
            T.import_source(BYTE_READER)
            i = T.get_thread_binding()
            T.call_intrin("void", "tl.device_assert_with_msg", True, message)
            if i < size:
                Output[i] = T.call_extern("uint8", "tl_test_string_byte", message, i)

    return main


@pytest.mark.parametrize("message", MESSAGES)
def test_emitted_cuda_literal_bytes_match_input(message, tmp_path):
    if not torch.cuda.is_available() or shutil.which("nvcc") is None:
        pytest.skip("CUDA device and nvcc are required for literal byte execution")
    program = _literal_kernel(message)
    device = {"kind": "cuda", "arch": "sm_" + "".join(map(str, torch.cuda.get_device_capability()))}
    artifact = tilelang.lower(program, target=device, target_host={"kind": "llvm", "mcpu": "generic"})
    (tmp_path / "kernel.cu").write_text(artifact.kernel_source)
    compiled = tilelang.compile(program, target=device, target_host={"kind": "llvm", "mcpu": "generic"})
    expected = np.frombuffer(message.encode("utf-8") + b"\0", dtype=np.uint8).copy()
    output = torch.full((len(expected),), 0xA5, dtype=torch.uint8, device="cuda")
    compiled(output)
    actual = output.cpu().numpy()
    (tmp_path / "expected.bin").write_bytes(expected.tobytes())
    (tmp_path / "actual.bin").write_bytes(actual.tobytes())
    np.testing.assert_array_equal(actual, expected)
