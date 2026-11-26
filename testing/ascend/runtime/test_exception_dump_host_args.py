"""Exercise the production Ascend argument packer without CANN or an NPU.

A subprocess builds a device module with the real code generator and a dummy
compiler callback. A local fake ACL library captures launches without executing
device code. No test-only entrypoints are added to the production runtime.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_FAKE_ACL = r"""
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static unsigned char captured[4096];
static size_t captured_size;
static uint32_t blocks, ubuf;
static int launches, registrations, registration_error;
static void (*callback)(void *);

int32_t aclrtBinaryLoadFromData(const void *data, size_t size,
                              const void *options, void **handle) {
  *handle = (void *)1;
  return 0;
}
int32_t aclrtBinaryGetFunction(void *binary, const char *name, void **function) {
  *function = (void *)2;
  return 0;
}
int32_t aclrtBinaryUnLoad(void *binary) { return 0; }
int32_t aclrtGetDevice(int32_t *device) { *device = 0; return 0; }
const char *aclGetRecentErrMsg(void) { return "fake ACL error"; }

struct Attribute { int32_t id; uint32_t value[4]; };
struct Config { struct Attribute *attrs; size_t count; };
int32_t aclrtLaunchKernelWithHostArgs(void *function, uint32_t num_blocks,
    void *stream, const struct Config *config, const void *args, size_t size,
    void *reserved, size_t reserved_size) {
  if (size > sizeof(captured)) return 99;
  memcpy(captured, args, size);
  captured_size = size;
  blocks = num_blocks;
  ubuf = config && config->count == 1 && config->attrs[0].id == 2
             ? config->attrs[0].value[0] : 0;
  ++launches;
  return 0;
}
int32_t aclrtExceptionInfoCallbackRegister(void (*value)(void *)) {
  ++registrations;
  if (registration_error) return registration_error;
  callback = value;
  return 0;
}
int32_t aclrtGetArgsFromExceptionInfo(const void *info, void **args, uint32_t *size) {
  return 1;
}
int32_t aclrtMemcpy(void *dst, size_t dst_size, const void *src, size_t size, int32_t kind) {
  return 1;
}
int32_t acldumpSaveExceptionInfo(const char *name, const char *tag,
                               const void *tensors, size_t count) { return 0; }

const void *test_args(void) { return captured; }
size_t test_args_size(void) { return captured_size; }
uint32_t test_blocks(void) { return blocks; }
uint32_t test_ubuf(void) { return ubuf; }
int test_launches(void) { return launches; }
int test_registrations(void) { return registrations; }
int test_has_callback(void) { return callback != NULL; }
void test_registration_error(int value) { registration_error = value; }
"""

_EXERCISE_RUNTIME = r"""
import ctypes
import struct
import sys
from pathlib import Path

import tilelang
from tilelang import tvm
from tvm import tirx
import tvm_ffi

# Exercise both a lazy local load and a runtime already visible globally.
acl = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL if sys.argv[2] == "global" else ctypes.RTLD_LOCAL)
acl.test_args.restype = ctypes.c_void_p
acl.test_args_size.restype = ctypes.c_size_t
acl.test_blocks.restype = acl.test_ubuf.restype = ctypes.c_uint32
acl.test_registration_error.argtypes = [ctypes.c_int]

tvm_ffi.register_global_func(
    "tilelang_callback_ascend_compile", f=lambda *args: "dummy device ELF", override=True
)
target = tvm.target.Target("ascend")
build = tvm_ffi.get_global_func("target.build.tilelang_ascend")

def make_module(enabled, metadata=None):
    params = [
        tirx.Var("x", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "global")),
        tirx.Var("left", "int32"),
        tirx.Var("wide", "int64"),
        tirx.Var("right", "int32"),
        tirx.Var("y", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global")),
    ]
    if enabled:
        params += [tirx.Var("x_numel", "uint64"), tirx.Var("y_numel", "uint64")]
    func = tirx.PrimFunc(params, tirx.Evaluate(0)).with_attr("global_symbol", "packing_kernel")
    func = func.with_attr("calling_conv", 2)
    func = func.with_attr(
        "tirx.kernel_launch_params", ["blockIdx.x", "blockIdx.y", "tirx.use_dyn_shared_memory"]
    )
    if enabled:
        func = func.with_attr("tl.ascend_exception_dump_tensors", [[0, 5, 1, 16], [4, 6, 0, 32]] if metadata is None else metadata)
    return build(tvm.IRModule({"packing_kernel": func}), target)

def bytes_captured():
    return ctypes.string_at(acl.test_args(), acl.test_args_size())

x, y = ctypes.c_void_p(0x112233445566), ctypes.c_void_p(0x778899AABBCC)
wide = (1 << 40) + 11
ordinary_args = [x, 17, wide, -9, y]
launch_args = [2, 3, 128]
ordinary = make_module(False)
ordinary["packing_kernel"](*ordinary_args, *launch_args)
assert acl.test_registrations() == 0
assert acl.test_args_size() == 40
ordinary_bytes = bytes_captured()
assert struct.unpack_from("<Q", ordinary_bytes, 0)[0] == x.value
assert struct.unpack_from("<i", ordinary_bytes, 8)[0] == 17
assert ordinary_bytes[12:16] == bytes(4)
assert struct.unpack_from("<q", ordinary_bytes, 16)[0] == wide
assert struct.unpack_from("<i", ordinary_bytes, 24)[0] == -9
assert ordinary_bytes[28:32] == bytes(4)
assert struct.unpack_from("<Q", ordinary_bytes, 32)[0] == y.value

module = make_module(True)
launch = module["packing_kernel"]
assert "ExceptionDumpInfo" in module.inspect_source("asc")
assert acl.test_registrations() == 0  # Build and function lookup are side-effect-free.

# Arity checks must precede raw argument reads and callback registration.
for bad_args in [[], ordinary_args + launch_args, ordinary_args + [13, 7] + launch_args + [99]]:
    try:
        launch(*bad_args)
    except Exception as error:
        assert "expects 10 arguments" in str(error), str(error)
    else:
        raise AssertionError("invalid launch arity was accepted")
assert acl.test_launches() == 1 and acl.test_registrations() == 0

# A failed registration must prevent the launch and allow a later retry.
acl.test_registration_error(73)
try:
    launch(*ordinary_args, 13, 7, *launch_args)
except Exception as error:
    assert "aclrtExceptionInfoCallbackRegister" in str(error), str(error)
else:
    raise AssertionError("callback registration failure was ignored")
assert acl.test_launches() == 1
assert acl.test_registrations() == 1
acl.test_registration_error(0)

for x_numel, y_numel in [(13, 7), (0, 257)]:
    launch(*ordinary_args, x_numel, y_numel, *launch_args)
    raw = bytes_captured()
    assert len(raw) == 56 + 656
    assert raw[:40] == ordinary_bytes
    assert struct.unpack_from("<QQ", raw, 40) == (x_numel, y_numel)
    magic, version, count, name = struct.unpack_from("<QII128s", raw, 56)
    assert (magic, version, count) == (0x474E414C454C4954, 1, 2)
    assert name.rstrip(b"\0") == b"packing_kernel"
    assert struct.unpack_from("<QQQii", raw, 56 + 144) == (x.value, x_numel * 2, x_numel, 1, 0)
    assert struct.unpack_from("<QQQii", raw, 56 + 176) == (y.value, y_numel * 4, y_numel, 0, 0)
    assert raw[56 + 208:] == bytes(14 * 32)
    assert acl.test_blocks() == 6
    assert acl.test_ubuf() == 128
    assert acl.test_registrations() == 2
    assert acl.test_has_callback()

# Export a C host container to exercise AscendModuleNode::SaveToBytes and the
# registered asc loader. Comparing the captured arguments checks that tensor
# metadata survives FunctionInfo serialization, independently of device output.
create_host = tvm_ffi.get_global_func("runtime.CSourceModuleCreate")
host = create_host('extern "C" void exception_dump_test_host() {}', "cc", [], [])
host.import_module(module)
archive = Path(sys.argv[1]).with_name("exception_dump_roundtrip.so")
host.export_library(str(archive))
restored = tvm.runtime.load_module(str(archive))
restored_launch = restored.get_function("packing_kernel", query_imports=True)
launch(*ordinary_args, 31, 19, *launch_args)
expected = bytes_captured()
before = acl.test_launches()
restored_launch(*ordinary_args, 31, 19, *launch_args)
assert acl.test_launches() == before + 1
assert bytes_captured() == expected
assert len(expected) == 56 + 656
assert struct.unpack_from("<QQQii", expected, 56 + 144) == (x.value, 62, 31, 1, 0)
assert struct.unpack_from("<QQQii", expected, 56 + 176) == (y.value, 76, 19, 0, 0)
assert acl.test_blocks() == 6
assert acl.test_ubuf() == 128
assert acl.test_registrations() == 2

# uint64 numel must be checked before converting elements to bytes.
before = acl.test_launches()
try:
    launch(*ordinary_args, 13, 1 << 62, *launch_args)
except Exception as error:
    assert "overflows" in str(error), str(error)
else:
    raise AssertionError("overflowing tensor byte size was accepted")
assert acl.test_launches() == before
print("Ascend exception dump host argument checks passed")

for invalid in [[[0, 5, 999, 16]], [[0, 5, 1, 24]], [[0, 2, 1, 16]]]:
    try:
        make_module(True, invalid)["packing_kernel"]
    except Exception as error:
        assert "exception dump" in str(error), str(error)
    else:
        raise AssertionError(f"invalid dump metadata accepted: {invalid}")

# Match the Cython host-boundary cases with the real native packer and ACL
# configuration builder. None of these invalid configurations may reach ACL.
for enabled in (False, True):
    boundary_launch = make_module(enabled)["packing_kernel"]
    arguments = ordinary_args + ([13, 17] if enabled else [])
    configurations = [
        (0, 1, 0, "positive"),
        (-1, 1, 0, "uint32 range"),
        (1 << 32, 1, 0, "uint32 range"),
        (65536, 65536, 0, "uint32 range"),
        (1, 1, -1, "UBUF"),
        (1, 1, 1 << 32, "UBUF"),
    ]
    for x_grid, y_grid, ub, message in configurations:
        before = acl.test_launches()
        try:
            boundary_launch(*arguments, x_grid, y_grid, ub)
        except Exception as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"invalid launch configuration accepted: {(x_grid, y_grid, ub)}")
        assert acl.test_launches() == before
    boundary_launch(*arguments, (1 << 32) - 1, 1, (1 << 32) - 1)
    assert acl.test_blocks() == acl.test_ubuf() == (1 << 32) - 1
print("Ascend native launch configuration boundary checks passed")
"""


@pytest.mark.parametrize("load_mode", ["local", "global"])
def test_exception_dump_packs_launch_arguments_with_fake_acl(tmp_path, load_mode):
    if sys.platform != "linux":
        pytest.skip("the Ascend runtime is Linux-only")
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None or shutil.which("c++") is None:
        pytest.skip("C and C++ compilers are required for the fake ACL library and module export")
    source = tmp_path / "fake_acl.c"
    library = tmp_path / "libascendcl.so"
    source.write_text(_FAKE_ACL)
    compiled = subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", str(library)], capture_output=True, text=True, timeout=60, check=False
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    script = tmp_path / "exercise_runtime.py"
    script.write_text(_EXERCISE_RUNTIME)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(tmp_path) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(script), str(library), load_mode], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Ascend exception dump host argument checks passed" in result.stdout
