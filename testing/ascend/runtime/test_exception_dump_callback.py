"""Host-only tests of the production exception callback with a fake ACL API.

No NPU, CANN installation, or TileLang import is needed. The argument decoder
and callback are compiled directly from the production C++ source.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


class Tensor(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("elements", ctypes.c_uint64),
        ("dtype", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class Info(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("version", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("kernel_name", ctypes.c_char * 128),
        ("tensors", Tensor * 16),
    ]


class Descriptor(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("size", ctypes.c_size_t),
        ("format", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("address", ctypes.c_void_p),
        ("addr_type", ctypes.c_int32),
        ("placement", ctypes.c_int32),
        ("args_offset", ctypes.c_uint32),
        ("shape_num", ctypes.c_uint32),
        ("origin_shape_num", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 25),
        ("origin_shape", ctypes.c_uint64 * 25),
    ]


ACL_SHIM = r"""
#include "ascend/runtime/ascend_exception_dump.h"
#include "ascend/stubs/ascendcl.h"
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using tvm::tl::ascendcl::DumpTensorInfo;
static tvm::tl::ascendcl::ExceptionInfoCallback callback = nullptr;
static std::vector<DumpTensorInfo> saved;
static std::string kernel;
static int register_count = 0, register_result = 0, save_result = 0;
static int copies = 0, saves = 0;
struct Exception { void *args; uint32_t size; int32_t error; };

extern "C" {
int32_t aclrtExceptionInfoCallbackRegister(
    tvm::tl::ascendcl::ExceptionInfoCallback value) {
  ++register_count;
  if (register_result == 0) callback = value;
  return register_result;
}
int32_t aclrtGetArgsFromExceptionInfo(const void *raw, void **args,
                                    uint32_t *size) {
  auto *info = static_cast<const Exception *>(raw);
  *args = info->args;
  *size = info->size;
  return info->error;
}
int32_t aclrtMemcpy(void *dst, size_t dst_size, const void *src,
                   size_t count, int32_t kind) {
  ++copies;
  if (kind != 2 || count > dst_size) return 1;
  std::memcpy(dst, src, count);
  return 0;
}
int32_t acldumpSaveExceptionInfo(const char *name, const char *tag,
                               const DumpTensorInfo *tensors, size_t count) {
  ++saves;
  if (std::strcmp(tag, "tilelang") != 0) return 99;
  kernel = name;
  saved.assign(tensors, tensors + count);
  return save_result;
}
void reset_records() { saved.clear(); kernel.clear(); copies = saves = 0; }
int find_info(const void *args, size_t size) {
  tl::ascend::ExceptionDumpInfo info;
  return tvm::ascend::FindExceptionDumpInfo(args, size, &info);
}
int dump_args(const void *args, size_t size) {
  return tvm::ascend::DumpExceptionHostArgs(args, size);
}
size_t saved_count() { return saved.size(); }
const DumpTensorInfo *saved_tensor(size_t index) { return &saved.at(index); }
const char *saved_kernel() { return kernel.c_str(); }
int copy_count() { return copies; }
int save_count() { return saves; }
void set_save_result(int value) { save_result = value; }
int register_once(int value) {
  register_result = value;
  try { tvm::ascend::EnsureExceptionDumpCallback(); return 0; }
  catch (...) { return -1; }
}
int registration_count() { return register_count; }
void register_concurrently() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back(tvm::ascend::EnsureExceptionDumpCallback);
  }
  for (auto &thread : threads) thread.join();
}
void invoke_callback(void *args, uint32_t size, int32_t error) {
  Exception info{args, size, error};
  callback(&info);
}
}

namespace tvm { namespace tl { namespace ascendcl {
AscendCLAPI *AscendCLAPI::get_or_null() {
  static AscendCLAPI api = []() {
    AscendCLAPI value{};
    value.aclrtExceptionInfoCallbackRegister_ = &aclrtExceptionInfoCallbackRegister;
    value.aclrtGetArgsFromExceptionInfo_ = &aclrtGetArgsFromExceptionInfo;
    value.aclrtMemcpy_ = &aclrtMemcpy;
    value.acldumpSaveExceptionInfo_ = &acldumpSaveExceptionInfo;
    return value;
  }();
  return &api;
}
}}}
"""


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("the Ascend runtime is Linux-only")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    tmp = tmp_path_factory.mktemp("exception_callback")
    shim = tmp / "acl_shim.cc"
    shim.write_text(ACL_SHIM)
    library = tmp / "callback.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            "-DTILELANG_ASCEND_USE_STUBS=1",
            "-Wl,-Bsymbolic",
            f"-I{root / 'src'}",
            str(root / "src/ascend/runtime/ascend_exception_dump.cc"),
            str(shim),
            "-ldl",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Keep the fake ACL local, even when pytest already imported torch_npu.
    result = ctypes.CDLL(str(library))
    for name in ("find_info", "dump_args"):
        getattr(result, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        getattr(result, name).restype = ctypes.c_int32
    result.saved_count.restype = ctypes.c_size_t
    result.saved_tensor.argtypes = [ctypes.c_size_t]
    result.saved_tensor.restype = ctypes.POINTER(Descriptor)
    result.saved_kernel.restype = ctypes.c_char_p
    result.invoke_callback.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32]
    assert result.register_once(77) == -1
    assert result.register_once(0) == 0
    result.register_concurrently()
    assert result.registration_count() == 2
    return result


@pytest.fixture(autouse=True)
def clean_records(api):
    api.reset_records()
    api.set_save_result(0)


def metadata():
    info = Info()
    info.magic = 0x474E414C454C4954
    info.version = 1
    info.count = 1
    info.kernel_name = b"copy_kernel"
    info.tensors[0] = Tensor(0x12345678, 512, 128, 0, 0)
    return info


def test_descriptor_fields_and_unaligned_host_buffer(api):
    assert ctypes.sizeof(Info) == 656
    assert ctypes.sizeof(Descriptor) == 456
    info = metadata()
    storage = ctypes.create_string_buffer(b"!" + b"\0" * 8 + bytes(info))
    pointer = ctypes.cast(ctypes.addressof(storage) + 1, ctypes.c_void_p)
    assert api.dump_args(pointer, 8 + ctypes.sizeof(info)) == 0
    assert api.saved_count() == 1
    assert api.saved_kernel() == b"copy_kernel"
    tensor = api.saved_tensor(0).contents
    assert (tensor.type, tensor.format, tensor.dtype) == (0, 2, 0)
    assert (tensor.size, tensor.address) == (512, 0x12345678)
    assert (tensor.addr_type, tensor.placement, tensor.args_offset) == (2, 0, 0)
    assert (tensor.shape_num, tensor.origin_shape_num) == (1, 1)
    assert (tensor.shape[0], tensor.origin_shape[0]) == (128, 128)


@pytest.mark.parametrize("case", ["null", "truncated", "magic", "version", "count0", "count17", "name"])
def test_rejects_malformed_metadata(api, case):
    info = metadata()
    size = ctypes.sizeof(info)
    pointer = ctypes.byref(info)
    if case == "null":
        pointer = None
    elif case == "truncated":
        size -= 1
    elif case == "magic":
        info.magic = 0
    elif case == "version":
        info.version = 2
    elif case == "count0":
        info.count = 0
    elif case == "count17":
        info.count = 17
    else:
        info.kernel_name = b"x" * 128
    assert api.find_info(pointer, size) == 0
    assert api.dump_args(pointer, size) == -1
    assert api.save_count() == 0


def test_false_magic_does_not_hide_later_record(api):
    bad = metadata()
    bad.count = 17
    data = ctypes.create_string_buffer(bytes(bad) + bytes(metadata()))
    assert api.dump_args(data, 2 * ctypes.sizeof(Info)) == 0
    assert api.saved_count() == 1


@pytest.mark.parametrize("case", ["dtype", "bytes", "reserved", "overflow"])
def test_rejects_corrupt_tensor_descriptors(api, case):
    info = metadata()
    if case == "dtype":
        info.tensors[0].dtype = 999
    elif case == "bytes":
        info.tensors[0].size = 128
    elif case == "reserved":
        info.tensors[0].reserved = 1
    else:
        info.tensors[0].elements = 1 << 63
        info.tensors[0].size = 512
    assert api.find_info(ctypes.byref(info), ctypes.sizeof(info)) == 0
    assert api.dump_args(ctypes.byref(info), ctypes.sizeof(info)) == -1
    assert api.save_count() == 0


def test_skips_null_and_empty_tensors(api):
    info = metadata()
    info.count = 4
    info.tensors[1] = Tensor(0, 512, 128, 0, 0)
    info.tensors[2] = Tensor(0x23456789, 0, 128, 0, 0)
    info.tensors[3] = Tensor(0x34567890, 512, 0, 0, 0)
    assert api.dump_args(ctypes.byref(info), ctypes.sizeof(info)) == 0
    assert api.saved_count() == 1
    info.tensors[0].address = 0
    api.reset_records()
    assert api.dump_args(ctypes.byref(info), ctypes.sizeof(info)) == -2
    assert api.save_count() == 0


def test_propagates_save_error(api):
    info = metadata()
    api.set_save_result(42)
    assert api.dump_args(ctypes.byref(info), ctypes.sizeof(info)) == 42


def test_callback_copies_device_args_and_saves(api):
    info = metadata()
    api.invoke_callback(ctypes.byref(info), ctypes.sizeof(info), 0)
    assert api.copy_count() == 1
    assert api.save_count() == 1


@pytest.mark.parametrize("size,error", [(0, 0), (655, 0), (17 * 1024 * 1024, 0), (656, 1)])
def test_callback_rejects_invalid_argument_metadata(api, size, error):
    info = metadata()
    api.invoke_callback(ctypes.byref(info), size, error)
    assert api.copy_count() == 0
    assert api.save_count() == 0
