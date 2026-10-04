"""Exercise HIP stub discovery without depending on a host ROCm installation."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope="module", params=["default", "vendored"])
def hip_stub_probe(request, tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("Requires the POSIX HIP stub and ELF linker wrappers")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Requires a host C++ compiler")

    root = Path(__file__).resolve().parents[3]
    directory = tmp_path_factory.mktemp("hip_stub_probe")
    source = directory / "probe.cc"
    source.write_text(
        r"""
#include "hip.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

// Each subprocess starts with a fresh lazy-loader singleton. Linker wrappers
// isolate the test from ROCm libraries installed or loaded on the host.
static int mode;
static void *const mock_handle = reinterpret_cast<void *>(1);
static void unused_symbol() {}
static hipError_t device_count(int *count) {
  if (mode == 2) return hipErrorUnknown;
  *count = 2;
  return hipSuccess;
}
extern "C" void *__wrap_dlopen(const char *, int) {
  return mode == 0 ? nullptr : mock_handle;
}
extern "C" void *__wrap_dlsym(void *handle, const char *name) {
  if (handle != mock_handle || mode == 1) return nullptr;
  if (std::strcmp(name, "hipGetDeviceCount") == 0) {
    return reinterpret_cast<void *>(&device_count);
  }
  return reinterpret_cast<void *>(&unused_symbol);
}
extern "C" char *__wrap_dlerror() { return nullptr; }

int main(int argc, char **argv) {
  assert(argc == 2);
  mode = std::atoi(argv[1]);
  assert(hipGetDeviceCount(nullptr) == hipErrorInvalidValue);
  for (int i = 0; i < 2; ++i) {
    int count = -1;
    hipError_t status = hipGetDeviceCount(&count);
    if (mode == 3) {
      assert(status == hipSuccess && count == 2);
    } else {
      assert(status != hipSuccess && count == 0);
      if (mode == 2) assert(status == hipErrorUnknown);
    }
  }
}
""",
        encoding="utf-8",
    )
    executable = directory / "probe"
    command = [compiler, "-std=c++17", "-D__HIP_PLATFORM_AMD__", "-I" + str(root / "src/rocm/stubs")]
    if request.param == "vendored":
        command.append("-I" + str(root / "3rdparty/hip-headers/include"))
    command.extend(
        [
            str(root / "src/rocm/stubs/hip.cc"),
            str(source),
            "-Wl,--wrap=dlopen,--wrap=dlsym,--wrap=dlerror",
            "-ldl",
            "-o",
            str(executable),
        ]
    )
    subprocess.run(command, check=True)
    return executable


@pytest.mark.parametrize("mode", [0, 1, 2, 3], ids=["missing-runtime", "missing-symbols", "init-error", "success"])
def test_hip_stub_device_count(hip_stub_probe, mode):
    subprocess.run([str(hip_stub_probe), str(mode)], check=True)
