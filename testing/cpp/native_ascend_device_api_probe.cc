/*!
 * \file native_ascend_device_api_probe.cc
 * \brief Exercise the real Ascend DeviceAPI without emulating NPU hardware.
 */
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>

#include <array>
#include <cstdint>

namespace tvm::tl {
namespace {

Device AscendDevice(int index) { return Device{kDLExtDev, index}; }

runtime::DeviceAPI *AscendAPI(int index) {
  return runtime::DeviceAPI::Get(AscendDevice(index));
}

void CopyWithSentinels(int from_type, int to_type) {
  const std::array<int32_t, 2> original_from{17, 29};
  const std::array<int32_t, 2> original_to{101, 103};
  auto from_data = original_from;
  auto to_data = original_to;
  int64_t shape = 1;
  DLTensor from{from_data.data(),
                {static_cast<DLDeviceType>(from_type), 0},
                1,
                {kDLInt, 32, 1},
                &shape,
                nullptr,
                sizeof(int32_t)};
  DLTensor to{to_data.data(),
              {static_cast<DLDeviceType>(to_type), 0},
              1,
              {kDLInt, 32, 1},
              &shape,
              nullptr,
              sizeof(int32_t)};
  try {
    AscendAPI(0)->CopyDataFromTo(&from, &to, nullptr);
  } catch (const ffi::Error &) {
    TVM_FFI_CHECK(from_data == original_from && to_data == original_to,
                  RuntimeError)
        << "Rejected copy must not alter either borrowed buffer";
    throw;
  }
  TVM_FFI_THROW(RuntimeError)
      << "Unsupported Ascend copy unexpectedly returned";
}

void FreeBorrowedWorkspace(bool through_c_api) {
  std::array<int32_t, 2> borrowed{107, 109};
  const auto before = borrowed;
  try {
    if (through_c_api) {
      TVM_FFI_CHECK_SAFE_CALL(
          TVMBackendFreeWorkspace(kDLExtDev, 0, borrowed.data()));
    } else {
      AscendAPI(0)->FreeWorkspace(AscendDevice(0), borrowed.data());
    }
  } catch (const ffi::Error &) {
    TVM_FFI_CHECK(borrowed == before, RuntimeError)
        << "Rejected workspace free must not alter the caller's buffer";
    throw;
  }
  TVM_FFI_THROW(RuntimeError)
      << "Unsupported Ascend workspace free unexpectedly returned";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.testing.AscendDeviceAPI.Address",
           [](int index) {
             return static_cast<int64_t>(
                 reinterpret_cast<uintptr_t>(AscendAPI(index)));
           })
      .def("tl.testing.AscendDeviceAPI.SetDevice",
           [](int index) {
             TVM_FFI_CHECK_SAFE_CALL(TVMBackendSetDevice(kDLExtDev, index));
           })
      .def("tl.testing.AscendDeviceAPI.GetDeviceIndex",
           []() {
             int index = -1;
             TVM_FFI_CHECK_SAFE_CALL(
                 TVMBackendGetDeviceIndex(kDLExtDev, &index));
             return index;
           })
      .def("tl.testing.AscendDeviceAPI.AllocWorkspace",
           [](int64_t size) {
             void *ptr =
                 TVMBackendAllocWorkspace(kDLExtDev, 0, size, kDLUInt, 8);
             TVM_FFI_CHECK_SAFE_CALL(ptr == nullptr ? -1 : 0);
             TVM_FFI_THROW(RuntimeError) << "Unsupported Ascend workspace "
                                            "allocation unexpectedly returned";
           })
      .def("tl.testing.AscendDeviceAPI.FreeWorkspace", FreeBorrowedWorkspace)
      .def("tl.testing.AscendDeviceAPI.Copy", CopyWithSentinels)
      .def("tl.testing.AscendDeviceAPI.GetStream",
           [](int index) {
             return static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                 AscendAPI(index)->GetCurrentStream(AscendDevice(index))));
           })
      .def("tl.testing.AscendDeviceAPI.EnvStream", [](int index) {
        return static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(TVMFFIEnvGetStream(kDLExtDev, index)));
      });
}

} // namespace
} // namespace tvm::tl
