/*!
 * \file ascend_device_api.cc
 * \brief Minimal DeviceAPI for the Ascend NPU, registered as
 *        "device_api.ext_dev".
 *
 * The Ascend target reports DLPack device type kDLExtDev (12) — the same type
 * torch_npu assigns to NPU tensors. Because kDLExtDev != kDLCPU,
 * `MakePackedAPI` injects a `__tvm_set_device(12, dev_id)` call into the host
 * stub (see DeviceAPI::NeedSetDevice). That call resolves the DeviceAPI factory
 * "device_api." + DLDeviceType2Str(kDLExtDev) == "device_api.ext_dev".
 *
 * On the tvm_ffi execution path the actual device context, allocation and
 * stream are owned by PyTorch (torch_npu), exactly as in the legacy cython
 * path. So this DeviceAPI only needs to exist and answer kExist; SetDevice and
 * StreamSync are no-ops. Allocation / copy are not used on this path and throw
 * if reached, to surface accidental misuse rather than silently corrupt state.
 */
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {

class AscendDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(Device dev) final {}

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any *rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void *AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not allocate device memory; NPU buffers are "
           "managed by PyTorch (torch_npu).";
    TVM_FFI_UNREACHABLE();
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not free device memory; NPU buffers are "
           "managed by PyTorch (torch_npu).";
    TVM_FFI_UNREACHABLE();
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  static AscendDeviceAPI *Global() {
    // Explicit new to avoid exit-time destruction of global state; the OS
    // reclaims it at process exit (matches CPUDeviceAPI).
    static auto *inst = new AscendDeviceAPI();
    return inst;
  }

protected:
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t size, Device dev_from,
                      Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not implement CopyDataFromTo; copies are "
           "performed by PyTorch (torch_npu).";
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.ext_dev",
                               [](ffi::PackedArgs args, ffi::Any *rv) {
                                 DeviceAPI *ptr = AscendDeviceAPI::Global();
                                 *rv = static_cast<void *>(ptr);
                               });
}

} // namespace runtime
} // namespace tvm
