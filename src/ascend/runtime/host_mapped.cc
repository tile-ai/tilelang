/*!
 * \file host_mapped.cc
 * \brief Runtime queries for explicit mapped host parameters.
 */
#include "ascend/stubs/ascendcl.h"

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {

// Registration and lifetime belong to the caller, just as on CUDA. Query on
// every eager launch rather than caching addresses that can be unregistered.
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.ascend.current_device",
           []() {
             int32_t device_id;
             int32_t result = aclrtGetDevice(&device_id);
             TVM_FFI_ICHECK_EQ(result, 0)
                 << "Cannot determine the current Ascend device: " << result;
             return device_id;
           })
      .def("tl.ascend.get_host_mapped_pointer", [](void *host_ptr) -> void * {
        // Empty buffers have no storage to map.
        if (host_ptr == nullptr) {
          return nullptr;
        }
        void *device_ptr = nullptr;
        int32_t result = aclrtHostGetDevicePointer(host_ptr, &device_ptr, 0);
        if (result != 0) {
          // We report this query failure here. Consume its thread-local ACL
          // error so a later Cython triple-chevron launch does not report it
          // again as a kernel failure. ACL_RT_THREAD_LEVEL = 0.
          aclrtGetLastError(0);
          TVM_FFI_THROW(ValueError)
              << "Host-mapped buffer requires Ascend-registered host memory "
                 "accessible from the current NPU. Enable "
                 "PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True before "
                 "initializing torch_npu, or register the storage explicitly. "
                 "The CANN runtime and device must support "
                 "aclrtHostGetDevicePointer (error "
              << result << ")";
        }
        return device_ptr;
      });
}

} // namespace tvm::tl
