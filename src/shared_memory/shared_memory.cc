/*!
 * \file shared_memory/shared_memory.cc
 * \brief VMM/IPC/multicast shared memory ops registered via TVM FFI.
 *
 * All functions are registered under the "tl.shared_memory.*" namespace
 * and accessed from Python via tvm_ffi.get_global_func().
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tvm;
using namespace tvm::ffi;

// ---------- helpers ----------

#define SM_CUDA_CHECK(cmd)                                                     \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      LOG_FATAL << "CUDA error " << __FILE__ << ":" << __LINE__ << " '"        \
                << cudaGetErrorString(e) << "'";                               \
    }                                                                          \
  } while (0)

#define SM_CU_CHECK(cmd)                                                       \
  do {                                                                         \
    CUresult e = (cmd);                                                        \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *err_str = nullptr;                                           \
      cuGetErrorString(e, &err_str);                                           \
      LOG_FATAL << "CU error " << __FILE__ << ":" << __LINE__ << " '"          \
                << (err_str ? err_str : "unknown") << "'";                     \
    }                                                                          \
  } while (0)

static void cu_mem_set_access_all(void *ptr, size_t size) {
  int device_count;
  SM_CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::vector<CUmemAccessDesc> access_desc(device_count);
  for (int idx = 0; idx < device_count; ++idx) {
    access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[idx].location.id = idx;
    access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  SM_CU_CHECK(
      cuMemSetAccess((CUdeviceptr)ptr, size, access_desc.data(), device_count));
}

static void cu_mem_set_access_devices(void *ptr, size_t size,
                                      int64_t num_devices) {
  std::vector<CUmemAccessDesc> access_desc(num_devices);
  for (int64_t idx = 0; idx < num_devices; ++idx) {
    access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[idx].location.id = (int)idx;
    access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  SM_CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc.data(),
                             (size_t)num_devices));
}

static size_t align_to_granularity(size_t size_raw, size_t granularity) {
  size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
  if (size == 0)
    size = granularity;
  return size;
}

// ---------- VMM malloc/free ----------

static int64_t vmm_malloc_impl(int64_t size_raw) {
  CUdevice device;
  SM_CU_CHECK(cuCtxGetDevice(&device));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.id = device;

  size_t granularity = 0;
  SM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t size = align_to_granularity((size_t)size_raw, granularity);

  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

  void *ptr = nullptr;
  SM_CU_CHECK(
      cuMemAddressReserve((CUdeviceptr *)&ptr, size, granularity, 0, 0));
  SM_CU_CHECK(cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0));
  cu_mem_set_access_all(ptr, size);

  return (int64_t)(uintptr_t)ptr;
}

static void vmm_free_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>((uintptr_t)ptr_val);
  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

  size_t size = 0;
  SM_CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));

  SM_CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  SM_CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  SM_CU_CHECK(cuMemRelease(handle));
}

// ---------- handle export/import ----------

// Returns serialized handle as Bytes.
// Format: 8 bytes size + sizeof(CUmemFabricHandle) bytes fabric handle.
static ffi::Bytes create_vmm_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>((uintptr_t)ptr_val);
  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

  size_t size = 0;
  SM_CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));

  CUmemFabricHandle fabric_handle;
  SM_CU_CHECK(cuMemExportToShareableHandle(&fabric_handle, handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC, 0));

  std::string raw(sizeof(size_t) + sizeof(CUmemFabricHandle), '\0');
  std::memcpy(&raw[0], &size, sizeof(size_t));
  std::memcpy(&raw[sizeof(size_t)], &fabric_handle, sizeof(CUmemFabricHandle));
  return ffi::Bytes(raw);
}

static int64_t open_vmm_handle_impl(ffi::Bytes handle_bytes) {
  ICHECK(handle_bytes.size() == sizeof(size_t) + sizeof(CUmemFabricHandle));
  const char *data = handle_bytes.data();

  size_t size = 0;
  std::memcpy(&size, data, sizeof(size_t));

  CUmemFabricHandle fabric_handle;
  std::memcpy(&fabric_handle, data + sizeof(size_t), sizeof(CUmemFabricHandle));

  CUmemGenericAllocationHandle alloc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(&alloc_handle, &fabric_handle,
                                             CU_MEM_HANDLE_TYPE_FABRIC));

  void *ptr = nullptr;
  SM_CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&ptr, size, 0, 0, 0));
  SM_CU_CHECK(cuMemMap((CUdeviceptr)ptr, size, 0, alloc_handle, 0));
  cu_mem_set_access_all(ptr, size);

  return (int64_t)(uintptr_t)ptr;
}

static void close_vmm_handle_impl(int64_t ptr_val) { vmm_free_impl(ptr_val); }

// ---------- IPC handle ----------

static ffi::Bytes create_ipc_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>((uintptr_t)ptr_val);
  cudaIpcMemHandle_t handle{};
  SM_CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
  return ffi::Bytes(reinterpret_cast<const char *>(handle.reserved),
                    CUDA_IPC_HANDLE_SIZE);
}

static int64_t open_ipc_handle_impl(ffi::Bytes handle_bytes) {
  ICHECK(handle_bytes.size() == CUDA_IPC_HANDLE_SIZE);
  cudaIpcMemHandle_t handle{};
  std::memcpy(handle.reserved, handle_bytes.data(), CUDA_IPC_HANDLE_SIZE);

  void *ptr = nullptr;
  SM_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)(uintptr_t)ptr;
}

static void close_ipc_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>((uintptr_t)ptr_val);
  SM_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
}

// ---------- support detection ----------

static bool supports_vmm_fabric_impl() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev;
    SM_CU_CHECK(cuDeviceGet(&dev, i));
    int supported = 0;
    SM_CU_CHECK(cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev));
    if (!supported)
      return false;
  }
  return true;
}

static bool supports_multicast_impl() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev;
    SM_CU_CHECK(cuDeviceGet(&dev, i));
    int supported = 0;
    SM_CU_CHECK(cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    if (!supported)
      return false;
  }
  return true;
}

// ---------- Multicast (NVSwitch) ----------
// Multi-process multi-GPU with fabric handles (same as vmm_malloc).
// Each process manages one GPU. MC handle shared via fabric export/import.

// Create multicast object with FABRIC handle type, returns handle as int64.
static int64_t mc_create_impl(int64_t size_raw, int64_t num_devices) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)num_devices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));

  size_t size = align_to_granularity((size_t)size_raw, granularity);
  prop.size = size;

  CUmemGenericAllocationHandle mc_handle;
  SM_CU_CHECK(cuMulticastCreate(&mc_handle, &prop));

  return (int64_t)mc_handle;
}

// Export multicast handle as fabric handle bytes (for sharing across processes)
static ffi::Bytes mc_export_handle_impl(int64_t mc_handle_val) {
  CUmemGenericAllocationHandle mc_handle =
      (CUmemGenericAllocationHandle)mc_handle_val;

  CUmemFabricHandle fabric_handle;
  SM_CU_CHECK(cuMemExportToShareableHandle(&fabric_handle, mc_handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC, 0));

  return ffi::Bytes(reinterpret_cast<const char *>(&fabric_handle),
                    sizeof(CUmemFabricHandle));
}

// Import multicast handle from fabric handle bytes, returns handle as int64.
static int64_t mc_import_handle_impl(ffi::Bytes handle_bytes) {
  ICHECK(handle_bytes.size() == sizeof(CUmemFabricHandle));

  CUmemFabricHandle fabric_handle;
  std::memcpy(&fabric_handle, handle_bytes.data(), sizeof(CUmemFabricHandle));

  CUmemGenericAllocationHandle mc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(&mc_handle, &fabric_handle,
                                             CU_MEM_HANDLE_TYPE_FABRIC));

  return (int64_t)mc_handle;
}

// Add a device to the multicast object
static void mc_add_device_impl(int64_t mc_handle_val, int64_t device_id) {
  CUmemGenericAllocationHandle mc_handle =
      (CUmemGenericAllocationHandle)mc_handle_val;
  CUdevice device;
  SM_CU_CHECK(cuDeviceGet(&device, (int)device_id));
  SM_CU_CHECK(cuMulticastAddDevice(mc_handle, device));
}

// Bind a physical memory VA (from vmm_malloc) to the multicast object
static void mc_bind_mem_impl(int64_t mc_handle_val, int64_t ptr_val,
                             int64_t size) {
  CUmemGenericAllocationHandle mc_handle =
      (CUmemGenericAllocationHandle)mc_handle_val;
  void *ptr = reinterpret_cast<void *>((uintptr_t)ptr_val);

  // Retrieve the physical allocation handle from the mapped pointer
  CUmemGenericAllocationHandle phys_handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&phys_handle, ptr));

  // Bind to multicast
  SM_CU_CHECK(
      cuMulticastBindMem(mc_handle, 0, phys_handle, 0, (size_t)size, 0));

  // Release the temporary handle reference
  SM_CU_CHECK(cuMemRelease(phys_handle));
}

// Map multicast object to a VA, returns mc_ptr. Does NOT release handle.
static int64_t mc_map_impl(int64_t mc_handle_val, int64_t size_raw,
                           int64_t num_devices) {
  CUmemGenericAllocationHandle mc_handle =
      (CUmemGenericAllocationHandle)mc_handle_val;

  CUmulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)num_devices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t size = align_to_granularity((size_t)size_raw, granularity);

  void *mc_ptr = nullptr;
  SM_CU_CHECK(
      cuMemAddressReserve((CUdeviceptr *)&mc_ptr, size, granularity, 0, 0));
  SM_CU_CHECK(cuMemMap((CUdeviceptr)mc_ptr, size, 0, mc_handle, 0));
  cu_mem_set_access_devices(mc_ptr, size, num_devices);

  return (int64_t)(uintptr_t)mc_ptr;
}

// Release a multicast handle (call after map)
static void mc_release_handle_impl(int64_t mc_handle_val) {
  CUmemGenericAllocationHandle mc_handle =
      (CUmemGenericAllocationHandle)mc_handle_val;
  SM_CU_CHECK(cuMemRelease(mc_handle));
}

// Free multicast VA mapping
static void mc_unmap_impl(int64_t mc_ptr_val, int64_t size_raw,
                          int64_t num_devices) {
  void *ptr = reinterpret_cast<void *>((uintptr_t)mc_ptr_val);

  CUmulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)num_devices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));
  size_t size = align_to_granularity((size_t)size_raw, granularity);

  SM_CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  SM_CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
}

// Get the aligned size for multicast
static int64_t mc_get_aligned_size_impl(int64_t size_raw, int64_t num_devices) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)num_devices;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));
  return (int64_t)align_to_granularity((size_t)size_raw, granularity);
}

// ---------- sync helpers ----------

// Synchronize VMM handles: open all peer handles and write pointers to GPU.
// peer_handles is a comma-separated list of hex-encoded handle bytes (or "SELF"
// for local rank). We pass individual handle open results back via buffer_ptrs.
// packed_handles: num_ranks concatenated raw handle bytes
static void sync_vmm_handles_impl(int64_t rank, int64_t num_ranks,
                                  int64_t buffer_ptrs_gpu_addr,
                                  ffi::Bytes packed_handles) {
  const size_t handle_size = sizeof(size_t) + sizeof(CUmemFabricHandle);
  ICHECK(packed_handles.size() == handle_size * (size_t)num_ranks);

  std::vector<void *> buffer_ptrs(num_ranks, nullptr);

  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i != rank) {
      ffi::Bytes h(packed_handles.data() + i * handle_size, handle_size);
      buffer_ptrs[i] =
          reinterpret_cast<void *>((uintptr_t)open_vmm_handle_impl(h));
    }
  }

  void **gpu_ptr = reinterpret_cast<void **>((uintptr_t)buffer_ptrs_gpu_addr);
  SM_CUDA_CHECK(cudaMemcpy(gpu_ptr, buffer_ptrs.data(),
                           sizeof(void *) * buffer_ptrs.size(),
                           cudaMemcpyHostToDevice));
  SM_CUDA_CHECK(cudaDeviceSynchronize());
}

static void sync_ipc_handles_impl(int64_t rank, int64_t num_ranks,
                                  int64_t buffer_ptrs_gpu_addr,
                                  ffi::Bytes packed_handles) {
  ICHECK(packed_handles.size() == CUDA_IPC_HANDLE_SIZE * (size_t)num_ranks);

  std::vector<void *> buffer_ptrs(num_ranks, nullptr);

  for (int64_t i = 0; i < num_ranks; ++i) {
    if (i != rank) {
      ffi::Bytes h(packed_handles.data() + i * CUDA_IPC_HANDLE_SIZE,
                   CUDA_IPC_HANDLE_SIZE);
      buffer_ptrs[i] =
          reinterpret_cast<void *>((uintptr_t)open_ipc_handle_impl(h));
    }
  }

  void **gpu_ptr = reinterpret_cast<void **>((uintptr_t)buffer_ptrs_gpu_addr);
  SM_CUDA_CHECK(cudaMemcpy(gpu_ptr, buffer_ptrs.data(),
                           sizeof(void *) * buffer_ptrs.size(),
                           cudaMemcpyHostToDevice));
  SM_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------- Registration ----------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // VMM
  refl::GlobalDef().def("tl.shared_memory.vmm_malloc", vmm_malloc_impl);
  refl::GlobalDef().def("tl.shared_memory.vmm_free", vmm_free_impl);
  refl::GlobalDef().def("tl.shared_memory.create_vmm_handle",
                        create_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.open_vmm_handle",
                        open_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.close_vmm_handle",
                        close_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.sync_vmm_handles",
                        sync_vmm_handles_impl);

  // IPC
  refl::GlobalDef().def("tl.shared_memory.create_ipc_handle",
                        create_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.open_ipc_handle",
                        open_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.close_ipc_handle",
                        close_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.sync_ipc_handles",
                        sync_ipc_handles_impl);

  // Support detection
  refl::GlobalDef().def("tl.shared_memory.supports_vmm_fabric",
                        supports_vmm_fabric_impl);
  refl::GlobalDef().def("tl.shared_memory.supports_multicast",
                        supports_multicast_impl);

  // Multicast (NVSwitch)
  refl::GlobalDef().def("tl.shared_memory.mc_create", mc_create_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_export_handle",
                        mc_export_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_import_handle",
                        mc_import_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_add_device", mc_add_device_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_bind_mem", mc_bind_mem_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_map", mc_map_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_release_handle",
                        mc_release_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_unmap", mc_unmap_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_get_aligned_size",
                        mc_get_aligned_size_impl);
}
