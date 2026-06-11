/*!
 * \file tilescale_cuda_module.cc
 * \brief TileScale extended CUDA module with distributed table initialization
 * support.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

#include "runtime/cuda/cuda_common.h"
#include "runtime/file_utils.h"
#include "runtime/metadata.h"
#include "runtime/pack_args.h"
#include "runtime/thread_storage_scope.h"
#include "cuda/runtime.h"
#include "cuda/stubs/dynlib.h"
#include "support/bytes_io.h"
#include "support/check.h"

namespace tvm {
namespace runtime {
namespace {

// Maximum number of GPUs supported (same as TVM's default)
constexpr int kTileScaleMaxNumGPUs = 32;

inline void EnsureCurrentDeviceContext(int device_id) {
  CUDA_CALL(cudaSetDevice(device_id));
}

void *LoadLibCuda() {
#if defined(_WIN32) && !defined(__CYGWIN__)
  constexpr const char *kLibCudaPaths[] = {"nvcuda.dll"};
#else
  constexpr const char *kLibCudaPaths[] = {"libcuda.so.1", "libcuda.so"};
#endif
  for (const char *path : kLibCudaPaths) {
    if (void *handle = tvm::tl::stubs::dynlib_open(path)) {
      return handle;
    }
  }
  return nullptr;
}

template <typename T> T LoadRequiredCudaSymbol(void *handle, const char *name) {
  auto *sym = tvm::tl::stubs::dynlib_sym(handle, name);
  if (sym == nullptr) {
    const char *error = tvm::tl::stubs::dynlib_error();
    LOG(FATAL) << "Failed to load CUDA driver symbol '" << name
               << "': " << (error ? error : "unknown");
  }
  return reinterpret_cast<T>(sym);
}

struct TileScaleCudaContextAPI {
  decltype(&cuDeviceGet) cuDeviceGet_;
  decltype(&cuDevicePrimaryCtxRetain) cuDevicePrimaryCtxRetain_;
  decltype(&cuCtxSetCurrent) cuCtxSetCurrent_;
  decltype(&cuDevicePrimaryCtxRelease) cuDevicePrimaryCtxRelease_;
  decltype(&cuModuleGetGlobal) cuModuleGetGlobal_;

  static TileScaleCudaContextAPI *Get() {
    static TileScaleCudaContextAPI api = [] {
      void *handle = LoadLibCuda();
      if (handle == nullptr) {
        LOG(FATAL) << "CUDA driver library (libcuda.so) not found.";
      }
      TileScaleCudaContextAPI api{};
      api.cuDeviceGet_ =
          LoadRequiredCudaSymbol<decltype(api.cuDeviceGet_)>(
              handle, "cuDeviceGet");
      api.cuDevicePrimaryCtxRetain_ =
          LoadRequiredCudaSymbol<decltype(api.cuDevicePrimaryCtxRetain_)>(
              handle, "cuDevicePrimaryCtxRetain");
      api.cuCtxSetCurrent_ =
          LoadRequiredCudaSymbol<decltype(api.cuCtxSetCurrent_)>(
              handle, "cuCtxSetCurrent");
      api.cuDevicePrimaryCtxRelease_ =
          LoadRequiredCudaSymbol<decltype(api.cuDevicePrimaryCtxRelease_)>(
              handle, "cuDevicePrimaryCtxRelease");
      api.cuModuleGetGlobal_ =
          LoadRequiredCudaSymbol<decltype(api.cuModuleGetGlobal_)>(
              handle, "cuModuleGetGlobal_v2");
      return api;
    }();
    return &api;
  }
};

} // namespace

// Forward declaration
class TileScaleCUDAModuleNode;

// TileScale: Initialize distributed table by copying host data to device
// meta_data symbol
class TileScaleInitDistributedTable {
public:
  // meta_data symbol size: 1024 * sizeof(uint64_t)
  static constexpr size_t kMetaDataSize = 1024 * sizeof(uint64_t);

  TileScaleInitDistributedTable(TileScaleCUDAModuleNode *m,
                                ffi::ObjectPtr<ffi::Object> sptr)
      : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  // args: host_table_ptr (void*), table_size (int64_t), stream (void*)
  void operator()(const ffi::PackedArgs &args, ffi::Any *rv) const;

private:
  // internal module
  TileScaleCUDAModuleNode *m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUdeviceptr, kTileScaleMaxNumGPUs> pcache_;
};

/*!
 * \brief TileScale extended CUDA module with distributed table support.
 *
 * This module extends TVM's CUDAModule with:
 * - __tilescale_init_table: Initialize distributed table by copying host
 *   data to the device's meta_data symbol
 */
class TileScaleCUDAModuleNode : public ffi::ModuleObj {
public:
  explicit TileScaleCUDAModuleNode(
      ffi::Bytes data, ffi::String fmt,
      ffi::Map<ffi::String, FunctionInfo> fmap,
      ffi::Map<ffi::String, ffi::String> source)
      : data_(std::move(data)), fmt_(std::move(fmt)), fmap_(std::move(fmap)),
        source_(std::move(source)) {
    std::fill(module_.begin(), module_.end(), nullptr);
    std::fill(context_.begin(), context_.end(), nullptr);
  }

  ~TileScaleCUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        BindPrimaryContext(static_cast<int>(i));
        CUresult result = cuModuleUnload(module_[i]);
        (void)result;
      }
      if (context_[i] != nullptr) {
        ReleasePrimaryContext(static_cast<int>(i));
      }
    }
  }

  const char *kind() const final { return "tilescale_cuda"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final;

  void WriteToFile(const ffi::String &file_name,
                   const ffi::String &format) const final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cu") {
      auto cuda_source = source_.Get("cuda");
      ICHECK(cuda_source.has_value());
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string(cuda_source.value()));
    } else {
      ICHECK_EQ(fmt, std::string(fmt_)) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, std::string(data_));
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(data_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    if (format == fmt_) {
      return ffi::String(data_.data(), data_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("cuda"); it != source_.end()) {
        return (*it).second;
      }
      if (fmt_ == "ptx" || fmt_ == "cuda") {
        return ffi::String(data_.data(), data_.size());
      }
    }
    return ffi::String();
  }

  // Get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string &func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    BindPrimaryContext(device_id);
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.data()));
      static auto nvshmem_init_hook =
          ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void *>(module_[device_id]));
      }
    }
    CUfunction func;
    CUresult result =
        cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetFunction " << func_name
                 << " failed with error: " << msg;
    }
    return func;
  }

  // Get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id, const std::string &global_name,
                        size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    BindPrimaryContext(device_id);
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.data()));
      static auto nvshmem_init_hook =
          ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void *>(module_[device_id]));
      }
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = TileScaleCudaContextAPI::Get()->cuModuleGetGlobal_(
        &global, &nbytes, module_[device_id], global_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetGlobal " << global_name
                 << " failed with error: " << msg;
    }
    ICHECK_EQ(nbytes, expect_nbytes);
    return global;
  }

  void BindPrimaryContext(int device_id) {
    EnsureCurrentDeviceContext(device_id);
    auto *api = TileScaleCudaContextAPI::Get();
    CUdevice device;
    CUDA_DRIVER_CALL(api->cuDeviceGet_(&device, device_id));
    if (context_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(api->cuDevicePrimaryCtxRetain_(&context_[device_id],
                                                      device));
    }
    CUDA_DRIVER_CALL(api->cuCtxSetCurrent_(context_[device_id]));
  }

  void ReleasePrimaryContext(int device_id) {
    auto *api = TileScaleCudaContextAPI::Get();
    CUdevice device;
    CUresult result = api->cuDeviceGet_(&device, device_id);
    if (result == CUDA_SUCCESS) {
      result = api->cuDevicePrimaryCtxRelease_(device);
    }
    (void)result;
    context_[device_id] = nullptr;
  }

private:
  ffi::Bytes data_;
  ffi::String fmt_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  ffi::Map<ffi::String, ffi::String> source_;
  std::array<CUmodule, kTileScaleMaxNumGPUs> module_;
  std::array<CUcontext, kTileScaleMaxNumGPUs> context_;
  std::mutex mutex_;
};

// Implementation of TileScaleInitDistributedTable::operator()
void TileScaleInitDistributedTable::operator()(const ffi::PackedArgs &args,
                                               ffi::Any *rv) const {
  // Accept int64_t from Python and cast to pointers internally
  // This is necessary because TVM FFI doesn't auto-convert int to void*
  int64_t host_table_ptr = args[0].cast<int64_t>();
  int64_t table_size = args[1].cast<int64_t>();

  void *host_table = reinterpret_cast<void *>(host_table_ptr);
  auto *table_ptr = reinterpret_cast<const uint64_t *>(host_table);
  tl::SetRemoteTensorMapMetaData(table_ptr, static_cast<size_t>(table_size));

  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  // Get the device pointer for meta_data symbol (lazy initialization)
  if (pcache_[device_id] == 0) {
    pcache_[device_id] = m_->GetGlobal(device_id, "meta_data", kMetaDataSize);
  }

  // Copy data from host to device constant memory. The symbol lives in a
  // dynamically loaded CUmodule, so use the resolved device pointer directly.
  size_t bytes = static_cast<size_t>(table_size) * sizeof(uint64_t);
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(pcache_[device_id]),
                       host_table, bytes, cudaMemcpyHostToDevice));

  // Return success
  *rv = 0;
}

// Wrapped function class similar to TVM's CUDAWrappedFunc
class TileScaleCUDAWrappedFunc {
public:
  void Init(TileScaleCUDAModuleNode *m, ffi::ObjectPtr<ffi::Object> sptr,
            const std::string &func_name, size_t num_void_args,
            const ffi::Array<ffi::String> &launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    std::fill(dyn_smem_initialized_.begin(), dyn_smem_initialized_.end(),
              false);
    std::fill(cluster_attr_initialized_.begin(),
              cluster_attr_initialized_.end(), false);
    use_dyn_shared_memory_ = false;
    for (const auto &tag : launch_param_tags) {
      if (tag == launch_param::kUseDynamicSharedMemoryTag) {
        use_dyn_shared_memory_ = true;
        break;
      }
    }
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }

  void operator()(ffi::PackedArgs args, ffi::Any *rv, void **void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    EnsureCurrentDeviceContext(device_id);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);

    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    bool need_dyn_attr = use_dyn_shared_memory_ || (wl.dyn_shmem_size > 0);
    if (need_dyn_attr) {
      if (!dyn_smem_initialized_[device_id] ||
          dyn_smem_last_[device_id] != wl.dyn_shmem_size) {
        CUresult attr_set = cuFuncSetAttribute(
            fcache_[device_id], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            wl.dyn_shmem_size);
        if (attr_set != CUDA_SUCCESS) {
          LOG(FATAL)
              << "Failed to set the allowed dynamic shared memory size to "
              << wl.dyn_shmem_size;
        }
        dyn_smem_last_[device_id] = wl.dyn_shmem_size;
        dyn_smem_initialized_[device_id] = true;
      }
    }
    CUstream strm =
        static_cast<CUstream>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    CUresult result;

    ICHECK(wl.grid_dim(0) > 0 && wl.grid_dim(1) > 0 && wl.grid_dim(2) > 0)
        << "CUDALaunch Error: grid dimension must be positive, but got"
        << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << ","
        << wl.grid_dim(2) << ")"
        << " in kernel " << func_name_
        << ". A zero grid dimension is often caused by a dynamic shape"
        << " (e.g. num_tokens) being 0 at runtime.";

    if (wl.use_cluster_launch()) {
      CUlaunchConfig config{};
      CUlaunchAttribute attribute[2]{};
      attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attribute[0].value.clusterDim.x = wl.cluster_dim[0];
      attribute[0].value.clusterDim.y = wl.cluster_dim[1];
      attribute[0].value.clusterDim.z = wl.cluster_dim[2];
      attribute[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[1].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 2;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      if (!cluster_attr_initialized_[device_id]) {
        CUresult attr_result = cuFuncSetAttribute(
            fcache_[device_id],
            CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
        if (attr_result != CUDA_SUCCESS) {
          const char *msg;
          cuGetErrorName(attr_result, &msg);
          LOG(FATAL) << "Failed to set cluster attribute for " << func_name_
                     << ": " << msg;
        }
        cluster_attr_initialized_[device_id] = true;
      }

      result = cuLaunchKernelEx(&config, fcache_[device_id], void_args,
                                nullptr);
    } else if (launch_param_config_.use_programtic_dependent_launch()) {
      CUlaunchConfig config{};
      CUlaunchAttribute attribute[1]{};
      attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[0].value.programmaticStreamSerializationAllowed = 1;

      config.attrs = attribute;
      config.numAttrs = 1;
      config.hStream = strm;
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
      config.sharedMemBytes = wl.dyn_shmem_size;

      result = cuLaunchKernelEx(&config, fcache_[device_id], void_args,
                                nullptr);
    } else if (launch_param_config_.use_cooperative_launch()) {
      result = cuLaunchCooperativeKernel(
          fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
          wl.block_dim(0), wl.block_dim(1), wl.block_dim(2),
          wl.dyn_shmem_size, strm, void_args);
    } else {
      result = cuLaunchKernel(
          fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
          wl.block_dim(0), wl.block_dim(1), wl.block_dim(2),
          wl.dyn_shmem_size, strm, void_args, nullptr);
    }

    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char *msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << ","
         << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << ","
         << wl.block_dim(2) << ")"
         << " dyn_smem_bytes=" << wl.dyn_shmem_size;
      std::string cuda = m_->InspectSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      LOG(FATAL) << os.str();
    }

    if (result == CUDA_SUCCESS) {
      cudaError_t last_err = cudaPeekAtLastError();
      if (last_err != cudaSuccess) {
        const char *err_name = nullptr;
        cuGetErrorName(static_cast<CUresult>(last_err), &err_name);
        const char *err_str = cudaGetErrorString(last_err);
        cudaGetLastError();
        LOG(FATAL) << func_name_ << ": "
                   << (err_name ? err_name : "unknown") << " - " << err_str;
      }
    }
  }

private:
  TileScaleCUDAModuleNode *m_;
  ffi::ObjectPtr<ffi::Object> sptr_;
  std::string func_name_;
  mutable std::array<CUfunction, kTileScaleMaxNumGPUs> fcache_;
  LaunchParamConfig launch_param_config_;
  mutable std::array<size_t, kTileScaleMaxNumGPUs> dyn_smem_last_;
  mutable std::array<bool, kTileScaleMaxNumGPUs> dyn_smem_initialized_;
  mutable std::array<bool, kTileScaleMaxNumGPUs> cluster_attr_initialized_;
  bool use_dyn_shared_memory_{false};
};

ffi::Optional<ffi::Function>
TileScaleCUDAModuleNode::GetFunction(const ffi::String &name) {
  ffi::ObjectPtr<ffi::Object> sptr_to_self =
      ffi::GetObjectPtr<ffi::Object>(this);
  ICHECK_EQ(sptr_to_self.get(), this);

  // TileScale: Handle distributed table initialization
  if (name == "__tilescale_init_table") {
    return ffi::Function(TileScaleInitDistributedTable(this, sptr_to_self));
  }

  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value())
    return ffi::Function();
  FunctionInfo info = opt_info.value();
  TileScaleCUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(),
         info->launch_param_tags);
  return PackFuncVoidAddr(f, info->arg_types, info->arg_extra_tags);
}

static ffi::Module TileScaleCUDAModuleCreateImpl(
    ffi::Bytes data, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
    ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<TileScaleCUDAModuleNode>(
      std::move(data), std::move(fmt), std::move(fmap), std::move(source));
  return ffi::Module(n);
}

/*!
 * \brief Create a TileScale extended CUDA module from data.
 *
 * \param data The module data, can be ptx, cubin
 * \param fmt The format of the data, can be "ptx", "cubin"
 * \param fmap The map function information map of each function.
 * \param cuda_source Optional, cuda source file
 */
ffi::Module
TileScaleCUDAModuleCreate(std::string data, std::string fmt,
                          ffi::Map<ffi::String, FunctionInfo> fmap,
                          std::string cuda_source) {
  ffi::Map<ffi::String, ffi::String> source;
  if (!cuda_source.empty()) {
    source.Set("cuda", ffi::String(std::move(cuda_source)));
  }
  return TileScaleCUDAModuleCreateImpl(ffi::Bytes(std::move(data)),
                                       ffi::String(std::move(fmt)),
                                       std::move(fmap), std::move(source));
}

// Load TileScale CUDA module from serialized bytes (deserialization).
ffi::Module TileScaleCUDAModuleLoadFromBytes(const ffi::Bytes &bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes data;
  stream.Read(&fmt);
  ICHECK(stream.Read(&fmap));
  stream.Read(&data);
  return TileScaleCUDAModuleCreateImpl(
      std::move(data), std::move(fmt), std::move(fmap),
      ffi::Map<ffi::String, ffi::String>());
}

// Load TileScale CUDA module from file.
ffi::Module TileScaleCUDAModuleLoadFile(const std::string &file_name,
                                        const ffi::String &format) {
  std::string data;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return TileScaleCUDAModuleCreateImpl(
      ffi::Bytes(std::move(data)), ffi::String(std::move(fmt)),
      std::move(fmap), ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.tilescale_cuda",
           TileScaleCUDAModuleLoadFromBytes)
      .def("ffi.Module.load_from_file.tilescale_cuda",
           TileScaleCUDAModuleLoadFile);
}

} // namespace runtime
} // namespace tvm
