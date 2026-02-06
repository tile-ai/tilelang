/*!
 * \file tilescale_cuda_module.cc
 * \brief TileScale extended CUDA module with distributed table initialization
 * support.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/cuda/cuda_common.h"
#include "runtime/file_utils.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"
#include "runtime/thread_storage_scope.h"

namespace tvm {
namespace runtime {

// Maximum number of GPUs supported (same as TVM's default)
constexpr int kTileScaleMaxNumGPUs = 32;

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
      std::string data, std::string fmt,
      std::unordered_map<std::string, FunctionInfo> fmap,
      std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }

  ~TileScaleCUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
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
      ICHECK_NE(cuda_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cuda_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream *stream = &ms;
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
    return ffi::Bytes(buffer);
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    if (format == fmt_)
      return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx")
        return data_;
      return "";
    }
  }

  // Get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string &func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
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
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
      static auto nvshmem_init_hook =
          ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void *>(module_[device_id]));
      }
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = cuModuleGetGlobal(&global, &nbytes, module_[device_id],
                                        global_name.c_str());
    ICHECK_EQ(nbytes, expect_nbytes);
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetGlobal " << global_name
                 << " failed with error: " << msg;
    }
    return global;
  }

private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string cuda_source_;
  std::array<CUmodule, kTileScaleMaxNumGPUs> module_;
  std::mutex mutex_;
};

// Implementation of TileScaleInitDistributedTable::operator()
void TileScaleInitDistributedTable::operator()(const ffi::PackedArgs &args,
                                               ffi::Any *rv) const {
  // Accept int64_t from Python and cast to pointers internally
  // This is necessary because TVM FFI doesn't auto-convert int to void*
  int64_t host_table_ptr = args[0].cast<int64_t>();
  int64_t table_size = args[1].cast<int64_t>();
  int64_t stream_ptr = args[2].cast<int64_t>();

  void *host_table = reinterpret_cast<void *>(host_table_ptr);
  // 打印host table前8个entry
  auto *table_ptr = reinterpret_cast<const uint64_t *>(host_table);
  std::ostringstream oss;
  int rank;
  CUDA_CALL(cudaGetDevice(&rank));
  CUstream stream = reinterpret_cast<CUstream>(stream_ptr);

  int device_id;
  CUDA_CALL(cudaGetDevice(&device_id));

  // Get the device pointer for meta_data symbol (lazy initialization)
  if (pcache_[device_id] == 0) {
    pcache_[device_id] = m_->GetGlobal(device_id, "meta_data", kMetaDataSize);
  }

  // Copy data from host to device constant memory.
  // Note: must use Driver API (cuMemcpyHtoD) instead of cudaMemcpyToSymbol,
  // because the symbol lives in a dynamically loaded CUmodule.
  size_t bytes = static_cast<size_t>(table_size) * sizeof(uint64_t);
  CUDA_DRIVER_CALL(cuMemcpyHtoD(pcache_[device_id], host_table, bytes));

  // Return success
  *rv = 0;
}

// Wrapped function class similar to TVM's CUDAWrappedFunc
class TileScaleCUDAWrappedFunc {
public:
  void Init(TileScaleCUDAModuleNode *m, ffi::ObjectPtr<ffi::Object> sptr,
            const std::string &func_name, size_t num_void_args,
            const std::vector<std::string> &launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    std::fill(dyn_smem_initialized_.begin(), dyn_smem_initialized_.end(),
              false);
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
    // Get stream from TVM's device API
    CUstream strm = nullptr;
    static auto get_stream =
        ffi::Function::GetGlobal("device_api.cuda.get_stream");
    if (get_stream.has_value()) {
      strm = static_cast<CUstream>((*get_stream)(device_id).cast<void *>());
    }
    CUresult result = cuLaunchKernel(
        fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2),
        wl.block_dim(0), wl.block_dim(1), wl.block_dim(2), wl.dyn_shmem_size,
        strm, void_args, nullptr);
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << ","
         << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << ","
         << wl.block_dim(2) << ")\n";
      std::string cuda_err = os.str();
      LOG(FATAL) << "CUDALaunch Error: " << cuda_err;
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
  bool use_dyn_shared_memory_{false};
};

// Prepare global barrier class
class TileScaleCUDAPrepGlobalBarrier {
public:
  TileScaleCUDAPrepGlobalBarrier(TileScaleCUDAModuleNode *m,
                                 ffi::ObjectPtr<ffi::Object> sptr)
      : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const ffi::PackedArgs &args, ffi::Any *rv) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] = m_->GetGlobal(
          device_id, symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    CUDA_DRIVER_CALL(cuMemsetD32(pcache_[device_id], 0, 1));
  }

private:
  TileScaleCUDAModuleNode *m_;
  ffi::ObjectPtr<ffi::Object> sptr_;
  mutable std::array<CUdeviceptr, kTileScaleMaxNumGPUs> pcache_;
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

  // TVM: Handle global barrier preparation
  if (name == symbol::tvm_prepare_global_barrier) {
    return ffi::Function(TileScaleCUDAPrepGlobalBarrier(this, sptr_to_self));
  }

  auto it = fmap_.find(name);
  if (it == fmap_.end())
    return ffi::Function();
  const FunctionInfo &info = it->second;
  TileScaleCUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(),
         info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types, info.arg_extra_tags);
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
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source) {
  auto n =
      ffi::make_object<TileScaleCUDAModuleNode>(data, fmt, fmap, cuda_source);
  return ffi::Module(n);
}

// Load TileScale CUDA module from serialized bytes (deserialization).
ffi::Module TileScaleCUDAModuleLoadFromBytes(const ffi::Bytes &bytes) {
  dmlc::MemoryFixedSizeStream ms(const_cast<char *>(bytes.data()),
                                 bytes.size());
  dmlc::Stream *stream = &ms;
  std::string fmt;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string data;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return TileScaleCUDAModuleCreate(data, fmt, fmap, std::string());
}

// Load TileScale CUDA module from file.
ffi::Module TileScaleCUDAModuleLoadFile(const std::string &file_name,
                                        const ffi::String &format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return TileScaleCUDAModuleCreate(data, fmt, fmap, std::string());
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
