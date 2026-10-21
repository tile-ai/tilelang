/*!
 * \file ascend_module.cc
 * \brief Runnable tvm_ffi runtime module for executable Ascend device ELFs.
 *
 * The module mirrors CUDA's binary runtime module: codegen stores executable
 * device bytes and launch metadata, while the runtime loads functions and
 * launches them on the stream supplied by TVM-FFI's DLPack Exchange API.
 * CANN symbols are resolved lazily by the ascendcl stub library
 * (src/ascend/stubs/) so TileLang keeps no CANN build-time dependency.
 * The Ascend backend is Linux-only.
 */
#include "ascend/stubs/ascendcl.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/thread_storage_scope.h"
#include "support/bytes_io.h"
#include "support/check.h"

#if TILELANG_ASCEND_PROFILER
#include "ascend_profiling.h"
#endif

namespace tvm {
namespace ascend {

using namespace tvm::runtime;

namespace {

using AclError = int32_t;
using AclBinHandle = void *;
using AclFuncHandle = void *;
using AclStream = void *;

constexpr AclError kAclSuccess = 0;
constexpr int32_t kAclLaunchKernelAttrDynUbufSize = 2;
constexpr size_t kAclArgMinAlignment = 4;
constexpr size_t kAclArgBufferAlignment = 8;

size_t AlignUp(size_t value, size_t alignment) {
  TVM_FFI_ICHECK_NE(alignment, 0U);
  TVM_FFI_ICHECK_EQ(alignment & (alignment - 1), 0U);
  return (value + alignment - 1) & ~(alignment - 1);
}

enum class AclArgKind {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat32,
  kFloat64,
  kHandle,
};

struct AclArgLayout {
  AclArgKind kind;
  size_t offset;
};

struct AclArgPackPlan {
  std::vector<AclArgLayout> args;
  size_t buffer_size{0};
};

AclArgKind GetAclArgKind(DLDataType dtype, size_t index,
                         const std::string &function_name) {
  if (dtype.lanes != 1) {
    TVM_FFI_THROW(RuntimeError)
        << "Ascend kernel `" << function_name << "` argument " << index
        << " has " << dtype.lanes
        << " lanes; aclrtLaunchKernelWithHostArgs only supports scalar "
           "kernel arguments";
  }
  if (dtype.code == kDLOpaqueHandle) {
    return AclArgKind::kHandle;
  }
  if (dtype.code == kDLInt) {
    switch (dtype.bits) {
    case 8:
      return AclArgKind::kInt8;
    case 16:
      return AclArgKind::kInt16;
    case 32:
      return AclArgKind::kInt32;
    case 64:
      return AclArgKind::kInt64;
    default:
      break;
    }
  } else if (dtype.code == kDLUInt) {
    switch (dtype.bits) {
    case 8:
      return AclArgKind::kUInt8;
    case 16:
      return AclArgKind::kUInt16;
    case 32:
      return AclArgKind::kUInt32;
    case 64:
      return AclArgKind::kUInt64;
    default:
      break;
    }
  } else if (dtype.code == kDLFloat) {
    if (dtype.bits == 32) {
      return AclArgKind::kFloat32;
    }
    if (dtype.bits == 64) {
      return AclArgKind::kFloat64;
    }
  }
  TVM_FFI_THROW(RuntimeError)
      << "Ascend kernel `" << function_name << "` argument " << index
      << " has unsupported type code=" << static_cast<int>(dtype.code)
      << ", bits=" << static_cast<int>(dtype.bits) << ", lanes=" << dtype.lanes;
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgSize(AclArgKind kind) {
  switch (kind) {
  case AclArgKind::kInt8:
    return sizeof(int8_t);
  case AclArgKind::kInt16:
    return sizeof(int16_t);
  case AclArgKind::kInt32:
    return sizeof(int32_t);
  case AclArgKind::kInt64:
    return sizeof(int64_t);
  case AclArgKind::kUInt8:
    return sizeof(uint8_t);
  case AclArgKind::kUInt16:
    return sizeof(uint16_t);
  case AclArgKind::kUInt32:
    return sizeof(uint32_t);
  case AclArgKind::kUInt64:
    return sizeof(uint64_t);
  case AclArgKind::kFloat32:
    return sizeof(float);
  case AclArgKind::kFloat64:
    return sizeof(double);
  case AclArgKind::kHandle:
    return sizeof(void *);
  }
  TVM_FFI_UNREACHABLE();
}

size_t GetAclArgAlignment(AclArgKind kind) {
  size_t alignment = 0;
  switch (kind) {
  case AclArgKind::kInt8:
    alignment = alignof(int8_t);
    break;
  case AclArgKind::kInt16:
    alignment = alignof(int16_t);
    break;
  case AclArgKind::kInt32:
    alignment = alignof(int32_t);
    break;
  case AclArgKind::kInt64:
    alignment = alignof(int64_t);
    break;
  case AclArgKind::kUInt8:
    alignment = alignof(uint8_t);
    break;
  case AclArgKind::kUInt16:
    alignment = alignof(uint16_t);
    break;
  case AclArgKind::kUInt32:
    alignment = alignof(uint32_t);
    break;
  case AclArgKind::kUInt64:
    alignment = alignof(uint64_t);
    break;
  case AclArgKind::kFloat32:
    alignment = alignof(float);
    break;
  case AclArgKind::kFloat64:
    alignment = alignof(double);
    break;
  case AclArgKind::kHandle:
    alignment = alignof(void *);
    break;
  }
  return std::max(alignment, kAclArgMinAlignment);
}

AclArgPackPlan MakeAclArgPackPlan(const ffi::Array<DLDataType> &arg_types,
                                  const std::string &function_name) {
  AclArgPackPlan plan;
  plan.args.reserve(arg_types.size());
  size_t offset = 0;
  for (size_t i = 0; i < arg_types.size(); ++i) {
    AclArgKind kind = GetAclArgKind(arg_types[i], i, function_name);
    offset = AlignUp(offset, GetAclArgAlignment(kind));
    plan.args.push_back({kind, offset});
    offset += GetAclArgSize(kind);
  }
  plan.buffer_size = AlignUp(offset, kAclArgBufferAlignment);
  return plan;
}

template <typename T, typename U>
void StoreAclArg(uint8_t *destination, U value) {
  T converted = static_cast<T>(value);
  std::memcpy(destination, &converted, sizeof(converted));
}

void PackAclArg(const TVMFFIAny &source, AclArgKind kind,
                uint8_t *destination) {
  switch (kind) {
  case AclArgKind::kInt8:
    return StoreAclArg<int8_t>(destination, source.v_int64);
  case AclArgKind::kInt16:
    return StoreAclArg<int16_t>(destination, source.v_int64);
  case AclArgKind::kInt32:
    return StoreAclArg<int32_t>(destination, source.v_int64);
  case AclArgKind::kInt64:
    return StoreAclArg<int64_t>(destination, source.v_int64);
  case AclArgKind::kUInt8:
    return StoreAclArg<uint8_t>(destination, source.v_int64);
  case AclArgKind::kUInt16:
    return StoreAclArg<uint16_t>(destination, source.v_int64);
  case AclArgKind::kUInt32:
    return StoreAclArg<uint32_t>(destination, source.v_int64);
  case AclArgKind::kUInt64:
    return StoreAclArg<uint64_t>(destination, source.v_int64);
  case AclArgKind::kFloat32:
    return StoreAclArg<float>(destination, source.v_float64);
  case AclArgKind::kFloat64:
    return StoreAclArg<double>(destination, source.v_float64);
  case AclArgKind::kHandle:
    std::memcpy(destination, &source.v_ptr, sizeof(source.v_ptr));
    return;
  }
  TVM_FFI_UNREACHABLE();
}

template <size_t kNumWords> class AclArgBuffer {
public:
  explicit AclArgBuffer(size_t num_words) {
    TVM_FFI_ICHECK_LE(num_words, kNumWords);
  }

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  alignas(kAclArgBufferAlignment) std::array<uint64_t, kNumWords> storage_{};
};

template <> class AclArgBuffer<0> {
public:
  explicit AclArgBuffer(size_t num_words) : storage_(num_words, 0) {}

  uint8_t *data() { return reinterpret_cast<uint8_t *>(storage_.data()); }

private:
  std::vector<uint64_t> storage_;
};

template <size_t kNumWords, typename F>
ffi::Function PackFuncAclHostArgs_(F function, AclArgPackPlan plan) {
  size_t num_words = plan.buffer_size / sizeof(uint64_t);
  auto wrapped = [function = std::move(function), plan = std::move(plan),
                  num_words](ffi::PackedArgs args, ffi::Any *return_value) {
    AclArgBuffer<kNumWords> buffer(num_words);
    const TVMFFIAny *raw_args =
        reinterpret_cast<const TVMFFIAny *>(args.data());
    for (size_t i = 0; i < plan.args.size(); ++i) {
      const AclArgLayout &layout = plan.args[i];
      PackAclArg(raw_args[i], layout.kind, buffer.data() + layout.offset);
    }
    function(args, return_value, buffer.data(), plan.buffer_size);
  };
  return ffi::Function(wrapped);
}

template <typename F>
ffi::Function PackFuncAclHostArgs(F function,
                                  const ffi::Array<DLDataType> &arg_types,
                                  const std::string &function_name) {
  AclArgPackPlan plan = MakeAclArgPackPlan(arg_types, function_name);
  size_t num_words = plan.buffer_size / sizeof(uint64_t);
  if (num_words <= 4) {
    return PackFuncAclHostArgs_<4>(std::move(function), std::move(plan));
  }
  if (num_words <= 8) {
    return PackFuncAclHostArgs_<8>(std::move(function), std::move(plan));
  }
  if (num_words <= 16) {
    return PackFuncAclHostArgs_<16>(std::move(function), std::move(plan));
  }
  return PackFuncAclHostArgs_<0>(std::move(function), std::move(plan));
}

union AclLaunchKernelAttrValue {
  uint8_t schem_mode;
  uint32_t dyn_ubuf_size;
  uint32_t engine_type;
  uint32_t block_dim_offset;
  uint8_t is_block_task_prefetch;
  uint8_t is_data_dump;
  uint16_t timeout;
  uint32_t reserved[4];
};

struct AclLaunchKernelAttr {
  int32_t id;
  AclLaunchKernelAttrValue value;
};

struct AclLaunchKernelCfg {
  AclLaunchKernelAttr *attrs;
  size_t num_attrs;
};

void CheckAcl(AclError result, const char *operation) {
  if (result == kAclSuccess) {
    return;
  }
  const char *message = aclGetRecentErrMsg();
  TVM_FFI_THROW(RuntimeError)
      << operation << " failed with ACL error " << result
      << (message == nullptr ? "" : std::string(": ") + message);
}

#if TILELANG_ASCEND_PROFILER
// Values mirrored from the public aclrtFuncAttribute / aclrtKernelType enums
// in CANN's acl/acl_rt.h. They are spelled out (rather than included) so the
// build still works on CANN versions whose headers predate KERNEL_RATIO; the
// stub resolves the symbols at runtime and profiling degrades gracefully.
enum class AclKernelType : int64_t {
  kAiCore = 0,
  kCube = 1,
  kVector = 2,
  kMix = 3,
  kAiCpu = 100,
};
constexpr int32_t kAclFuncAttrKernelType = 1;
constexpr int32_t kAclFuncAttrKernelRatio = 2;

bool TryGetLogicDeviceId(int32_t user_device_id, int32_t *logic_device_id) {
  // The user-to-logic mapping is stable for the process lifetime, so cache it
  // and avoid an ioctl plus a redundant ACL API report on every launch.
  static std::mutex cache_mutex;
  static std::unordered_map<int32_t, int32_t> cache;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(user_device_id);
    if (it != cache.end()) {
      *logic_device_id = it->second;
      return true;
    }
  }
  const AclError result =
      aclrtGetLogicDevIdByUserDevId(user_device_id, logic_device_id);
  if (result != kAclSuccess) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      LOG(WARNING) << "Ascend profiling could not map user device "
                   << user_device_id << " to a logic device: " << result
                   << "; skipping native reports";
    }
    return false;
  }
  std::lock_guard<std::mutex> lock(cache_mutex);
  cache[user_device_id] = *logic_device_id;
  return true;
}

AscendProfiler::KernelInfo QueryKernelInfo(AclFuncHandle function,
                                           const std::string &kernel_name) {
  AscendProfiler::KernelInfo info;
  if (function == nullptr) {
    return info;
  }
  int64_t kernel_type = 0;
  if (aclrtGetFunctionAttribute(function, kAclFuncAttrKernelType,
                                &kernel_type) != kAclSuccess) {
    return info;
  }
  switch (static_cast<AclKernelType>(kernel_type)) {
  case AclKernelType::kAiCore:
  case AclKernelType::kCube:
    info.task_type = MSPROF_GE_TASK_TYPE_AI_CORE;
    break;
  case AclKernelType::kVector:
    info.task_type = MSPROF_GE_TASK_TYPE_AIV;
    break;
  case AclKernelType::kAiCpu:
    info.task_type = MSPROF_GE_TASK_TYPE_AI_CPU;
    break;
  case AclKernelType::kMix: {
    info.is_mix = true;
    int64_t ratio = 0;
    const AclError result =
        aclrtGetFunctionAttribute(function, kAclFuncAttrKernelRatio, &ratio);
    if (result == kAclSuccess) {
      // ACL encodes AIC:AIV as two uint16_t values, AIC in the high half.
      const uint32_t aic_ratio =
          (static_cast<uint64_t>(ratio) >> 16) & UINT16_MAX;
      const uint32_t aiv_ratio = static_cast<uint64_t>(ratio) & UINT16_MAX;
      if (aic_ratio == 1 && (aiv_ratio == 0 || aiv_ratio == 2)) {
        info.task_type = MSPROF_GE_TASK_TYPE_MIX_AIC;
        info.mix_ratio = aiv_ratio;
      } else if (aiv_ratio == 1 && (aic_ratio == 0 || aic_ratio == 2)) {
        info.task_type = MSPROF_GE_TASK_TYPE_MIX_AIV;
        info.mix_ratio = aic_ratio;
      }
    }
    if (info.task_type == MSPROF_GE_TASK_TYPE_INVALID) {
      LOG(WARNING)
          << "Ascend profiling could not determine the MIX task type and "
             "core ratio for "
          << kernel_name
          << ": aclrtGetFunctionAttribute(ACL_FUNC_ATTR_KERNEL_RATIO) "
             "returned "
          << result << ", ratio=" << ratio
          << "; check runtime support for this attribute and the kernel core "
             "ratio. Reporting unspecified taskType and blockDim in basic "
             "metadata";
    }
    break;
  }
  default:
    break;
  }
  return info;
}
#endif

} // namespace

class AscendModuleNode : public ffi::ModuleObj {
public:
  AscendModuleNode(ffi::Bytes code, ffi::String fmt,
                   ffi::Map<ffi::String, FunctionInfo> fmap,
                   ffi::Map<ffi::String, ffi::String> source)
      : code_(std::move(code)), fmt_(std::move(fmt)), fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  ~AscendModuleNode() {
    if (device_modules_.empty()) {
      return;
    }
    for (const auto &[device_id, device_module] : device_modules_) {
      if (device_module.binary != nullptr) {
        // binary != nullptr implies a prior successful stub call, so the
        // lazy-loaded library is guaranteed to be present here.
        AclError result = aclrtBinaryUnLoad(device_module.binary);
        if (result != kAclSuccess) {
          LOG(WARNING) << "aclrtBinaryUnLoad failed for Ascend device "
                       << device_id << " with error " << result;
        }
      }
    }
  }

  const char *kind() const final { return "asc"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final;

  ffi::Bytes SaveToBytes() const final {
    // Keep the same payload shape as CUDA: [fmt][fmap][device code].
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    if (auto source = source_.Get(format)) {
      return source.value();
    }
    if (format.empty()) {
      if (auto source = source_.Get("asc")) {
        return source.value();
      }
    }
    return ffi::String();
  }

  AclFuncHandle GetFunctionHandle(int32_t device_id, const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    DeviceModule &device_module = device_modules_[device_id];
    if (device_module.binary == nullptr) {
      CheckAcl(aclrtBinaryLoadFromData(code_.data(), code_.size(), nullptr,
                                       &device_module.binary),
               "aclrtBinaryLoadFromData");
    }

    auto cached = device_module.functions.find(name);
    if (cached != device_module.functions.end()) {
      return cached->second;
    }

    AclFuncHandle function{nullptr};
    CheckAcl(
        aclrtBinaryGetFunction(device_module.binary, name.c_str(), &function),
        "aclrtBinaryGetFunction");
    device_module.functions.emplace(name, function);
    return function;
  }

private:
  struct DeviceModule {
    AclBinHandle binary{nullptr};
    std::unordered_map<std::string, AclFuncHandle> functions;
  };

  ffi::Bytes code_;
  ffi::String fmt_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  ffi::Map<ffi::String, ffi::String> source_;

  std::mutex mutex_;
  std::unordered_map<int32_t, DeviceModule> device_modules_;
};

struct DirectLaunchPolicy {
  template <typename Launch>
  AclError operator()(Launch &&launch, const std::string &, AclFuncHandle,
                      int32_t, uint32_t) const {
    return launch();
  }
};

#if TILELANG_ASCEND_PROFILER
class ProfiledLaunchPolicy {
public:
  explicit ProfiledLaunchPolicy(AscendProfiler *profiler)
      : profiler_(profiler),
        metadata_cache_(std::make_shared<KernelMetadataCache>()) {}

  template <typename Launch>
  AclError operator()(Launch &&launch, const std::string &kernel_name,
                      AclFuncHandle function, int32_t device_id,
                      uint32_t num_blocks) const {
    if (!profiler_->IsCollecting()) {
      return launch();
    }
    // ACL launches use user IDs, while Msprof callbacks report logic IDs.
    int32_t logic_device_id = 0;
    if (!TryGetLogicDeviceId(device_id, &logic_device_id)) {
      return launch();
    }
    const uint64_t flags = profiler_->GetCollectionFlags(logic_device_id);
    if (flags == 0) {
      return launch();
    }
    const auto kernel = GetOrQueryKernelInfo(function, kernel_name);
    const uint64_t begin = profiler_->GetCycleTime();
    const AclError result = launch();
    const uint64_t end = profiler_->GetCycleTime();
    if (result == kAclSuccess) {
      profiler_->ReportLaunch(kernel_name, begin, end, flags, kernel,
                              num_blocks);
    }
    return result;
  }

private:
  struct KernelMetadataCache {
    std::mutex mutex;
    std::unordered_map<AclFuncHandle, AscendProfiler::KernelInfo> kernels;
  };

  AscendProfiler::KernelInfo
  GetOrQueryKernelInfo(AclFuncHandle function,
                       const std::string &kernel_name) const {
    std::lock_guard<std::mutex> lock(metadata_cache_->mutex);
    auto it = metadata_cache_->kernels.find(function);
    if (it != metadata_cache_->kernels.end()) {
      return it->second;
    }
    auto info = QueryKernelInfo(function, kernel_name);
    if (info.task_type == MSPROF_GE_TASK_TYPE_INVALID && !info.is_mix) {
      LOG(WARNING) << "Ascend profiling could not determine the task type and "
                      "core ratio for "
                   << kernel_name << "; reporting unspecified basic metadata";
    }
    metadata_cache_->kernels.emplace(function, info);
    return info;
  }

  AscendProfiler *profiler_;
  std::shared_ptr<KernelMetadataCache> metadata_cache_;
};
#endif

template <typename LaunchPolicy> class AscendWrappedFunc {
public:
  explicit AscendWrappedFunc(LaunchPolicy launch_policy)
      : launch_policy_(std::move(launch_policy)) {}

  void Init(AscendModuleNode *module, ffi::ObjectPtr<ffi::Object> module_ref,
            std::string function_name, size_t num_kernel_args,
            const ffi::Array<ffi::String> &launch_param_tags) {
    module_ = module;
    module_ref_ = std::move(module_ref);
    function_name_ = std::move(function_name);
    launch_param_config_.Init(num_kernel_args, launch_param_tags);
  }

  void operator()(ffi::PackedArgs args, ffi::Any *return_value,
                  void *packed_args, size_t packed_args_size) const {
    ThreadWorkLoad workload = launch_param_config_.Extract(args);
    size_t num_blocks = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
      size_t extent = workload.grid_dim(axis);
      TVM_FFI_CHECK(extent > 0, RuntimeError)
          << "Ascend launch grid must be positive for kernel "
          << function_name_;
      TVM_FFI_CHECK(extent <= std::numeric_limits<uint32_t>::max() / num_blocks,
                    RuntimeError)
          << "Ascend launch grid exceeds uint32 range for kernel "
          << function_name_;
      num_blocks *= extent;
    }
    TVM_FFI_CHECK(workload.dyn_shmem_size <=
                      std::numeric_limits<uint32_t>::max(),
                  RuntimeError)
        << "Ascend dynamic UBUF size exceeds uint32 range for kernel "
        << function_name_;

    int32_t device_id = 0;
    CheckAcl(aclrtGetDevice(&device_id), "aclrtGetDevice");
    AclFuncHandle function =
        module_->GetFunctionHandle(device_id, function_name_);
    AclStream stream = TVMFFIEnvGetStream(kDLExtDev, device_id);

    AclLaunchKernelAttr attribute{};
    AclLaunchKernelCfg config{};
    AclLaunchKernelCfg *config_ptr = nullptr;
    if (workload.dyn_shmem_size != 0) {
      attribute.id = kAclLaunchKernelAttrDynUbufSize;
      attribute.value.dyn_ubuf_size =
          static_cast<uint32_t>(workload.dyn_shmem_size);
      config.attrs = &attribute;
      config.num_attrs = 1;
      config_ptr = &config;
    }

    auto launch = [&]() {
      return aclrtLaunchKernelWithHostArgs(
          function, static_cast<uint32_t>(num_blocks), stream, config_ptr,
          packed_args, packed_args_size, nullptr, 0);
    };
    const AclError result =
        launch_policy_(launch, function_name_, function, device_id,
                       static_cast<uint32_t>(num_blocks));
    if (result != kAclSuccess) {
      const char *message = aclGetRecentErrMsg();
      std::ostringstream error;
      error << "aclrtLaunchKernelWithHostArgs failed for " << function_name_
            << " with ACL error " << result << ", grid=" << num_blocks
            << ", dyn_ubuf_bytes=" << workload.dyn_shmem_size;
      if (message != nullptr) {
        error << ": " << message;
      }
      ffi::String source = module_->InspectSource("asc");
      if (!source.empty()) {
        error << "\n// Ascend Source\n" << source;
      }
      TVM_FFI_THROW(RuntimeError) << error.str();
    }
  }

private:
  AscendModuleNode *module_{nullptr};
  ffi::ObjectPtr<ffi::Object> module_ref_;
  std::string function_name_;
  LaunchParamConfig launch_param_config_;
  LaunchPolicy launch_policy_;
};

ffi::Optional<ffi::Function>
AscendModuleNode::GetFunction(const ffi::String &name) {
  auto function_info = fmap_.Get(name);
  if (!function_info.has_value()) {
    return ffi::Function();
  }
  ffi::ObjectPtr<ffi::Object> module_ref = ffi::GetObjectPtr<ffi::Object>(this);
  FunctionInfo info = function_info.value();
  TVM_FFI_CHECK(info->arg_extra_tags.empty(), RuntimeError)
      << "Ascend runtime does not support extra kernel argument tags";
#if TILELANG_ASCEND_PROFILER
  if (auto *profiler = AscendProfiler::GetIfEnabled()) {
    AscendWrappedFunc<ProfiledLaunchPolicy> function{
        ProfiledLaunchPolicy(profiler)};
    function.Init(this, std::move(module_ref), name, info->arg_types.size(),
                  info->launch_param_tags);
    return PackFuncAclHostArgs(std::move(function), info->arg_types, name);
  }
#endif
  AscendWrappedFunc<DirectLaunchPolicy> function{DirectLaunchPolicy()};
  function.Init(this, std::move(module_ref), name, info->arg_types.size(),
                info->launch_param_tags);
  return PackFuncAclHostArgs(std::move(function), info->arg_types, name);
}

ffi::Module AscendModuleCreate(ffi::Bytes code, ffi::String fmt,
                               ffi::Map<ffi::String, FunctionInfo> fmap,
                               ffi::Map<ffi::String, ffi::String> source) {
  auto module = ffi::make_object<AscendModuleNode>(
      std::move(code), std::move(fmt), std::move(fmap), std::move(source));
  return ffi::Module(module);
}

static ffi::Module AscendModuleLoadFromBytes(const ffi::Bytes &bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  TVM_FFI_CHECK(fmt == "aibin", RuntimeError)
      << "Unsupported Ascend module format `" << fmt
      << "`. This is likely a legacy source-only cache entry; remove it and "
         "recompile.";
  return AscendModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                            ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
#if TILELANG_ASCEND_PROFILER
  // Register when the library loads, before a later profiler.start(). CANN
  // replays state asynchronously if registration is deferred until launch.
  AscendProfiler::GetIfEnabled();
#endif
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.asc", AscendModuleLoadFromBytes)
      .def("tl.ascend.ModuleCreate", AscendModuleCreate);
}

} // namespace ascend
} // namespace tvm
