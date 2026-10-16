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
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
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

namespace tvm {
namespace ascend {

using namespace tvm::runtime;

class AscendModuleNode;

namespace {

using AclError = int32_t;
using AclBinHandle = void *;
using AclFuncHandle = void *;
using AclStream = void *;

constexpr AclError kAclSuccess = 0;
constexpr int32_t kAclLaunchKernelAttrSchemMode = 1;
constexpr int32_t kAclLaunchKernelAttrDynUbufSize = 2;
constexpr int32_t kAclLaunchKernelAttrBlockTaskPrefetch = 5;
constexpr int32_t kAclLaunchKernelAttrDataDump = 6;
constexpr int32_t kAclLaunchKernelAttrTimeout = 7;
constexpr int32_t kAclLaunchKernelAttrTimeoutUs = 8;
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
  struct {
    uint32_t timeout_low;
    uint32_t timeout_high;
  } timeout_us;
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

// ===== Parameter validation and LaunchKernel cfg construction =====
//
// Target chip: Ascend 950 (A5). cfg strategy:
//   - DYN_UBUF_SIZE: filled only when dyn_ubuf_size > 0 (SIMT dynamic UB)
//   - SCHEM_MODE / TIMEOUT / TIMEOUT_US / DATA_DUMP / BLOCK_TASK_PREFETCH:
//     filled on demand via env vars
//   - ENGINE_TYPE / BLOCKDIM_OFFSET: not supported by 950, left unset
// Env vars: ASCEND_LAUNCH_SCHEM_MODE / ASCEND_LAUNCH_TIMEOUT /
//           ASCEND_LAUNCH_TIMEOUT_US / ASCEND_LAUNCH_DATA_DUMP /
//           ASCEND_LAUNCH_BLOCK_TASK_PREFETCH
//           (all optional; attr omitted if unset)
// Note: ASCEND_LAUNCH_TIMEOUT and ASCEND_LAUNCH_TIMEOUT_US are mutually
//       exclusive; setting both is a configuration error.

struct LaunchEnvConfig {
  bool has_schem_mode = false;
  uint8_t schem_mode = 0;
  bool has_timeout = false;
  uint16_t timeout = 0;
  bool has_timeout_us = false;
  uint64_t timeout_us = 0;
  bool has_data_dump = false;
  uint8_t is_data_dump = 0;
  bool has_block_task_prefetch = false;
  uint8_t is_block_task_prefetch = 0;
};

const LaunchEnvConfig &GetLaunchEnvConfig() {
  static const LaunchEnvConfig cfg = [] {
    LaunchEnvConfig c;
    if (const char *v = std::getenv("ASCEND_LAUNCH_SCHEM_MODE")) {
      long val = std::strtol(v, nullptr, 10);
      if (val >= 0 && val < 2) {
        c.has_schem_mode = true;
        c.schem_mode = static_cast<uint8_t>(val);
      }
    }
    if (const char *v = std::getenv("ASCEND_LAUNCH_TIMEOUT")) {
      long val = std::strtol(v, nullptr, 10);
      if (val >= 0 && val <= std::numeric_limits<uint16_t>::max()) {
        c.has_timeout = true;
        c.timeout = static_cast<uint16_t>(val);
      }
    }
    if (const char *v = std::getenv("ASCEND_LAUNCH_TIMEOUT_US")) {
      unsigned long long val = std::strtoull(v, nullptr, 10);
      c.has_timeout_us = true;
      c.timeout_us = static_cast<uint64_t>(val);
    }
    if (c.has_timeout && c.has_timeout_us) {
      std::cerr
          << "Warning: ASCEND_LAUNCH_TIMEOUT and ASCEND_LAUNCH_TIMEOUT_US "
          << "are mutually exclusive; ignoring TIMEOUT_US." << std::endl;
      c.has_timeout_us = false;
      c.timeout_us = 0;
    }
    if (const char *v = std::getenv("ASCEND_LAUNCH_DATA_DUMP")) {
      long val = std::strtol(v, nullptr, 10);
      if (val == 0 || val == 1) {
        c.has_data_dump = true;
        c.is_data_dump = static_cast<uint8_t>(val);
      }
    }
    if (const char *v = std::getenv("ASCEND_LAUNCH_BLOCK_TASK_PREFETCH")) {
      long val = std::strtol(v, nullptr, 10);
      if (val == 0 || val == 1) {
        c.has_block_task_prefetch = true;
        c.is_block_task_prefetch = static_cast<uint8_t>(val);
      }
    }
    return c;
  }();
  return cfg;
}

void ValidateLaunchParams(AclFuncHandle function, uint32_t num_blocks,
                          void *args, size_t args_size,
                          const std::string &function_name) {
  TVM_FFI_CHECK(function != nullptr, RuntimeError)
      << "Ascend kernel function handle is null for " << function_name;
  TVM_FFI_CHECK(num_blocks > 0, RuntimeError)
      << "Ascend launch grid must be positive for " << function_name;
  TVM_FFI_CHECK(args_size == 0 || args != nullptr, RuntimeError)
      << "Ascend kernel args buffer is null for " << function_name;
  TVM_FFI_CHECK(args_size <= std::numeric_limits<uint32_t>::max(), RuntimeError)
      << "Ascend kernel args size " << args_size << " exceeds uint32 range for "
      << function_name;
}

// Build launch cfg. `attrs` points to a caller-provided array (capacity >= 6).
// Returns the actual number of attrs populated.
size_t BuildLaunchCfg(AclLaunchKernelAttr *attrs, uint32_t dyn_ubuf_size) {
  const LaunchEnvConfig &env = GetLaunchEnvConfig();
  size_t n = 0;
  if (dyn_ubuf_size != 0) {
    attrs[n].id = kAclLaunchKernelAttrDynUbufSize;
    attrs[n].value.dyn_ubuf_size = dyn_ubuf_size;
    ++n;
  }
  if (env.has_schem_mode) {
    attrs[n].id = kAclLaunchKernelAttrSchemMode;
    attrs[n].value.schem_mode = env.schem_mode;
    ++n;
  }
  if (env.has_timeout) {
    attrs[n].id = kAclLaunchKernelAttrTimeout;
    attrs[n].value.timeout = env.timeout;
    ++n;
  }
  if (env.has_timeout_us) {
    attrs[n].id = kAclLaunchKernelAttrTimeoutUs;
    attrs[n].value.timeout_us.timeout_low =
        static_cast<uint32_t>(env.timeout_us & 0xFFFFFFFFULL);
    attrs[n].value.timeout_us.timeout_high =
        static_cast<uint32_t>(env.timeout_us >> 32);
    ++n;
  }
  if (env.has_data_dump) {
    attrs[n].id = kAclLaunchKernelAttrDataDump;
    attrs[n].value.is_data_dump = env.is_data_dump;
    ++n;
  }
  if (env.has_block_task_prefetch) {
    attrs[n].id = kAclLaunchKernelAttrBlockTaskPrefetch;
    attrs[n].value.is_block_task_prefetch = env.is_block_task_prefetch;
    ++n;
  }
  return n;
}

AclError LaunchAclKernelWithHostArgs(AclFuncHandle function,
                                     uint32_t num_blocks, AclStream stream,
                                     uint32_t dyn_ubuf_size, void *packed_args,
                                     size_t packed_args_size) {
  AclLaunchKernelAttr attrs[6];
  AclLaunchKernelCfg config{};
  config.attrs = attrs;
  config.num_attrs = BuildLaunchCfg(attrs, dyn_ubuf_size);
  AclLaunchKernelCfg *config_ptr = config.num_attrs > 0 ? &config : nullptr;

  return aclrtLaunchKernelWithHostArgs(function, num_blocks, stream, config_ptr,
                                       packed_args, packed_args_size, nullptr,
                                       0);
}

using TaskQueueLaunchFn = int (*)(void *context);
using TaskQueueDestroyFn = void (*)(void *context);
using TaskQueueSubmitFn = int (*)(const char *op_name, TaskQueueLaunchFn launch,
                                  TaskQueueDestroyFn destroy, void *context,
                                  int sync);

std::atomic<TaskQueueSubmitFn> g_task_queue_submit{nullptr};

struct AscendLaunchTask {
  // Keep the binary owning `function` loaded until torch_npu destroys the
  // queued std::function and the adapter calls DestroyTaskQueueContext.
  ffi::ObjectPtr<ffi::Object> module_ref;
  std::string function_name;
  // Raw pointer kept alive by module_ref above.
  AscendModuleNode *module{nullptr};
  AclFuncHandle function{nullptr};
  uint32_t num_blocks{0};
  AclStream stream{nullptr};
  uint32_t dyn_ubuf_size{0};

  std::vector<uint64_t> packed_args;
  size_t packed_args_size{0};
};

void SetTaskQueueSubmitFn(int64_t address) {
  TVM_FFI_CHECK(address != 0, ValueError)
      << "tl.ascend.SetTaskQueueSubmitFn expects a non-null address";
  g_task_queue_submit.store(reinterpret_cast<TaskQueueSubmitFn>(address),
                            std::memory_order_release);
}

int LaunchKernelForTaskQueue(AscendLaunchTask &task) {
  return static_cast<int>(LaunchAclKernelWithHostArgs(
      task.function, task.num_blocks, task.stream, task.dyn_ubuf_size,
      task.packed_args.data(), task.packed_args_size));
}

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

namespace {

// Runs on the torch_npu task queue worker thread.
void LogTaskQueueLaunchFailure(const AscendLaunchTask &task, int status,
                               const char *details) noexcept {
  try {
    std::ostringstream error;
    error << "Ascend task queue launch failed for " << task.function_name
          << " with status " << status << ", grid=" << task.num_blocks
          << ", dyn_ubuf_bytes=" << task.dyn_ubuf_size;
    if (details != nullptr && details[0] != '\0') {
      error << ": " << details;
    }
    if (task.module != nullptr) {
      ffi::String source = task.module->InspectSource("asc");
      if (!source.empty()) {
        error << "\n// Ascend Source\n" << source;
      }
    }
    LOG(ERROR) << error.str();
  } catch (...) {
    // Logging must not let an exception escape the task queue callback.
  }
}

int LaunchTaskQueueCallback(void *context) noexcept {
  auto &task = *static_cast<AscendLaunchTask *>(context);
  try {
    int result = LaunchKernelForTaskQueue(task);
    if (result != kAclSuccess) {
      LogTaskQueueLaunchFailure(task, result, aclGetRecentErrMsg());
    }
    return result;
  } catch (const std::exception &exception) {
    LogTaskQueueLaunchFailure(task, -1, exception.what());
  } catch (...) {
    // Never let a TileLang C++ exception cross the C ABI boundary into an
    // adapter built with the customer's torch toolchain.
    LogTaskQueueLaunchFailure(task, -1, "unknown C++ exception");
  }
  return -1;
}

void DestroyTaskQueueContext(void *context) noexcept {
  delete static_cast<AscendLaunchTask *>(context);
}

} // namespace

class AscendWrappedFunc {
public:
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
    const uint32_t dyn_ubuf_size =
        static_cast<uint32_t>(workload.dyn_shmem_size);
    const uint32_t grid = static_cast<uint32_t>(num_blocks);
    ValidateLaunchParams(function, grid, packed_args, packed_args_size,
                         function_name_);

    TaskQueueSubmitFn submit =
        g_task_queue_submit.load(std::memory_order_acquire);
    if (submit == nullptr) {
      AclError result = LaunchAclKernelWithHostArgs(
          function, grid, stream, dyn_ubuf_size, packed_args, packed_args_size);
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
    } else {
      auto task = std::make_unique<AscendLaunchTask>();
      task->module_ref = module_ref_;
      task->function_name = function_name_;
      task->module = module_;
      task->function = function;
      task->num_blocks = grid;
      task->stream = stream;

      task->dyn_ubuf_size = static_cast<uint32_t>(workload.dyn_shmem_size);
      task->packed_args_size = packed_args_size;
      const size_t num_words =
          (packed_args_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      task->packed_args.resize(std::max<size_t>(num_words, 1), 0);
      if (packed_args_size != 0) {
        std::memcpy(task->packed_args.data(), packed_args, packed_args_size);
      }

      AscendLaunchTask *context = task.release();
      int submit_result =
          submit(function_name_.c_str(), LaunchTaskQueueCallback,
                 DestroyTaskQueueContext, context, 0);
      TVM_FFI_CHECK(submit_result == 0, RuntimeError)
          << "torch_npu task queue submission failed for " << function_name_
          << " (submit result " << submit_result << ")";
    }
  }

private:
  AscendModuleNode *module_{nullptr};
  ffi::ObjectPtr<ffi::Object> module_ref_;
  std::string function_name_;
  LaunchParamConfig launch_param_config_;
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
  AscendWrappedFunc function;
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
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.asc", AscendModuleLoadFromBytes)
      .def("tl.ascend.ModuleCreate", AscendModuleCreate)
      .def("tl.ascend.SetTaskQueueSubmitFn", SetTaskQueueSubmitFn);
}

} // namespace ascend
} // namespace tvm
