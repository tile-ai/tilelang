/*!
 * \file ascend_exception_dump.cc
 * \brief Save tensor inputs when CANN reports an AI Core exception.
 */
#include "ascend_exception_dump.h"

#include "ascend/stubs/ascendcl.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if !TILELANG_ASCEND_USE_STUBS
#include <dlfcn.h>
#endif

namespace tvm {
namespace ascend {

namespace {

using ::tl::ascend::ExceptionDumpInfo;
using ::tl::ascend::kExceptionDumpMagic;
using ::tl::ascend::kExceptionDumpVersion;
using ::tl::ascend::kMaxExceptionDumpTensors;

constexpr int32_t kAclMemcpyDeviceToHost = 2;
constexpr int32_t kAclDumpTensorInput = 0;
constexpr int32_t kAclDumpAddrRaw = 2;
constexpr int32_t kAclDumpPlacementDevice = 0;
constexpr int32_t kAclFormatND = 2;
// Kernel argument buffers are small. Bound allocations in a fault callback in
// case a broken runtime returns corrupt argument metadata.
constexpr size_t kMaxExceptionArgsBytes = 16 * 1024 * 1024;

const tl::ascendcl::AscendCLAPI *GetExceptionDumpAPI() {
#if TILELANG_ASCEND_USE_STUBS
  return tl::ascendcl::AscendCLAPI::get_or_null();
#else
  // Direct-link builds must remain usable with older CANN when dumping is off.
  // Resolve these optional APIs lazily instead of adding new DT_NEEDED symbols.
  // libascendcl is already linked for ordinary launches; its dependency
  // libascend_dump provides acldumpSaveExceptionInfo.
  static const auto api = []() {
    tl::ascendcl::AscendCLAPI result{};
#define LOOKUP_EXCEPTION_API(symbol)                                           \
  result.symbol##_ = reinterpret_cast<decltype(result.symbol##_)>(             \
      dlsym(RTLD_DEFAULT, #symbol))
    LOOKUP_EXCEPTION_API(aclrtExceptionInfoCallbackRegister);
    LOOKUP_EXCEPTION_API(aclrtGetArgsFromExceptionInfo);
    LOOKUP_EXCEPTION_API(aclrtMemcpy);
    LOOKUP_EXCEPTION_API(acldumpSaveExceptionInfo);
#undef LOOKUP_EXCEPTION_API
    return result;
  }();
  return &api;
#endif
}

void ExceptionDumpCallback(void *exception_info) noexcept {
  if (exception_info == nullptr) {
    return;
  }
  try {
    void *device_args = nullptr;
    uint32_t args_size = 0;
    const auto *api = GetExceptionDumpAPI();
    const int32_t result = api->aclrtGetArgsFromExceptionInfo_(
        exception_info, &device_args, &args_size);
    if (result != 0 || device_args == nullptr ||
        args_size < sizeof(ExceptionDumpInfo) ||
        args_size > kMaxExceptionArgsBytes) {
      return;
    }
    std::vector<uint8_t> host_args(args_size);
    // CANN supplies device memory, not a host argument-array pointer. Copy it
    // first; the decoder only reads this bounded host-owned snapshot.
    if (api->aclrtMemcpy_(host_args.data(), host_args.size(), device_args,
                          args_size, kAclMemcpyDeviceToHost) != 0) {
      std::fprintf(stderr,
                   "TileLang: failed to copy AI Core exception arguments\n");
      return;
    }
    const int32_t dump_result =
        DumpExceptionHostArgs(host_args.data(), host_args.size());
    if (dump_result > 0) {
      std::fprintf(stderr,
                   "TileLang: acldumpSaveExceptionInfo failed with error %d\n",
                   dump_result);
    }
  } catch (...) {
    // CANN invokes this on its own reporting thread. In particular, an
    // allocation failure must not replace the original device exception.
    std::fprintf(stderr, "TileLang: AI Core exception dump callback failed\n");
  }
}

} // namespace

bool FindExceptionDumpInfo(const void *args, size_t size,
                           ExceptionDumpInfo *info) noexcept {
  if (args == nullptr || info == nullptr || size < sizeof(ExceptionDumpInfo)) {
    return false;
  }
  const auto *bytes = static_cast<const uint8_t *>(args);
  // Registration is process-wide, so other kernels may reach this callback.
  // Probe aligned ABI positions with memcpy: the host buffer itself need not
  // be aligned, and foreign/truncated records must be ignored safely.
  for (size_t offset = 0; offset <= size - sizeof(ExceptionDumpInfo);
       offset += alignof(ExceptionDumpInfo)) {
    uint64_t magic;
    std::memcpy(&magic, bytes + offset, sizeof(magic));
    if (magic != kExceptionDumpMagic) {
      continue;
    }
    ExceptionDumpInfo candidate;
    std::memcpy(&candidate, bytes + offset, sizeof(candidate));
    if (candidate.version != kExceptionDumpVersion || candidate.count == 0 ||
        candidate.count > kMaxExceptionDumpTensors ||
        candidate.kernel_name[0] == '\0' ||
        std::memchr(candidate.kernel_name, '\0',
                    sizeof(candidate.kernel_name)) == nullptr) {
      // A user scalar can happen to equal MAGIC. Keep looking for a complete,
      // valid record instead of treating the first match as authoritative.
      continue;
    }
    bool valid_tensors = true;
    for (uint32_t i = 0; i < candidate.count; ++i) {
      const auto &tensor = candidate.tensors[i];
      if (tensor.address == 0 || tensor.size == 0 || tensor.elements == 0) {
        continue;
      }
      const uint32_t bytes =
          ::tl::ascend::ExceptionDumpElementBytes(tensor.dtype);
      if (bytes == 0 || tensor.reserved != 0 ||
          tensor.elements > std::numeric_limits<uint64_t>::max() / bytes ||
          tensor.size != tensor.elements * bytes) {
        valid_tensors = false;
        break;
      }
    }
    if (!valid_tensors) {
      continue;
    }
    *info = candidate;
    return true;
  }
  return false;
}

int32_t DumpExceptionHostArgs(const void *args, size_t size) noexcept {
  try {
    ExceptionDumpInfo info;
    if (!FindExceptionDumpInfo(args, size, &info)) {
      return -1;
    }
    std::array<tl::ascendcl::DumpTensorInfo, kMaxExceptionDumpTensors>
        tensors{};
    size_t count = 0;
    for (uint32_t i = 0; i < info.count; ++i) {
      const auto &source = info.tensors[i];
      if (source.address == 0 || source.size == 0 || source.elements == 0) {
        continue;
      }
      auto &tensor = tensors[count++];
      // The compiler currently has no input/output classification for these
      // buffer parameters. Compact valid buffers as INPUT; dump indices may
      // therefore differ from original parameter/metadata indices.
      tensor.type = kAclDumpTensorInput;
      tensor.tensor_size = source.size;
      tensor.format = kAclFormatND;
      tensor.data_type = source.dtype;
      tensor.tensor_addr = reinterpret_cast<int64_t *>(source.address);
      tensor.addr_type = kAclDumpAddrRaw;
      tensor.placement = kAclDumpPlacementDevice;
      tensor.shape_num = 1;
      tensor.origin_shape_num = 1;
      // Describe a flat ND tensor in elements, while tensor_size is bytes.
      // CANN 9.2+ also requires RAW device addresses and explicit dimension
      // counts.
      tensor.shape[0] = source.elements;
      tensor.origin_shape[0] = source.elements;
    }
    if (count == 0) {
      return -2;
    }
    const auto *api = GetExceptionDumpAPI();
    if (api->acldumpSaveExceptionInfo_ == nullptr) {
      return -3;
    }
    return api->acldumpSaveExceptionInfo_(info.kernel_name, "tilelang",
                                          tensors.data(), count);
  } catch (...) {
    return -3;
  }
}

void EnsureExceptionDumpCallback() {
  static std::once_flag registered;
  // call_once retries if the callable throws. A failed registration must not
  // permanently mark the callback as installed or allow an unprotected launch.
  std::call_once(registered, []() {
    const auto *api = GetExceptionDumpAPI();
    if (api->aclrtExceptionInfoCallbackRegister_ == nullptr ||
        api->aclrtGetArgsFromExceptionInfo_ == nullptr ||
        api->aclrtMemcpy_ == nullptr ||
        api->acldumpSaveExceptionInfo_ == nullptr) {
      throw std::runtime_error(
          "tl.ascend_exception_dump requires CANN 9.2+ with "
          "aclrtExceptionInfoCallbackRegister, aclrtGetArgsFromExceptionInfo, "
          "aclrtMemcpy and acldumpSaveExceptionInfo; source CANN's set_env.sh "
          "before starting Python");
    }
    const int32_t result =
        api->aclrtExceptionInfoCallbackRegister_(ExceptionDumpCallback);
    if (result != 0) {
      throw std::runtime_error(
          "aclrtExceptionInfoCallbackRegister failed with ACL error " +
          std::to_string(result));
    }
  });
}

} // namespace ascend
} // namespace tvm
