#ifndef TVM_TL_ASCEND_RUNTIME_ASCEND_PROFILING_H_
#define TVM_TL_ASCEND_RUNTIME_ASCEND_PROFILING_H_

#include <aprof_pub.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <tvm/runtime/logging.h>

namespace tvm {
namespace ascend {

class AscendProfiler {
public:
  struct KernelInfo {
    uint32_t task_type{MSPROF_GE_TASK_TYPE_INVALID};
    uint32_t mix_ratio{0};
    bool is_mix{false};
  };

  static AscendProfiler *GetIfEnabled() {
    if (!IsEnabledByEnv()) {
      return nullptr;
    }
    static auto *profiler = new AscendProfiler();
    return profiler->available_ ? profiler : nullptr;
  }

  uint64_t GetCycleTime() const { return sys_cycle_time_(); }

  uint64_t GetCollectionFlags(int32_t device_id) const {
    auto &state = State();
    if (!state.active.load(std::memory_order_acquire)) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.device_flags.find(device_id);
    return it == state.device_flags.end() ? 0 : it->second;
  }

  void ReportLaunch(const std::string &kernel_name, uint64_t begin,
                    uint64_t end, uint64_t flags, const KernelInfo &kernel,
                    uint32_t num_blocks) const {
    const uint32_t thread_id = static_cast<uint32_t>(syscall(SYS_gettid));
    const uint64_t item_id =
        get_hash_id_(kernel_name.data(), kernel_name.size());

    MsprofApi event{};
    event.level = MSPROF_REPORT_NODE_LEVEL;
    event.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
    event.threadId = thread_id;
    event.beginTime = begin;
    event.endTime = end;
    event.itemId = item_id;
    WarnOnReportFailure(report_api_(1, &event), "MsprofReportApi");
    // Place metadata timestamps within the Node launch interval for
    // correlation.
    const uint64_t timestamp = begin < end ? begin + 1 : begin;
    if (kernel.is_mix) {
      ReportContextId(item_id, thread_id, timestamp);
    }
    if (flags & PROF_TASK_TIME_L1_MASK) {
      ReportBasicInfo(item_id, thread_id, timestamp, kernel, num_blocks);
    }
  }

private:
  using SysCycleTimeFn = decltype(&MsprofSysCycleTime);
  using GetHashIdFn = decltype(&MsprofGetHashId);
  using ReportApiFn = decltype(&MsprofReportApi);
  using ReportCompactInfoFn = decltype(&MsprofReportCompactInfo);
  using ReportAdditionalInfoFn = decltype(&MsprofReportAdditionalInfo);

  struct CollectionState {
    std::atomic<bool> active{false};
    std::mutex mutex;
    std::unordered_map<int32_t, uint64_t> device_flags;
  };

  static CollectionState &State() {
    // CANN retains the callback; keep its state alive until process exit.
    static auto *state = new CollectionState();
    return *state;
  }

  static int32_t ControlCallback(uint32_t type, void *data,
                                 uint32_t len) noexcept {
    if (type != PROF_CTRL_SWITCH) {
      return 0;
    }
    try {
      if (data == nullptr || len < sizeof(MsprofCommandHandle)) {
        LOG(WARNING) << "Ascend profiling rejected a control callback: size="
                     << len;
        return -1;
      }
      const auto &command = *static_cast<const MsprofCommandHandle *>(data);
      if (command.devNums > MSPROF_MAX_DEV_NUM) {
        LOG(WARNING) << "Ascend profiling rejected a control callback: devices="
                     << command.devNums;
        return -1;
      }
      auto &state = State();
      std::lock_guard<std::mutex> lock(state.mutex);
      if (command.type == PROF_COMMANDHANDLE_TYPE_FINALIZE) {
        state.device_flags.clear();
      } else if (command.type == PROF_COMMANDHANDLE_TYPE_START ||
                 command.type == PROF_COMMANDHANDLE_TYPE_STOP) {
        for (uint32_t i = 0; i < command.devNums; ++i) {
          const int32_t device_id = static_cast<int32_t>(command.devIdList[i]);
          if (command.type == PROF_COMMANDHANDLE_TYPE_START) {
            // Only enable collection flags the tool actually requested.
            const uint64_t flags =
                command.profSwitch &
                (PROF_TASK_TIME_MASK | PROF_TASK_TIME_L1_MASK);
            if (flags != 0) {
              state.device_flags[device_id] |= flags;
            }
          } else {
            // STOP stops collection for this device; it must not depend on the
            // profSwitch bits the tool carries. erase() is idempotent and
            // matches FINALIZE.
            state.device_flags.erase(device_id);
          }
        }
      }
      state.active.store(!state.device_flags.empty(),
                         std::memory_order_release);
      return 0;
    } catch (...) {
      // Never let a C++ exception escape into the CANN control callback.
      return -1;
    }
  }

  static bool IsEnabledByEnv() {
    const char *raw = std::getenv("TILELANG_ASCEND_PROFILE");
    if (raw == nullptr) {
      return false;
    }
    std::string value(raw);
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return false;
    }
    value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    for (char &character : value) {
      character = static_cast<char>(
          std::tolower(static_cast<unsigned char>(character)));
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
  }

  template <typename FunctionType>
  static FunctionType LoadSymbol(void *library, const char *name) {
    return reinterpret_cast<FunctionType>(dlsym(library, name));
  }

  void WarnOnReportFailure(int32_t result, const char *api) const {
    if (result != 0 &&
        !report_failure_warned_.exchange(true, std::memory_order_relaxed)) {
      LOG(WARNING) << "Ascend profiling " << api << " failed: " << result
                   << "; further report failures are suppressed";
    }
  }

  void ReportContextId(uint64_t item_id, uint32_t thread_id,
                       uint64_t timestamp) const {
    MsprofAdditionalInfo event{};
    event.level = MSPROF_REPORT_NODE_LEVEL;
    event.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    event.threadId = thread_id;
    event.timeStamp = timestamp;
    event.dataLen = static_cast<uint32_t>(sizeof(MsprofContextIdInfo));

    MsprofContextIdInfo context{};
    static_assert(sizeof(context) <= sizeof(event.data),
                  "Msprof context exceeds additional-info payload");
    context.opName = item_id;
    context.ctxIdNum = 1;
    context.ctxIds[0] = 0;
    std::memcpy(event.data, &context, sizeof(context));
    WarnOnReportFailure(report_additional_info_(
                            1, &event, static_cast<uint32_t>(sizeof(event))),
                        "MsprofReportAdditionalInfo(context)");
  }

  void ReportBasicInfo(uint64_t item_id, uint32_t thread_id, uint64_t timestamp,
                       const KernelInfo &kernel, uint32_t num_blocks) const {
    MsprofCompactInfo info{};
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
    info.threadId = thread_id;
    info.timeStamp = timestamp;
    auto &basic = info.data.nodeBasicInfo;
    basic.opName = item_id;
    basic.opType = item_id;
    basic.taskType = kernel.task_type;
    // Msprof packs the block count and secondary-core ratio into 16 bits each.
    // When the block count exceeds UINT16_MAX, clear the low half but keep the
    // high-half mix ratio, which stays representable.
    if (kernel.task_type != MSPROF_GE_TASK_TYPE_INVALID) {
      const uint32_t block = num_blocks <= UINT16_MAX ? num_blocks : 0;
      basic.blockDim = block | (kernel.mix_ratio << 16);
    }
    WarnOnReportFailure(
        report_compact_info_(1, &info, static_cast<uint32_t>(sizeof(info))),
        "MsprofReportCompactInfo(basic)");
  }

  AscendProfiler() {
    library_ = dlopen("libprofapi.so", RTLD_LAZY | RTLD_LOCAL);
    if (library_ == nullptr) {
      LOG(WARNING) << "Ascend native profiling could not load libprofapi.so: "
                   << dlerror();
      return;
    }
    sys_cycle_time_ =
        LoadSymbol<SysCycleTimeFn>(library_, "MsprofSysCycleTime");
    get_hash_id_ = LoadSymbol<GetHashIdFn>(library_, "MsprofGetHashId");
    report_api_ = LoadSymbol<ReportApiFn>(library_, "MsprofReportApi");
    report_compact_info_ =
        LoadSymbol<ReportCompactInfoFn>(library_, "MsprofReportCompactInfo");
    report_additional_info_ = LoadSymbol<ReportAdditionalInfoFn>(
        library_, "MsprofReportAdditionalInfo");
    const auto register_callback =
        LoadSymbol<decltype(&MsprofRegisterCallback)>(library_,
                                                      "MsprofRegisterCallback");
    if (sys_cycle_time_ == nullptr || get_hash_id_ == nullptr ||
        report_api_ == nullptr || report_compact_info_ == nullptr ||
        report_additional_info_ == nullptr || register_callback == nullptr) {
      LOG(WARNING)
          << "Ascend native profiling disabled: incomplete libprofapi API";
      return;
    }
    // Module 0 is shared by external frameworks; registration adds a callback.
    available_ = register_callback(0, ControlCallback) == 0;
    if (!available_) {
      LOG(WARNING) << "Ascend native profiling callback registration failed";
    }
  }

  void *library_{nullptr};
  bool available_{false};
  mutable std::atomic<bool> report_failure_warned_{false};
  SysCycleTimeFn sys_cycle_time_{nullptr};
  GetHashIdFn get_hash_id_{nullptr};
  ReportApiFn report_api_{nullptr};
  ReportCompactInfoFn report_compact_info_{nullptr};
  ReportAdditionalInfoFn report_additional_info_{nullptr};
};

} // namespace ascend
} // namespace tvm
#endif // TVM_TL_ASCEND_RUNTIME_ASCEND_PROFILING_H_
