/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

#ifndef TVM_TL_TRANSFORM_COMMON_ATTR_H_
#define TVM_TL_TRANSFORM_COMMON_ATTR_H_

#include <string>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

constexpr const char *HostMainBlockName = "root";

constexpr const char *DeviceMainBlockName = "tilelang_root";

inline bool IsHostMainBlock(const tirx::SBlockNode *node) {
  return node->name_hint == HostMainBlockName;
}

inline bool IsDeviceMainBlock(const tirx::SBlockNode *node) {
  return node->name_hint == DeviceMainBlockName;
}

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

constexpr const char *tilelang_is_npu_kernel_frame =
    "tilelang.is_npu_kernel_frame";

constexpr const char *tilelang_simt_vf_captures = "tl.simt_vf_captures";

constexpr const char *tilelang_simd_vf_captures = "tl.simd_vf_captures";

namespace attr {
// Attributes to mark CUDA sync calls
constexpr const char *kHasTriggerLaunch = "has_cuda_pdl_trigger";
constexpr const char *kHasGridSync = "has_cuda_pdl_sync";

// TileLang-only AttrStmt keys.
constexpr const char *volatile_scope = "volatile_scope";
constexpr const char *coproc_scope = "coproc_scope";
constexpr const char *pipeline_exec_scope = "pipeline_exec_scope";
// Marks user-authored assumptions that require a host runtime check. The
// corresponding tl.assume remains in the IR as an optimizer fact.
constexpr const char *kAssumeRequiresRuntimeCheck =
    "tl.assume_requires_runtime_check";

// Compiler-internal physical buffer version count.
constexpr const char *kBufferVersion = "tl.buffer_version";

// A user-authored logical Ascend per-core task whose dependency-bearing
// statements all issue on one hardware pipe. AutoSchedule schedules it once,
// then expands synchronization back to its guarded candidate sites.
constexpr const char *kAscendPerCoreTask = "tl.ascend_per_core_task";

// Groups statements into one AutoSchedule TaskNode. Inside a
// kAscendPerCoreTask region, each marker is one concrete candidate. This marker
// carries task semantics and compiler-internal core ownership; stage metadata
// lives on kScheduleUnit.
constexpr const char *kAscendTask = "tl.ascend_task";

// Assigns one frontend-requested software-pipeline stage to every scheduler
// task materialized from the enclosed statements. Consumed by
// MaterializeScheduleUnits before AutoSchedule.
constexpr const char *kAscendStage = "tl.ascend_stage";

// Short-lived scheduled-TIR wrapper shared by AutoSchedule, AssignCore,
// PrepareMultiBuffer, ResolveCore, InsertSync, MaterializeMultiBuffer, and
// LowerScheduledTIR. It carries stage/guard metadata without introducing task
// grouping or core-placement semantics.
constexpr const char *kScheduleUnit = "tl.schedule_unit";

// Attributes to implement SourceCodeBlock
constexpr const char *kCodeBlockSource = "code_block_source";
constexpr const char *kCodeBlockEntryName = "code_block_entry_name";

/*!
 * \brief Check if attr_key is a code block key extension
 * \param attr_key The attr key to be compared
 * \return true if it is a code block key
 */
inline bool IsCodeBlockKey(const std::string &attr_key) {
  return attr_key.compare(0, 11, "code_block_") == 0;
}

} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ATTR_H_
