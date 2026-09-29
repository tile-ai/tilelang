/*!
 * \file tl/ascend/transform/attr.h
 * \brief Attributes owned by the Ascend scheduler and its scheduled-TIR
 * pipeline.
 *
 * These annotate the IR only between MaterializeScheduleUnits and
 * LowerScheduledTIR, so they live with the passes that read them rather than in
 * transform/common/attr.h.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_ATTR_H_
#define TVM_TL_ASCEND_TRANSFORM_ATTR_H_

namespace tvm {
namespace tl {
namespace attr {

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

} // namespace attr
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_ATTR_H_
