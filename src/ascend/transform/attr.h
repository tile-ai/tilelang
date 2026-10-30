/*!
 * \file tl/ascend/transform/attr.h
 * \brief Attributes owned by the Ascend PTO pipeline.
 *
 * Scheduler attributes annotate the IR only between MaterializeScheduleUnits
 * and LowerScheduledTIR.  VMI allocation metadata travels from the frontend to
 * PTO code generation through the early Ascend PTO passes.  These attributes
 * live with the passes that read them rather than in transform/common/attr.h.
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

// Allocation category of an explicit VMI register allocation.  Frontend blocks
// carry a Var -> String map, NormalizePTOAllocMarkers turns it into scoped
// AttrStmt carriers inside the owning SIMD_VF block, and
// ImportPTOAllocMarkers transfers each kind to the matching AllocBuffer.
// `data` marks a mutable single-register slot, `data_array` marks a statically
// indexed array of registers.  Parsing and contract checks live in
// ascend/transform/vmi_alloc.h.
constexpr const char *kVmiAllocKind = "tl.vmi.alloc_kind";
constexpr const char *kVmiAllocKindData = "data";
constexpr const char *kVmiAllocKindDataArray = "data_array";

} // namespace attr
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_ATTR_H_
