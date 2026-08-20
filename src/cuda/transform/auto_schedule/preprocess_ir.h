#ifndef TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_PREPROCESS_IR_H_
#define TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_PREPROCESS_IR_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

// Give every schedulable statement a stable "tl.ws_op_id" marker in the
// forms MaterializeWSSchedule consumes. Existing ids are kept, so the
// rewrite is idempotent. Also enforces the schedulability contract:
// constructs no scheduler can handle (loop_break, atomics, asynchronous
// tile ops nested inside opaque ops) fail hard — auto scheduling is
// opt-in, so an unschedulable kernel is a caller bug, not a fallback.
tvm::tirx::Stmt PreprocessIR(tvm::tirx::Stmt body, const Target &target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_CUDA_TRANSFORM_AUTO_SCHEDULE_PREPROCESS_IR_H_
