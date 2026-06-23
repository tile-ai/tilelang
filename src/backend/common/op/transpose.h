/*!
 * \file tl/backend/common/op/transpose.h
 * \brief Shared tl.transpose lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_TRANSPOSE_H_
#define TVM_TL_BACKEND_COMMON_OP_TRANSPOSE_H_

#include "op/transpose.h"
#include <tvm/ir/cast.h>

#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <dlpack/dlpack.h>

#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;

struct Transpose {
  static Stmt Lower(const TransposeNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    bool is_cpu_target = lower_args.target->GetTargetDeviceType() == kDLCPU;
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

    if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
      return VectorizeLoop(fused_loop, lower_args.layout_map);
    }

    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           false,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(
        par_op->GetRoot(), loop_layout, lower_args.thread_var, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_var),
        /*parallel_loop=*/true, /*should_vectorize=*/true,
        par_op->LoopLayoutRequiresPaddingGuard());
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_TRANSPOSE_H_
