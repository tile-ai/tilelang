/*!
 * \file tl/backend/common/op/fill.h
 * \brief Shared tl.fill lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FILL_H_
#define TVM_TL_BACKEND_COMMON_OP_FILL_H_

#include "op/fill.h"
#include <tvm/runtime/logging.h>

#include "op/parallel.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace prim;

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    sym::AnalyzerObj *analyzer) {
    if (IsFragmentBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(
          par_op->GetRoot(), lower_args.thread_index, analyzer,
          par_op->GetLoopLayout(), par_op->LoopLayoutRequiresPaddingGuard());
      auto vectorized_loop = VectorizeLoop(
          thread_loop, analyzer, lower_args.layout_map, -1, lower_args.target);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

      if (par_op->GetPredicate(lower_args.thread_index).has_value()) {
        return If(par_op->GetPredicate(lower_args.thread_index).value(),
                  SeqStmt(unrolled_loop));
      }
      return unrolled_loop;
    }

    if (IsLocalBuffer(op.dst) || IsLocalVarBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(
          init_loop, analyzer, lower_args.layout_map, -1, lower_args.target);
      return PragmaUnrollLoop(vectorized_loop);
    }

    if (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(
          par_op->GetRoot(), lower_args.thread_index, analyzer,
          par_op->GetLoopLayout(), par_op->LoopLayoutRequiresPaddingGuard());
      auto vectorized_loop = VectorizeLoop(
          thread_loop, analyzer, lower_args.layout_map, -1, lower_args.target);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
      if (par_op->GetPredicate(lower_args.thread_index).has_value()) {
        return If(par_op->GetPredicate(lower_args.thread_index).value(),
                  SeqStmt(unrolled_loop));
      }
      return unrolled_loop;
    }

    LOG(FATAL) << "Unsupported scope " << op.dst.scope();
    TVM_FFI_THROW(ValueError) << "Unsupported lowering scope";
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FILL_H_
