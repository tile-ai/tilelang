/*!
 * \file tl/backend/common/op/fill.h
 * \brief Shared tl.fill lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FILL_H_
#define TVM_TL_BACKEND_COMMON_OP_FILL_H_

#include "op/fill.h"
#include "op/parallel.h"
#include <tvm/runtime/logging.h>

#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    if (IsLocalBuffer(op.dst, true)) {
      return PragmaUnrollLoop(VectorizeLoop(op.MakeSIMTLoop(analyzer), analyzer,
                                            lower_args.layout_map));
    }
    if (IsFragmentBuffer(op.dst) || IsSharedBuffer(op.dst) ||
        IsGlobalBuffer(op.dst)) {
      auto parallel = ParallelOp(op.MakeSIMTLoop(analyzer));
      parallel->InferLayout(
          {lower_args.target, lower_args.thread_bounds, lower_args.layout_map,
           analyzer, lower_args.buffer_remap, lower_args.bind_var_to_expr},
          InferLevel::kFree);
      return LowerParallelLoop(parallel->GetRoot(), parallel->GetLoopLayout(),
                               lower_args.thread_index, analyzer,
                               lower_args.layout_map,
                               parallel->GetPredicate(lower_args.thread_index),
                               true, parallel->LoopLayoutRequiresPaddingGuard(),
                               lower_args.target->kind->name != "c");
    }
    TVM_FFI_THROW(ValueError)
        << "Unsupported fill scope " << op.dst.scope().name();
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FILL_H_
