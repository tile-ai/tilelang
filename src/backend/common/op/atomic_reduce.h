/*!
 * \file tl/backend/common/op/atomic_reduce.h
 * \brief Shared tl.atomicmax/tl.atomicmin lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_ATOMIC_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_ATOMIC_REDUCE_H_

#include "backend/common/op/atomic_simt.h"
#include "op/atomic_reduce.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/cast.h>

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <optional>
#include <vector>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

namespace atomic_reduce {

inline For MakeSIMTLoop(const AtomicOpBaseNode &op, arith::Analyzer *analyzer) {
  Optional<BufferRegion> src_region;
  if (!op.src_value.defined()) {
    src_region = BufferRegion(op.src, op.src_range);
  }
  backend::AtomicSIMTIndexMap index_map = backend::MakeAtomicSIMTIndexMap(
      op.GetElemOp(), BufferRegion(op.dst, op.dst_range), src_region);
  const Array<IterVar> &loop_vars = index_map.loop_vars;
  for (const auto &iv : loop_vars) {
    analyzer->Bind(iv->var, iv->dom);
  }

  const Array<PrimExpr> &dst_indices = index_map.dst_indices;
  Array<PrimExpr> new_args;
  PrimExpr src_value_arg = op.src_value;
  if (!op.src_value.defined()) {
    src_value_arg = BufferLoad(op.src, index_map.src_indices);
  }

  if (src_value_arg->dtype != op.dst->dtype) {
    src_value_arg = Cast(op.dst->dtype, src_value_arg);
  }

  DataType idx_dtype =
      dst_indices.empty() ? DataType::Int(32) : dst_indices[0].dtype();
  PrimExpr dst_ptr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(op.dst, dst_indices), make_const(idx_dtype, 1),
            make_const(DataType::Int(32), 3)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value_arg);
  new_args.push_back(op.GetMemoryOrder());

  Call atomic_call =
      tvm::tirx::Call(op.dst->dtype, op.GetElemOp(), new_args, op.annotations);

  Stmt body = tvm::tirx::Evaluate(atomic_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (op.annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             op.annotations.Get(attr::kCoalescedWidth).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

inline LayoutMap InferSIMTLayout(const AtomicOpBaseNode &op,
                                 const LayoutInferArgs &layout_args,
                                 InferLevel) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    if (layout_args.layout_map.count(op.src) &&
        layout_args.layout_map.count(op.dst)) {
      Layout src_layout = layout_args.layout_map.at(op.src);
      Layout dst_layout = layout_args.layout_map.at(op.dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "Atomic reduce requires src and dst to have the same layout, but "
             "got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << op.src->name
          << ", dst buffer: " << op.dst->name;
    }
  }
  return {};
}

} // namespace atomic_reduce

struct AtomicReduce {
  static LayoutMap InferLayout(const AtomicOpBaseNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return atomic_reduce::InferSIMTLayout(op, layout_args, level);
  }

  static Stmt Lower(const AtomicOpBaseNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    auto simt_loop = atomic_reduce::MakeSIMTLoop(op, analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(
        fused_loop, loop_layout, lower_args.thread_index, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
        /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());
  }
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_ATOMIC_REDUCE_H_
