#include "compat/analyzer.h"
#ifndef TVM_TL_BACKEND_COMMON_OP_ATOMIC_REDUCE_H_
#define TVM_TL_BACKEND_COMMON_OP_ATOMIC_REDUCE_H_
#include "op/prim.h"

#include "op/atomic_reduce.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

namespace tvm::tl::backend {

namespace atomic_reduce {

struct AtomicIteration {
  Array<LayoutAxis> loops;
  Array<PrimExpr> destination_indices;
  PrimExpr value;
  PrimExpr predicate;
};

inline Array<PrimExpr> RemapIndices(const TensorVar &tensor,
                                    Array<PrimExpr> indices,
                                    const LowerArgs *lower_args) {
  if (lower_args && lower_args->buffer_remap.count(tensor) &&
      lower_args->layout_map.count(tensor)) {
    return lower_args->layout_map[tensor]->Forward(indices);
  }
  return indices;
}

inline TensorVar RemapTensor(const TensorVar &tensor,
                             const LowerArgs *lower_args) {
  return lower_args ? lower_args->buffer_remap.Get(tensor).value_or(tensor)
                    : tensor;
}

inline AtomicIteration MakeIteration(const AtomicOpBaseNode &op,
                                     sym::Analyzer analyzer,
                                     const LowerArgs *lower_args = nullptr) {
  Array<LayoutAxis> loops;
  for (const Range &range : op.dst_range) {
    if (!IsOne(range->extent)) {
      PrimVar variable("atomic_axis_" + std::to_string(loops.size()),
                       range->extent.ty());
      loops.push_back(LayoutAxis(
          Range::FromMinExtent(tl::MakeConst(variable.ty(), 0), range->extent),
          variable));
      tl::arith::Bind(analyzer, variable, loops.back()->dom);
    }
  }
  auto make_indices = [&](const Array<Range> &ranges) {
    Array<PrimExpr> indices;
    size_t axis = 0;
    for (const Range &range : ranges) {
      if (IsOne(range->extent)) {
        indices.push_back(range->begin);
      } else {
        TVM_FFI_CHECK_LT(axis, loops.size(), ValueError)
            << "Atomic region extent mismatch";
        indices.push_back(range->begin + loops[axis++]->var);
      }
    }
    return indices;
  };
  PrimExpr predicate = tl::MakeConst(PrimType::Bool(), true);
  auto add_bounds = [&](const TensorVar &tensor,
                        const Array<PrimExpr> &indices) {
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      PrimExpr condition =
          indices[axis] >= 0 && indices[axis] < tensor.shape()[axis];
      if (!tl::arith::CanProve(analyzer, condition)) {
        predicate = predicate && condition;
      }
    }
  };
  auto destination_indices = make_indices(op.dst_range);
  add_bounds(op.dst, destination_indices);
  destination_indices = RemapIndices(op.dst, destination_indices, lower_args);
  PrimExpr value = op.src_value.value_or(tl::MakeConst(op.dst.dtype(), 0));
  if (op.src.has_value()) {
    TensorVar source = op.src.value();
    auto source_indices = make_indices(op.src_range);
    add_bounds(source, source_indices);
    value = TensorLoad(RemapTensor(source, lower_args),
                       RemapIndices(source, source_indices, lower_args));
  }
  value = Cast(op.dst.dtype(), value);
  return {loops, destination_indices, value,
          tl::arith::Simplify(analyzer, predicate)};
}

inline For MakeSIMTLoop(const AtomicOpBaseNode &op, sym::Analyzer analyzer) {
  auto iteration = MakeIteration(op, analyzer);
  if (iteration.loops.empty()) {
    iteration.loops.push_back(
        LayoutAxis(Range(0, 1), PrimVar("atomic_scalar")));
  }
  Expr pointer = MakeAccessPtrFromBufferLoad(
      TensorLoad(op.dst, iteration.destination_indices), kAccessReadWrite);
  Call update(TupleType(Array<Type>{}), op.GetElemOp(),
              {pointer, iteration.value, PrimExpr(op.GetMemoryOrder())},
              DictAttrs(op.annotations));
  Stmt body = Evaluate(update);
  if (!IsOne(iteration.predicate)) {
    body = If(iteration.predicate, SeqStmt(body));
  }
  for (size_t axis = iteration.loops.size(); axis-- > 0;) {
    const auto &loop = iteration.loops[axis];
    BlockAnnotations annotations;
    if (axis == 0) {
      if (auto width = op.annotations.Get(attr::kCoalescedWidth)) {
        annotations.Set(attr::kCoalescedWidth, width.value());
      }
    }
    body = For(loop->var, tl::MakeConst(loop->var.ty(), 0), loop->dom->extent,
               tl::MakeConst(loop->var.ty(), 1), ForKind::kParallel,
               SeqStmt(body), annotations);
  }
  return body.as_or_throw<For>();
}

inline bool IsEmpty(const AtomicOpBaseNode &op, sym::Analyzer analyzer) {
  for (const Range &range : op.dst_range) {
    if (tl::arith::CanProve(analyzer, range->extent == 0)) {
      return true;
    }
  }
  return false;
}

inline LayoutMap InferSIMTLayout(const AtomicOpBaseNode &op,
                                 const LayoutInferArgs &layout_args,
                                 InferLevel level) {
  if (IsEmpty(op, layout_args.analyzer)) {
    return {};
  }
  if (!op.par_op_.defined()) {
    auto loop = MakeSIMTLoop(op, layout_args.analyzer);
    op.par_op_ = ParallelOp(ParallelLoopFuser::Fuse(loop).as_or_throw<For>());
  }
  return op.par_op_->InferLayout(layout_args, level);
}

} // namespace atomic_reduce

struct AtomicReduce {
  static LayoutMap InferLayout(const AtomicOpBaseNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return atomic_reduce::InferSIMTLayout(op, layout_args, level);
  }

  static Stmt Lower(const AtomicOpBaseNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    if (atomic_reduce::IsEmpty(op, analyzer)) {
      return SeqStmt(Array<Stmt>{});
    }
    for (auto level :
         {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
      InferLayout(op,
                  {lower_args.target,
                   lower_args.thread_bounds,
                   lower_args.layout_map,
                   analyzer,
                   lower_args.buffer_remap,
                   {}},
                  level);
    }
    auto loop_layout = op.par_op_->GetLoopLayout();
    return LowerParallelLoop(
        op.par_op_->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
        lower_args.layout_map,
        op.par_op_->GetPredicate(lower_args.thread_index), true,
        op.par_op_->LoopLayoutRequiresPaddingGuard());
  }
};

} // namespace tvm::tl::backend

#endif
