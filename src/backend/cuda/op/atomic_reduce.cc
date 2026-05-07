/*!
 * \file tl/backend/cuda/op/atomic_reduce.cc
 * \brief CUDA implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "op/atomic_reduce.h"

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

Array<IterVar> MakeIterVars(const AtomicOpBaseNode &op) {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < op.dst_range.size(); i++) {
    if (is_one(op.dst_range[i]->extent)) {
      continue;
    }
    Var var = Var(std::string{char('i' + idx)}, op.dst_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, op.dst_range[i]->extent), var, IterVarType::kDataPar});
  }

  if (loop_vars.empty()) {
    Var var = Var("i");
    loop_vars.push_back({Range(0, 1), var, IterVarType::kDataPar});
  }

  return loop_vars;
}

Array<PrimExpr> MakeIndices(const AtomicOpBaseNode &op,
                            const Array<IterVar> &ivs, int src_dst) {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? op.src_range : op.dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) {
      indices.push_back(ranges[i]->min);
    } else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }

  ICHECK(idx == ivs.size() || (idx == 0 && ivs.size() == 1))
      << "Unmatched indices: idx = " << idx << ", ivs.size() = " << ivs.size()
      << ", dst name = " << op.dst->name;
  return indices;
}

For MakeSIMTLoop(const AtomicOpBaseNode &op, arith::Analyzer *analyzer) {
  Array<IterVar> loop_vars = MakeIterVars(op);
  ICHECK(!loop_vars.empty()) << "MakeIterVars in AtomicOp should not return "
                                "empty vars (at least 1 var)";

  for (const auto &iv : loop_vars) {
    analyzer->Bind(iv->var, iv->dom);
  }

  ICHECK(loop_vars.size() <= op.dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << op.dst_range.size()
      << ", dst = " << op.dst->name;

  Array<PrimExpr> dst_indices = MakeIndices(op, loop_vars, 1);
  Array<PrimExpr> new_args;

  PrimExpr src_value_arg;

  if (!op.src_value.defined()) {
    ICHECK(loop_vars.size() <= op.src_range.size())
        << "loop_vars.size() = " << loop_vars.size()
        << ", src_range.size() = " << op.src_range.size()
        << ", src = " << op.src->name << ", dst = " << op.dst->name;

    Array<PrimExpr> src_indices = MakeIndices(op, loop_vars, 0);
    src_value_arg = BufferLoad(op.src, src_indices);
  } else {
    src_value_arg = op.src_value;
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
      tvm::tir::Call(op.dst->dtype, op.GetElemOp(), new_args, op.annotations);

  Stmt body = tvm::tir::Evaluate(atomic_call);

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

LayoutMap InferSIMTLayout(const AtomicOpBaseNode &op, const LayoutInferArgs &T,
                          InferLevel) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    if (T.layout_map.count(op.src) && T.layout_map.count(op.dst)) {
      Layout src_layout = T.layout_map.at(op.src);
      Layout dst_layout = T.layout_map.at(op.dst);
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

} // namespace

struct AtomicReduce {
  static LayoutMap InferLayout(const AtomicOpBaseNode &op,
                               const LayoutInferArgs &T, InferLevel level) {
    return InferSIMTLayout(op, T, level);
  }

  static Stmt Lower(const AtomicOpBaseNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto simt_loop = MakeSIMTLoop(op, analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                             T.layout_map, par_op->GetPredicate(T.thread_var));
  }
};

} // namespace cuda

namespace {

bool MatchCudaAtomicReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "cuda.AtomicReduce",
      MatchCudaAtomicReduceTarget,
      cuda::AtomicReduce::InferLayout,
      cuda::AtomicReduce::Lower,
  });
  return true;
}

const bool cuda_atomic_reduce_registered = RegisterCudaAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
