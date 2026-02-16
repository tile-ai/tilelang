/*!
 * \file tl/op/atomic_reduce.cc
 *
 * Define atomic reduction operators (max/min).
 */

#include "./atomic_reduce.h"
#include "utils.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../layout/layout.h"
#include "../target/utils.h"

#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

// ============================================================================
// AtomicMax Implementation
// ============================================================================

AtomicMax::AtomicMax(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "AtomicMax expects at least 2 arguments (src, dst), got "
      << args.size();
  ObjectPtr<AtomicMaxNode> node = tvm::ffi::make_object<AtomicMaxNode>();

  if (IsBufferLikeExpr(args[0])) {
    auto region = NormalizeToBufferRegion(args[0]);
    node->src = region->buffer;
    node->src_range = region->region;
  } else {
    node->src_value = args[0];
  }

  auto region = NormalizeToBufferRegion(args[1]);
  node->dst = region->buffer;
  node->dst_range = region->region;

  node->annotations = annotations;
  data_ = std::move(node);
}

TileOperator AtomicMaxNode::Clone() const {
  auto op = tvm::ffi::make_object<AtomicMaxNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return AtomicMax(op);
}

const Op &AtomicMaxNode::GetElemOp() const { return atomic_max_elem_op(); }

// ============================================================================
// AtomicMin Implementation
// ============================================================================

AtomicMin::AtomicMin(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "AtomicMin expects at least 2 arguments (src, dst), got "
      << args.size();
  ObjectPtr<AtomicMinNode> node = tvm::ffi::make_object<AtomicMinNode>();

  if (IsBufferLikeExpr(args[0])) {
    auto region = NormalizeToBufferRegion(args[0]);
    node->src = region->buffer;
    node->src_range = region->region;
  } else {
    node->src_value = args[0];
  }

  auto region = NormalizeToBufferRegion(args[1]);
  node->dst = region->buffer;
  node->dst_range = region->region;

  node->annotations = annotations;
  data_ = std::move(node);
}

TileOperator AtomicMinNode::Clone() const {
  auto op = tvm::ffi::make_object<AtomicMinNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return AtomicMin(op);
}

const Op &AtomicMinNode::GetElemOp() const { return atomic_min_elem_op(); }

// ============================================================================
// Common AtomicOpBaseNode Implementation
// ============================================================================

Array<IterVar> AtomicOpBaseNode::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  // Make IterVars according to dst, not src
  // Since src may be a scalar Expr
  for (size_t i = 0; i < dst_range.size(); i++) {
    if (is_one(dst_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, dst_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, dst_range[i]->extent), var, IterVarType::kDataPar});
  }

  // If is scalar, create a dummy loop var
  if (loop_vars.empty()) {
    Var var = Var("i");
    loop_vars.push_back({Range(0, 1), var, IterVarType::kDataPar});
  }

  return loop_vars;
}

Array<PrimExpr> AtomicOpBaseNode::MakeIndices(const Array<IterVar> &ivs,
                                              int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }

  // Special case: scalar range, when there is one var and one range(0, 1)
  ICHECK(idx == ivs.size() || (idx == 0 && ivs.size() == 1))
      << "Unmatched indices: idx = " << idx << ", ivs.size() = " << ivs.size()
      << ", dst name = " << dst->name;
  return indices;
}

PrimExpr AtomicOpBaseNode::MakePredicate(arith::Analyzer *analyzer,
                                         const Array<IterVar> &ivs,
                                         Array<PrimExpr> extents,
                                         int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

For AtomicOpBaseNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  ICHECK(!loop_vars.empty()) << "MakeIterVars in AtomicOp should not return "
                                "empty vars (at least 1 var)";

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", dst = " << dst->name;

  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);
  Array<PrimExpr> new_args;

  // Src arg to be passed to the Call atomic operation
  PrimExpr src_value_arg;

  // If src is a Buffer
  if (!src_value.defined()) {
    ICHECK(loop_vars.size() <= src_range.size())
        << "loop_vars.size() = " << loop_vars.size()
        << ", src_range.size() = " << src_range.size()
        << ", src = " << src->name << ", dst = " << dst->name;

    Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
    // Load source value
    src_value_arg = BufferLoad(src, src_indices);
  } else {
    src_value_arg = src_value;
  }
  // Cast to dst dtype if needed
  if (src_value_arg->dtype != dst->dtype)
    src_value_arg = Cast(dst->dtype, src_value_arg);

  // Build an access pointer to the destination element (rw).
  DataType idx_dtype =
      dst_indices.empty() ? DataType::Int(32) : dst_indices[0].dtype();
  PrimExpr dst_ptr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(dst, dst_indices), make_const(idx_dtype, 1),
            make_const(DataType::Int(32), 3)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value_arg);
  new_args.push_back(GetMemoryOrder());

  // Use the appropriate elem_op based on the derived type (via virtual call)
  Call atomic_call =
      tvm::tir::Call(dst->dtype, GetElemOp(), new_args, annotations);

  Stmt body = tvm::tir::Evaluate(atomic_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

LayoutMap AtomicOpBaseNode::InferLayout(const LayoutInferArgs &T,
                                        InferLevel level) const {
  // For atomic reduce operations, check that src and dst have the same layout
  // if both are fragments
  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst)) {
    if (T.layout_map.count(src) && T.layout_map.count(dst)) {
      Layout src_layout = T.layout_map.at(src);
      Layout dst_layout = T.layout_map.at(dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "Atomic reduce requires src and dst to have the same layout, but "
             "got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << src->name << ", dst buffer: " << dst->name;
    }
  }
  return {};
}

Stmt AtomicOpBaseNode::Lower(const LowerArgs &T,
                             arith::Analyzer *analyzer) const {
  Target target = T.target;

  auto simt_loop = MakeSIMTLoop(analyzer);
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
  auto lowered_loop =
      LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                        T.layout_map, par_op->GetPredicate(T.thread_var));
  return lowered_loop;
}

// ============================================================================
// Operator Registration
// ============================================================================

TIR_REGISTER_TL_TILE_OP(AtomicMax, atomicmax)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_TILE_OP(AtomicMin, atomicmin)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  AtomicMaxNode::RegisterReflection();
  AtomicMinNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
