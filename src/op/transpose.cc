/*!
 * \file tl/op/transpose.cc
 * \brief Transpose operator: dst[j, i] = src[i, j] using SIMT loops.
 */

#include "transpose.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

Transpose::Transpose(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<TransposeNode> node = tvm::ffi::make_object<TransposeNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  data_ = std::move(node);
}

TileOperator TransposeNode::Clone() const {
  auto op = tvm::ffi::make_object<TransposeNode>(*this);
  return Transpose(op);
}

Array<IterVar> TransposeNode::MakeIterVars() const {
  // Use src_range as the iteration domain (src is the "inner" side).
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, src_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

Array<PrimExpr> TransposeNode::MakeIndices(const Array<IterVar> &ivs,
                                           int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  if (src_dst == 1) {
    // Transpose: reverse the loop variable assignment for non-trivial dims.
    std::vector<size_t> nontrivial;
    for (size_t i = 0; i < ranges.size(); i++) {
      if (!is_one(ranges[i]->extent))
        nontrivial.push_back(i);
    }
    ICHECK(nontrivial.size() == ivs.size())
        << "Transpose: nontrivial dims (" << nontrivial.size()
        << ") != ivs size (" << ivs.size() << ") for dst=" << dst->name;
    size_t N = nontrivial.size();
    size_t nt_idx = 0;
    for (size_t i = 0; i < ranges.size(); i++) {
      if (is_one(ranges[i]->extent)) {
        indices.push_back(ranges[i]->min);
      } else {
        size_t rev = N - 1 - nt_idx;
        indices.push_back(ranges[i]->min + ivs[rev]->var);
        nt_idx++;
      }
    }
  } else {
    // Source: direct mapping.
    size_t idx = 0;
    for (size_t i = 0; i < ranges.size(); i++) {
      if (is_one(ranges[i]->extent))
        indices.push_back(ranges[i]->min);
      else {
        indices.push_back(ranges[i]->min + ivs[idx]->var);
        idx++;
      }
    }
    ICHECK(idx == ivs.size())
        << "idx = " << idx << ", ivs.size() = " << ivs.size()
        << " src name = " << src->name << ", dst name = " << dst->name;
  }
  return indices;
}

PrimExpr TransposeNode::MakePredicate(arith::Analyzer *analyzer,
                                      const Array<IterVar> &ivs,
                                      Array<PrimExpr> extents,
                                      int src_dst) const {
  bool do_transpose = (src_dst == 1);
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  size_t num_nontrivial = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (!is_one(ranges[i]->extent))
      num_nontrivial++;
  }

  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
    size_t iv_idx = do_transpose ? (num_nontrivial - 1 - idx) : idx;
    PrimExpr cond = ranges[i]->min + ivs[iv_idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[iv_idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  PrimExpr cond = cond_list[0];
  for (size_t i = 1; i < cond_list.size(); i++)
    cond = And(cond, cond_list[i]);
  return cond;
}

For TransposeNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body);
  }
  return Downcast<For>(body);
}

Stmt TransposeNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  // Transpose always uses normal SIMT lowering (no TMA/LDSM/etc.).
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  if (is_cpu_target || IsLocalBuffer(src) || IsLocalBuffer(dst)) {
    auto vectorized_loop = VectorizeLoop(fused_loop, T.layout_map);
    return vectorized_loop;
  } else {
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
    return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                             analyzer, T.layout_map,
                             par_op->GetPredicate(T.thread_var));
  }
}

LayoutMap TransposeNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  // Transpose always uses SIMT loops; no special layout inference needed.
  return {};
}

TIR_REGISTER_TL_TILE_OP(Transpose, transpose)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { TransposeNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
