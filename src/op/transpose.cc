/*!
 * \file tl/op/transpose.cc
 * \brief Transpose operator that swaps the final two axes using SIMT loops.
 */

#include "transpose.h"
#include "support/check.h"
#include <tvm/ir/cast.h>

#include <dlpack/dlpack.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

size_t SourceAxisForDestinationAxis(size_t dst_axis, size_t rank) {
  ICHECK(rank >= 2) << "Transpose requires at least two dimensions";
  if (dst_axis == rank - 2)
    return rank - 1;
  if (dst_axis == rank - 1)
    return rank - 2;
  return dst_axis;
}

std::vector<int> MakeSourceAxisToIterVarMap(Array<Range> ranges,
                                            size_t iv_count) {
  std::vector<int> source_axis_to_iv(ranges.size(), -1);
  size_t iv_idx = 0;
  for (size_t axis = 0; axis < ranges.size(); axis++) {
    if (is_one(ranges[axis]->extent))
      continue;
    source_axis_to_iv[axis] = static_cast<int>(iv_idx++);
  }
  ICHECK_EQ(iv_idx, iv_count)
      << "Transpose: source nontrivial dimensions (" << iv_idx
      << ") != iterator count (" << iv_count << ")";
  return source_axis_to_iv;
}

std::vector<TransposeImpl> &TransposeImplRegistry() {
  static std::vector<TransposeImpl> registry;
  return registry;
}

const TransposeImpl &ResolveTransposeImpl(Target target) {
  const auto &registry = TransposeImplRegistry();
  const TransposeImpl *matched_impl = nullptr;
  for (const TransposeImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.transpose found multiple target-specific implementations for "
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.transpose requires a target-specific implementation, but no "
         "transpose implementation is registered for "
      << target->str();
  return *matched_impl;
}

} // namespace

void RegisterTransposeImpl(TransposeImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  TransposeImplRegistry().push_back(impl);
}

Transpose::Transpose(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<TransposeNode> node = make_object<TransposeNode>();
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
  auto op = make_object<TransposeNode>(*this);
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
    ICHECK_EQ(src_range.size(), ranges.size())
        << "Transpose source and destination ranks must match";
    std::vector<int> source_axis_to_iv =
        MakeSourceAxisToIterVarMap(src_range, ivs.size());
    for (size_t dst_axis = 0; dst_axis < ranges.size(); dst_axis++) {
      size_t src_axis = SourceAxisForDestinationAxis(dst_axis, ranges.size());
      int iv_idx = source_axis_to_iv[src_axis];
      if (iv_idx < 0)
        indices.push_back(ranges[dst_axis]->min);
      else
        indices.push_back(ranges[dst_axis]->min + ivs[iv_idx]->var);
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
                                      const Array<PrimExpr> &indices,
                                      const Array<PrimExpr> &extents) const {
  Array<PrimExpr> cond_list;
  ICHECK_EQ(indices.size(), extents.size())
      << "Transpose index rank (" << indices.size()
      << ") must match buffer rank (" << extents.size() << ")";
  for (size_t i = 0; i < indices.size(); i++) {
    PrimExpr cond = indices[i] < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = indices[i] >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
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

  PrimExpr src_predicate = MakePredicate(analyzer, src_indices, src->shape);
  PrimExpr dst_predicate = MakePredicate(analyzer, dst_indices, dst->shape);

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

Stmt TransposeNode::Lower(const LowerArgs &lower_args,
                          arith::Analyzer *analyzer) const {
  return ResolveTransposeImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap TransposeNode::InferLayout(const LayoutInferArgs &layout_args,
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
