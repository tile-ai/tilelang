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

#include "builtin.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<AtomicReduceImpl> &AtomicReduceImplRegistry() {
  static std::vector<AtomicReduceImpl> registry;
  return registry;
}

const AtomicReduceImpl &ResolveAtomicReduceImpl(Target target) {
  const auto &registry = AtomicReduceImplRegistry();
  const AtomicReduceImpl *matched_impl = nullptr;
  for (const AtomicReduceImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.atomic_reduce found multiple target-specific "
             "implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_reduce requires a target-specific implementation, but no "
         "atomic_reduce implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterAtomicReduceImpl(AtomicReduceImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  AtomicReduceImplRegistry().push_back(impl);
}

// ============================================================================
// AtomicMax Implementation
// ============================================================================

AtomicMax::AtomicMax(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "AtomicMax expects at least 2 arguments (src, dst), got "
      << args.size();
  ObjectPtr<AtomicMaxNode> node = tvm::ffi::make_object<AtomicMaxNode>();
  std::vector<AccessRegion> access_regions;

  if (IsBufferLikeExpr(args[0])) {
    auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
    node->src = src_access.region->buffer;
    node->src_range = src_access.region->region;
    access_regions.push_back(std::move(src_access));
  } else {
    node->src_value = args[0];
  }

  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  dst_access.access_mask = kAccessReadWrite;
  node->dst = dst_access.region->buffer;
  node->dst_range = dst_access.region->region;
  access_regions.push_back(std::move(dst_access));
  node->SetAccessRegions(std::move(access_regions));

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
  std::vector<AccessRegion> access_regions;

  if (IsBufferLikeExpr(args[0])) {
    auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
    node->src = src_access.region->buffer;
    node->src_range = src_access.region->region;
    access_regions.push_back(std::move(src_access));
  } else {
    node->src_value = args[0];
  }

  auto dst_access = NormalizeToAccessRegion(args[1], kAccessReadWrite);
  dst_access.access_mask = kAccessReadWrite;
  node->dst = dst_access.region->buffer;
  node->dst_range = dst_access.region->region;
  access_regions.push_back(std::move(dst_access));
  node->SetAccessRegions(std::move(access_regions));

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

LayoutMap AtomicOpBaseNode::InferLayout(const LayoutInferArgs &T,
                                        InferLevel level) const {
  return ResolveAtomicReduceImpl(T.target).infer_layout(*this, T, level);
}

Stmt AtomicOpBaseNode::Lower(const LowerArgs &T,
                             arith::Analyzer *analyzer) const {
  return ResolveAtomicReduceImpl(T.target).lower(*this, T, analyzer);
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
