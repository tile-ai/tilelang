/*!
 * \file tl/op/atomic_reduce.cc
 *
 * Define atomic reduction operators (max/min).
 */

#include "./atomic_reduce.h"
#include "support/check.h"
#include "utils.h"

#include "../layout/layout.h"

#include "builtin.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_reduce requires a target-specific implementation, but no "
         "atomic_reduce implementation is registered for "
      << target->str();
  return *matched_impl;
}

} // namespace

void AtomicOpBaseNode::Initialize(const Array<Expr> &arguments,
                                  BlockAnnotations options) {
  TVM_FFI_CHECK_EQ(arguments.size(), 2, ValueError)
      << "Atomic tile operators require source and destination operands";
  annotations = std::move(options);
  std::vector<AccessRegion> accesses;
  if (IsBufferLikeExpr(arguments[0])) {
    auto access = NormalizeToAccessRegion(arguments[0], kAccessRead);
    src = access.region->source.as_or_throw<TensorVar>();
    src_range = access.region->region;
    accesses.push_back(std::move(access));
  } else {
    src_value = arguments[0].as_or_throw<PrimExpr>();
  }
  auto destination = NormalizeToAccessRegion(arguments[1], kAccessReadWrite);
  destination.access_mask = kAccessReadWrite;
  dst = destination.region->source.as_or_throw<TensorVar>();
  dst_range = destination.region->region;
  accesses.push_back(std::move(destination));
  SetAccessRegions(std::move(accesses));
  sym::Analyzer analyzer;
  Array<PrimExpr> source_extents;
  Array<PrimExpr> destination_extents;
  for (const Range &range : src_range) {
    TVM_FFI_CHECK(!analyzer->CanProve(range->extent < 0), ValueError)
        << "Atomic source region cannot have negative extents";
    if (!IsOne(range->extent)) {
      source_extents.push_back(range->extent);
    }
  }
  for (const Range &range : dst_range) {
    TVM_FFI_CHECK(!analyzer->CanProve(range->extent < 0), ValueError)
        << "Atomic destination region cannot have negative extents";
    if (!IsOne(range->extent)) {
      destination_extents.push_back(range->extent);
    }
  }
  if (!source_extents.empty()) {
    TVM_FFI_CHECK_EQ(source_extents.size(), destination_extents.size(),
                     ValueError)
        << "Atomic source and destination must have matching non-unit extents";
    for (size_t axis = 0; axis < source_extents.size(); ++axis) {
      TVM_FFI_CHECK(!analyzer->CanProve(source_extents[axis] !=
                                        destination_extents[axis]),
                    ValueError)
          << "Atomic source and destination must have matching non-unit "
             "extents";
    }
  }
  GetMemoryOrder();
  TVM_FFI_CHECK(!UseTMA() || src.has_value(), ValueError)
      << "Atomic TMA operations require a tensor source";
}

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

AtomicMax::AtomicMax(Array<Expr> args, BlockAnnotations annotations) {
  auto node = ffi::make_object<AtomicMaxNode>(ffi::UnsafeInit{});
  node->Initialize(args, std::move(annotations));
  data_ = std::move(node);
}

TileOperator AtomicMaxNode::Clone() const {
  auto op = make_object<AtomicMaxNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = par_op_->Clone().as_or_throw<ParallelOp>();
  }
  return AtomicMax(op);
}

const Op &AtomicMaxNode::GetElemOp() const { return atomic_max_elem_op(); }

// ============================================================================
// AtomicMin Implementation
// ============================================================================

AtomicMin::AtomicMin(Array<Expr> args, BlockAnnotations annotations) {
  auto node = ffi::make_object<AtomicMinNode>(ffi::UnsafeInit{});
  node->Initialize(args, std::move(annotations));
  data_ = std::move(node);
}

TileOperator AtomicMinNode::Clone() const {
  auto op = make_object<AtomicMinNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = par_op_->Clone().as_or_throw<ParallelOp>();
  }
  return AtomicMin(op);
}

const Op &AtomicMinNode::GetElemOp() const { return atomic_min_elem_op(); }

LayoutMap AtomicOpBaseNode::InferLayout(const LayoutInferArgs &layout_args,
                                        InferLevel level) const {
  return ResolveAtomicReduceImpl(layout_args.target)
      .infer_layout(*this, layout_args, level);
}

Stmt AtomicOpBaseNode::Lower(const LowerArgs &lower_args,
                             sym::Analyzer analyzer) const {
  return ResolveAtomicReduceImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

// ============================================================================
// Operator Registration
// ============================================================================

TIR_REGISTER_TL_TILE_OP(AtomicMax, atomicmax, 2)

TIR_REGISTER_TL_TILE_OP(AtomicMin, atomicmin, 2)

TVM_FFI_STATIC_INIT_BLOCK() {
  AtomicMaxNode::RegisterReflection();
  AtomicMinNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
