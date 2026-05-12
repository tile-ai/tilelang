/*!
 * \file tl/op/atomic_add.cc
 *
 * Define element-wise operators.
 */

#include "./atomic_add.h"
#include "utils.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "builtin.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<AtomicAddImpl> &AtomicAddImplRegistry() {
  static std::vector<AtomicAddImpl> registry;
  return registry;
}

const AtomicAddImpl &ResolveAtomicAddImpl(Target target) {
  const auto &registry = AtomicAddImplRegistry();
  const AtomicAddImpl *matched_impl = nullptr;
  for (const AtomicAddImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.atomic_add found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_add requires a target-specific implementation, but no "
         "atomic_add implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterAtomicAddImpl(AtomicAddImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  AtomicAddImplRegistry().push_back(impl);
}

/**
 * @brief Construct an AtomicAdd operator from call arguments and annotations.
 *
 * Builds the internal AtomicAddNode, extracts the source and destination
 * regions and their backing Buffers from the first two region-style expressions
 * in `args` (BufferLoad/BufferRegion), and stores them along with their
 * ranges. Annotations are copied directly from the Call node.
 *
 * @param args Call-style PrimExprs where:
 *             - args[0] is the source region call,
 *             - args[1] is the destination region call.
 * @param annotations Map containing optional keys:
 *             - "use_tma": whether to use TMA for memory operations
 *             - "memory_order": memory order for atomic operations
 * Notes:
 * - The constructor checks that args[0] and args[1] are region-compatible.
 * - The constructed node is stored in this->data_.
 */
AtomicAdd::AtomicAdd(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() >= 2)
      << "AtomicAdd expects at least 2 arguments (src, dst), got "
      << args.size();
  ObjectPtr<AtomicAddNode> node = tvm::ffi::make_object<AtomicAddNode>();
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

  // Copy annotations from the Call node
  node->annotations = annotations;
  data_ = std::move(node);
}

/**
 * @brief Create a deep copy of this AtomicAdd node wrapped as a TileOperator.
 *
 * Produces a new AtomicAddNode object copied from this node. If this node has
 * an associated ParallelOp (par_op_), the parallel op is cloned and attached to
 * the new node so the cloned operator preserves parallelization state.
 *
 * @return TileOperator A TileOperator owning the cloned AtomicAddNode.
 */
TileOperator AtomicAddNode::Clone() const {
  auto op = tvm::ffi::make_object<AtomicAddNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return AtomicAdd(op);
}

const Op &AtomicAddNode::GetElemOp() const { return atomic_add_elem_op(); }

LayoutMap AtomicAddNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  return ResolveAtomicAddImpl(T.target).infer_layout(*this, T, level);
}

Stmt AtomicAddNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return ResolveAtomicAddImpl(T.target).lower(*this, T, analyzer);
}

TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { AtomicAddNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
