/*!
 * \file tl/op/atomic_add.cc
 *
 * Define element-wise operators.
 */

#include "./atomic_add.h"
#include "support/check.h"
#include "utils.h"

#include "builtin.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

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
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_add requires a target-specific implementation, but no "
         "atomic_add implementation is registered for "
      << target->str();
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
 * @param annotations ffi::Map containing optional keys:
 *             - "use_tma": whether to use TMA for memory operations
 *             - "memory_order": memory order for atomic operations
 * Notes:
 * - The constructor checks that args[0] and args[1] are region-compatible.
 * - The constructed node is stored in this->data_.
 */
AtomicAdd::AtomicAdd(ffi::Array<Expr> args, BlockAnnotations annotations) {
  auto node = ffi::make_object<AtomicAddNode>(ffi::UnsafeInit{});
  node->Initialize(args, std::move(annotations));
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
    op->par_op_ = par_op_->Clone().as_or_throw<ParallelOp>();
  }
  return AtomicAdd(op);
}

const Op &AtomicAddNode::GetElemOp() const { return atomic_add_elem_op(); }

LayoutMap AtomicAddNode::InferLayout(const LayoutInferArgs &layout_args,
                                     InferLevel level) const {
  return ResolveAtomicAddImpl(layout_args.target)
      .infer_layout(*this, layout_args, level);
}

Stmt AtomicAddNode::Lower(const LowerArgs &lower_args,
                          sym::Analyzer analyzer) const {
  return ResolveAtomicAddImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

TIR_REGISTER_TL_TILE_OP(AtomicAdd, atomicadd, 2)

TVM_FFI_STATIC_INIT_BLOCK() { AtomicAddNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
