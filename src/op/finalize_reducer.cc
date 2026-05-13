/*!
 * \file src/op/finalize_reducer.cc
 *
 * Define finalize_reducer operator.
 */

#include "finalize_reducer.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<FinalizeReducerImpl> &FinalizeReducerImplRegistry() {
  static std::vector<FinalizeReducerImpl> registry;
  return registry;
}

const FinalizeReducerImpl &ResolveFinalizeReducerImpl(Target target) {
  const auto &registry = FinalizeReducerImplRegistry();
  const FinalizeReducerImpl *matched_impl = nullptr;
  for (const FinalizeReducerImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.finalize_reducer found multiple target-specific "
             "implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.finalize_reducer requires a target-specific implementation, but "
         "no finalize_reducer implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterFinalizeReducerImpl(FinalizeReducerImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  FinalizeReducerImplRegistry().push_back(impl);
}

/**
 * @brief Construct a FinalizeReducerOp from TL operator arguments and a buffer
 * map.
 *
 * Extracts the reducer Buffer from `vmap` using the variable referenced by
 * `args[0]` and sets the reduction operation type from the integer code in
 * `args[1]`.
 *
 * @param args TL operator arguments: expects at least two elements where
 *             `args[0]` is an access pointer identifying the reducer variable
 * and `args[1]` is an integer encoding a `ReducerOpType` (e.g., Sum/Max/Min).
 */
FinalizeReducerOp::FinalizeReducerOp(Array<PrimExpr> args,
                                     Map<String, ObjectRef> annotations) {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>();
  auto reducer_access = NormalizeToAccessRegion(args[0], kAccessReadWrite);
  reducer_access.region =
      BufferRegion::FullRegion(reducer_access.region->buffer);
  reducer_access.access_mask = kAccessReadWrite;
  node->reducer = reducer_access.region->buffer;
  node->SetAccessRegions({reducer_access});
  node->op = (ReducerOpType)*as_const_int(args[1]);
  // Read explicit batch size from annotations (0 means auto-detect).
  if (annotations.count("batch")) {
    node->batch = (int)*as_const_int(Downcast<PrimExpr>(annotations["batch"]));
    CHECK_GE(node->batch, 1)
        << "finalize_reducer: batch must be >= 1, got " << node->batch;
  }
  data_ = std::move(node);
}

Stmt FinalizeReducerOpNode::Lower(const LowerArgs &T,
                                  arith::Analyzer *analyzer) const {
  return ResolveFinalizeReducerImpl(T.target).lower(*this, T, analyzer);
}

/**
 * @brief Infer and return the layout mapping for the reducer buffer.
 *
 * Copies the existing layout for the reducer from the provided LayoutInferArgs
 * into a new LayoutMap and returns it. The inference does not modify the
 * layout; it preserves the reducer's current layout.
 *
 * @param T Provides the input layout map from which the reducer's layout is
 * copied.
 * @param level Unused by this operator; present for API compatibility.
 * @return LayoutMap A map that contains the reducer buffer mapped to its
 * original layout.
 */
LayoutMap FinalizeReducerOpNode::InferLayout(const LayoutInferArgs &T,
                                             InferLevel level) const {
  LayoutMap layout_map;
  layout_map.Set(reducer, T.layout_map.Get(reducer).value());
  return layout_map;
}

/**
 * @brief Create a deep copy of this FinalizeReducerOpNode and wrap it as a
 * TileOperator.
 *
 * Constructs a new FinalizeReducerOpNode by copying the current node state and
 * returns a TileOperator that owns the copied node.
 *
 * @return TileOperator A TileOperator that contains a deep copy of this node.
 */
TileOperator FinalizeReducerOpNode::Clone() const {
  auto node = tvm::ffi::make_object<FinalizeReducerOpNode>(*this);
  return TileOperator(node);
}

TIR_REGISTER_TL_TILE_OP(FinalizeReducerOp, finalize_reducer)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { FinalizeReducerOpNode::RegisterReflection(); }
} // namespace tl
} // namespace tvm
