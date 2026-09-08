/*!
 * \file tl/op/simd_vf.cc
 * \brief SimdVF control TileOperator.
 */

#include "simd_vf.h"

namespace tvm {
namespace tl {

using namespace tirx;

void SimdVFOpNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SimdVFOpNode>().def_ro("root", &SimdVFOpNode::root_);
}

TileOperator SimdVFOpNode::Clone() const { return SimdVFOp(root_); }

Stmt SimdVFOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  (void)T;
  return root_;
}

LayoutMap SimdVFOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TVM_FFI_STATIC_INIT_BLOCK() { SimdVFOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
