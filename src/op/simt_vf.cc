/*!
 * \file tl/op/simt_vf.cc
 * \brief SimtVF control TileOperator.
 */

#include "simt_vf.h"

namespace tvm {
namespace tl {

using namespace tirx;

void SimtVFOpNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<SimtVFOpNode>().def_ro("root", &SimtVFOpNode::root_);
}

TileOperator SimtVFOpNode::Clone() const { return SimtVFOp(root_); }

Stmt SimtVFOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  (void)T;
  return root_;
}

LayoutMap SimtVFOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  (void)T;
  (void)level;
  return {};
}

TVM_FFI_STATIC_INIT_BLOCK() { SimtVFOpNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
