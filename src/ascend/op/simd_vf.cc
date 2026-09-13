/*!
 * \file tl/ascend/op/simd_vf.cc
 * \brief SimdVF control TileOperator.
 */

#include "simd_vf.h"

#include "op/region_op.h"

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

namespace {

// SimdVF is scalar CCE code with no thread dimension: no region scope, so
// nested operators keep the enclosing execution scope.
TileOperator MakeSimdVFOp(const SBlock &block) { return SimdVFOp(block); }

bool RegisterSimdVFRegionOp() {
  RegisterRegionOpImpl({"SIMD_VF", MakeSimdVFOp, nullptr});
  return true;
}

const bool simd_vf_region_op_registered = RegisterSimdVFRegionOp();

} // namespace

} // namespace tl
} // namespace tvm
