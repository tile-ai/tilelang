/*!
 * \file tl/ascend/op/simt_vf.cc
 * \brief SimtVF control TileOperator.
 */

#include "simt_vf.h"

#include "op/region_op.h"

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

namespace {

// Region-local execution scope of a SIMT_VF block: tile operators inside the
// region lower against the region's own threadIdx.x lanes instead of the
// enclosing kernel's launch threads. The synthetic Var is replaced by the
// real bound IterVar when LowerTileOp visits the region's leading
// thread_extent AttrStmt.
std::optional<std::pair<IterVar, Range>>
SimtVFRegionScope(const SBlockNode *op) {
  const auto *attr = op->body.as<AttrStmtNode>();
  ICHECK(attr && attr->attr_key == tirx::attr::thread_extent)
      << "SIMT_VF block body must start with thread_extent AttrStmt";
  const auto *iv = attr->node.as<IterVarNode>();
  ICHECK(iv && iv->thread_tag == "threadIdx.x")
      << "SIMT_VF block body must bind threadIdx.x first";
  PrimExpr threads = attr->value;
  DataType dtype = threads.dtype();
  IterVar active_thread_var = IterVar(
      Range::FromMinExtent(make_zero(dtype), threads), Var("simtvf_tx", dtype),
      IterVarType::kThreadIndex, "threadIdx.x");
  return std::make_pair(active_thread_var, active_thread_var->dom);
}

TileOperator MakeSimtVFOp(const SBlock &block) { return SimtVFOp(block); }

bool RegisterSimtVFRegionOp() {
  RegisterRegionOpImpl({"SIMT_VF", MakeSimtVFOp, SimtVFRegionScope});
  return true;
}

const bool simt_vf_region_op_registered = RegisterSimtVFRegionOp();

} // namespace

} // namespace tl
} // namespace tvm
