/*!
 * \file tl/ascend/transform/vf_regions.h
 * \brief VF region helpers shared by the Ascend lowering passes.
 *
 * SIMT_VF and SIMD_VF are opaque SBlock regions that survive to codegen.
 * AscendLayoutInference / AscendLowerTileOp visit their bodies without the
 * generic block bookkeeping, and SIMT_VF additionally carries its own lane
 * scope: nested tile operators lower and infer against the region's
 * threadIdx.x lanes instead of the enclosing kernel's launch threads.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_
#define TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_

#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <utility>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace ascend {

/*! \brief Region-local execution scope; while one is active, tile operators
 *  use the innermost region's thread var and bounds instead of the kernel's.
 */
struct RegionThreadScope {
  tirx::IterVar thread_var;
  Range thread_bounds;
};

inline bool IsVFRegion(const std::string &name_hint) {
  return name_hint == "SIMT_VF" || name_hint == "SIMD_VF";
}

/*! \brief The region-local scope of a VF block: SIMT_VF returns its lane
 *  scope, SIMD_VF (scalar CCE code) keeps the enclosing scope.
 *
 *  The synthetic Var is replaced by the real bound IterVar when the visiting
 *  pass reaches the region's leading thread_extent AttrStmt.
 */
inline std::optional<std::pair<tirx::IterVar, Range>>
VFRegionScope(const tirx::SBlockNode *op) {
  using namespace tirx;
  if (op->name_hint != "SIMT_VF") {
    return std::nullopt;
  }
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

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_
