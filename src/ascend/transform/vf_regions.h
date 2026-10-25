/*!
 * \file tl/ascend/transform/vf_regions.h
 * \brief VF region helpers shared by the Ascend lowering passes.
 *
 * SIMT_VF and SIMD_VF are native regions that survive to codegen.
 * AscendLayoutInference / AscendLowerTileOp visit their bodies without the
 * generic block bookkeeping, and SIMT_VF additionally carries its own lane
 * scope: nested tile operators lower and infer against the region's
 * threadIdx.x lanes instead of the enclosing kernel's launch threads.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_
#define TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_

#include <tvm/tirx/op/region.h>
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
  PrimVar thread_var;
  Range thread_bounds;
};

inline ffi::String VFRegionName(const RegionStmtNode *region) {
  return region->attrs.GetAttr<ffi::String>("name_hint").value_or("");
}

inline bool IsVFRegion(const std::string &name_hint) {
  return name_hint == "SIMT_VF" || name_hint == "SIMD_VF";
}

/*! \brief The region-local scope of a VF block: SIMT_VF returns its lane
 *  scope, SIMD_VF (scalar CCE code) keeps the enclosing scope.
 *
 *  The returned variable is the actual lexical thread parameter.
 */
inline std::optional<std::pair<PrimVar, Range>>
VFRegionScope(const RegionStmtNode *op) {
  using namespace tirx;
  if (VFRegionName(op) != "SIMT_VF") {
    return std::nullopt;
  }
  SeqStmt body = op->body;
  while (body->seq.size() == 1) {
    const auto *region = body->seq[0].as<RegionStmtNode>();
    if (!region || !region->op.same_as(launch_thread_op()))
      break;
    if (region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x") {
      auto variable = region->body_params[0].as_or_throw<PrimVar>();
      auto extent = region->args[1].as_or_throw<PrimExpr>();
      return std::make_pair(
          variable, Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
    }
    body = region->body;
  }
  TVM_FFI_THROW(ValueError)
      << "SIMT_VF must bind a lexical threadIdx.x parameter";
  TVM_FFI_UNREACHABLE();
}

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_VF_REGIONS_H_
