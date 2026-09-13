/*!
 * \file tl/op/region_op.h
 * \brief Registry of region operators keyed by opaque SBlock name.
 *
 * A region operator wraps a named opaque SBlock region (e.g. Ascend's
 * "SIMT_VF") that shared passes treat as a unit: LayoutInference wraps the
 * block into the operator for its inference worklist, and both
 * LayoutInference and LowerTileOp push the region's execution scope while
 * visiting its body. Backends register their region operators here,
 * mirroring the CopyImpl / GemmImpl pattern, so the shared passes carry no
 * backend-specific dispatch.
 */

#ifndef TVM_TL_OP_REGION_OP_H_
#define TVM_TL_OP_REGION_OP_H_

#include <tvm/tirx/stmt.h>

#include <optional>
#include <string>
#include <utility>

#include "./operator.h"

namespace tvm {
namespace tl {

struct RegionOpImpl {
  const char *block_name;
  /*! \brief Wrap the block into the operator representing the region. */
  TileOperator (*make)(const tirx::SBlock &block);
  /*!
   * \brief Region-local execution scope pushed while the region body is
   * visited: tile operators inside lower and infer against the returned
   * {thread var, bounds} instead of the enclosing kernel scope. nullptr (or
   * a nullopt result) keeps the enclosing scope.
   */
  std::optional<std::pair<tirx::IterVar, Range>> (*enter_scope)(
      const tirx::SBlockNode *block);
};

void RegisterRegionOpImpl(RegionOpImpl impl);

/*! \brief Returns nullptr when no impl is registered for \p block_name. */
const RegionOpImpl *LookupRegionOpImpl(const std::string &block_name);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REGION_OP_H_
