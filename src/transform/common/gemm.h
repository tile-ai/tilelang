#ifndef TVM_TL_TRANSFORM_COMMON_GEMM_H_
#define TVM_TL_TRANSFORM_COMMON_GEMM_H_

#include "../../op/gemm.h"
#include "../../op/gemm_py.h"
#include <variant>

namespace tvm {
namespace tl {

using namespace tir;

using GemmNodeVariant = std::variant<const GemmNode *, const GemmPyNode *>;

/*!
 * \brief Try to parse a CallNode as a GemmNode or GemmPyNode.
 *
 * This utility function attempts to interpret the provided CallNode as either a
 * GemmNode (native C++) or a GemmPyNode (Python-side). It returns a
 * std::optional containing a variant pointer to the respective type if
 * successful, or std::nullopt otherwise.
 *
 * \param call The CallNode to be analyzed.
 * \return std::optional<GemmNodeVariant> A variant holding the matching GEMM
 * node type, or std::nullopt if not a GEMM.
 */
inline std::optional<GemmNodeVariant> TryParseGemmNode(const CallNode &call) {
  TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<Call>(&call));
  if (auto gemm = tile_op.as<GemmNode>()) {
    return gemm;
  } else if (auto gemm_py = tile_op.as<GemmPyNode>()) {
    return gemm_py;
  }
  return std::nullopt;
}

} // namespace tl
} // namespace tvm
#endif // TVM_TL_TRANSFORM_COMMON_GEMM_H_
