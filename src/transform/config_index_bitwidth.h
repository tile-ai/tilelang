/*!
 * \file config_index_bitwidth.h
 * \brief Shared signed address legalization at native lowering boundaries.
 */
#ifndef TVM_TL_TRANSFORM_CONFIG_INDEX_BITWIDTH_H_
#define TVM_TL_TRANSFORM_CONFIG_INDEX_BITWIDTH_H_

#include <tvm/ir/transform.h>

namespace tvm::tl::transform {

// Widen only address arithmetic that does not provably fit its current width.
// This does not rewrite scalar ABI types or apply a narrowing policy.
tvm::transform::Pass LegalizeIndexArithmetic(bool force_wide);

} // namespace tvm::tl::transform
#endif // TVM_TL_TRANSFORM_CONFIG_INDEX_BITWIDTH_H_
