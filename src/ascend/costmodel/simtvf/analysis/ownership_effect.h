/*!
 * \file ascend/costmodel/simtvf/analysis/ownership_effect.h
 * \brief UB access cost of a legal SIMT ownership candidate.
 *
 * At the supplied storage layout, use LowerParallelLoop to map logical
 * elements to per-lane byte addresses, masks and vector accesses. Respect
 * strides, element offsets, broadcast and lowering-generated tail guards.
 * Estimate machine access widths (packed subword vectors up to b32,
 * native >=32-bit vectors up to b64), then pass each instruction to the shared
 * UB model. These widths are lowering assumptions, not a guarantee about
 * final backend ISA.
 *
 * Independent parallel elements feed a shared entry publishing pipeline;
 * their response paths overlap. Consecutive loads or stores form a batch.
 * Direction changes and TileOp boundaries drain the pipeline conservatively.
 * Each operator's enclosing loops repeat its whole cost, including the drain,
 * instead of adding their requests to an independent batch. Unknown domains
 * give an unknown score. Proven address/mask periods compress service work
 * without merging warp requests. Subsequent compiler optimizations are assumed
 * to realize ideal grouping of independent elements without register limits.
 *
 * Compare only the accumulated latency. Unknown candidates cannot replace a
 * known candidate, an unknown baseline is retained, and numerical ties keep
 * the current candidate. Candidate generation, legality and committing the
 * chosen IR remain the layout-inference pass's responsibility.
 *
 * This is ownership's UB effect, not a complete layout or VF-time model. It
 * excludes storage search, instruction supply, extra address/compute
 * instructions, register pressure and overlap between TileOps.
 */
#ifndef TILELANG_ASCEND_COSTMODEL_SIMTVF_ANALYSIS_OWNERSHIP_EFFECT_H_
#define TILELANG_ASCEND_COSTMODEL_SIMTVF_ANALYSIS_OWNERSHIP_EFFECT_H_

#include "../model/ub_access.h"
#include "op/parallel.h"

namespace tvm::tl::ascend::costmodel::simtvf {

// Effect of a legal ownership candidate at its supplied storage layout.
// Currently covers UB traffic only, not storage selection, address-computation
// instructions or the total layout cost.
struct OwnershipEffect {
  SimtUbAccessCost ub_access;
};

// The input is an already legal candidate; this function does not select
// ownership, modify operators, or write annotations. Unknown access geometry
// returns nullopt. Parallel operators and inferred SIMT copies are evaluated
// over their enclosing loop domains, indexed by operator like thread_bounds.
// The caller checks uniform execution and loop/copy legality. DMA boundaries
// are not part of the SIMT cost component; pipeline overlap is not modeled.
std::optional<OwnershipEffect> AnalyzeOwnershipEffect(
    const std::vector<TileOperator> &operators, const std::vector<int> &members,
    const std::vector<Range> &thread_bounds, const LayoutMap &layouts,
    const std::vector<std::vector<tirx::For>> &outer_loops);

// Compare estimated latency only. Missing estimates and ties retain the
// current candidate. Request geometry is handled by the UB model.
bool IsBetterOwnership(const OwnershipEffect &candidate,
                       const OwnershipEffect &current);

} // namespace tvm::tl::ascend::costmodel::simtvf
#endif // TILELANG_ASCEND_COSTMODEL_SIMTVF_ANALYSIS_OWNERSHIP_EFFECT_H_
