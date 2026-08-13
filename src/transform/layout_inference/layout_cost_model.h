/*!
 * \file layout_cost_model.h
 * \brief Score free-mode layout attempts by estimated memory access cost
 *        (layout RFC, design B2).
 *
 * The scoring walks a connected component's global-memory-touching
 * statements — fragment<->global copies and parallel loops with direct
 * global accesses — and charges each one max(bandwidth bytes, issue-
 * equivalent bytes) under the attempt's tentative layouts. Registers stay
 * as the lexicographic tiebreak, and are the entire score when the cost
 * model is disabled (reproducing the legacy register-count ordering).
 */

#ifndef TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
#define TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_

#include <cstdint>
#include <vector>

#include "../../op/operator.h"

namespace tvm {
namespace tl {

/*! \brief Score of one complete free-mode layout assignment. Compared
 *  lexicographically: estimated memory cost first, total register count as
 *  the tiebreak. With the cost model disabled `mem` stays 0 for every
 *  attempt and the ordering degenerates to the legacy register count. */
struct AttemptCost {
  int64_t mem{0};
  int64_t regs{0};
  bool BetterThan(const AttemptCost &other) const {
    if (mem != other.mem) {
      return mem < other.mem;
    }
    return regs < other.regs;
  }
};

/*! \brief Score one attempt: `members` indexes the component's operators
 *  inside `infer_list` (carrying the attempt's solved state, e.g. loop
 *  layouts), and `tmp_layout_map` holds the attempt's tentative buffer
 *  layouts. Statements outside the model are charged a conservative worst
 *  case — an attempt must never profit from opacity. */
AttemptCost ComputeAttemptCost(const std::vector<int> &members,
                               const std::vector<TileOperator> &infer_list,
                               const LayoutMap &tmp_layout_map,
                               bool cost_model_enabled);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
