/*!
 * \file layout_cost_model.h
 * \brief Cost models that rank free-mode layout attempts.
 *
 * The inference engine enumerates one attempt per candidate root inside a
 * connected component and keeps the cheapest complete layout assignment.
 * What "cheapest" means is a pluggable policy behind LayoutCostModel:
 *
 *  - RegisterCountCostModel (legacy): total fragment register slots,
 *    nothing else. Available through
 *    `tl.layout_cost_model="register-count"` for A/B comparisons.
 *  - IOAwareCostModel (layout RFC, design B2): walks the component's
 *    global-memory-touching statements (fragment<->global copies and
 *    parallel loops with direct global accesses) and charges each one
 *    max(bandwidth bytes, issue-equivalent bytes) under the attempt's
 *    tentative layouts; registers remain the lexicographic tiebreak. This
 *    is the default policy.
 *
 * Concrete models live in the .cc; callers go through Create().
 */

#ifndef TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
#define TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tvm/target/target.h>

#include "../../op/operator.h"

namespace tvm {
namespace tl {

/*! \brief Score of one complete free-mode layout assignment. Compared
 *  lexicographically: estimated memory cost first, total register count as
 *  the tiebreak. Models that do not estimate memory leave `mem` at 0, so
 *  their ordering degenerates to the register count. */
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

/*! \brief Policy interface: rank one attempt of a component.
 *
 *  `members` indexes the component's operators inside `infer_list` (which
 *  carries the attempt's solved state, e.g. loop layouts), and
 *  `tmp_layout_map` holds the attempt's tentative buffer layouts. */
class LayoutCostModel {
public:
  virtual ~LayoutCostModel() = default;

  virtual AttemptCost Score(const std::vector<int> &members,
                            const std::vector<TileOperator> &infer_list,
                            const LayoutMap &tmp_layout_map) const = 0;

  /*! \brief Model name for diagnostics. */
  virtual const char *Name() const = 0;

  /*! \brief Instantiate the model selected by `tl.layout_cost_model`
   *  by name ("io-aware" or "register-count" — each model's Name());
   *  unknown names are a hard error listing the valid values. `target`
   *  feeds the vectorizer's shared width-cap policy (MaxVectorLoadBits);
   *  the legacy model ignores it. */
  static std::unique_ptr<LayoutCostModel> Create(const std::string &name,
                                                 Target target);
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
