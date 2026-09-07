/*!
 * \file layout_cost_model.h
 * \brief Cost models that rank free-mode layout attempts.
 *
 * The inference engine enumerates attempts per candidate root inside a
 * connected component and keeps the cheapest complete layout assignment.
 * What "cheapest" means is a pluggable policy behind LayoutCostModel:
 *
 *  - RegisterCountCostModel (default): total fragment register slots.
 *    Also considers scalar plans at unannotated reducer-update roots,
 *    scored with the same spill/register ordering as native plans.
 *  - IOAwareCostModel (layout RFC, design B2): walks the component's
 *    global-memory-touching statements (fragment<->global copies and
 *    parallel loops with direct global accesses) and charges each one
 *    max(bandwidth bytes, issue-equivalent bytes) under the attempt's
 *    tentative layouts; registers remain the lexicographic tiebreak.
 *    Available through `tl.layout_cost_model="io-aware"` for opt-in use
 *    and A/B comparisons.
 *  - ReductionAwareCostModel (opt-in, CUDA): enumerates reducer vector
 *    widths and orders complete attempts by spills, issue-equivalent
 *    execution cost, and registers. Physical reducer plans are analyzed
 *    with the materializer's own narrow/wide and packed decisions.
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
#include <tvm/tirx/function.h>

#include "../../op/operator.h"

namespace tvm {
namespace tl {

/*! \brief Score of one complete free-mode layout assignment. Known scores
 *  precede unmeasurable attempts, then compare mem, execution, and regs.
 *  Legacy policies leave execution at zero. `mem` includes the estimated
 * local-memory traffic of register-array spills (a thread-dependent
 * register-array index demotes the whole array to local memory), priced in
 * bytes so it competes honestly with the io-aware model's global traffic
 * instead of vetoing it. Models that do not estimate global memory leave the
 * global part at 0, so their ordering is spill bytes, then register count —
 * attempts without spills keep the historical register-count ordering
 * untouched. */
struct AttemptCost {
  int64_t mem{0};
  int64_t execution{0};
  int64_t regs{0};
  bool known{true};
  bool BetterThan(const AttemptCost &other) const {
    if (known != other.known) {
      return known;
    }
    if (mem != other.mem) {
      return mem < other.mem;
    }
    if (execution != other.execution) {
      return execution < other.execution;
    }
    return regs < other.regs;
  }
};

enum class ReducerVectorSearch { kNative, kScalar, kAll };

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

  /*! \brief Width search at unannotated reducer roots. Alternatives cap
   *  only the root's first inference, never a Cartesian width search. */
  virtual ReducerVectorSearch GetReducerVectorSearch() const {
    return ReducerVectorSearch::kNative;
  }

  /*! \brief Instantiate the model selected by `tl.layout_cost_model`
   *  by name ("io-aware", "register-count", or "reduction-aware");
   *  unknown names are a hard error listing the valid values. `target`
   *  feeds the vectorizer's shared width-cap policy (MaxVectorLoadBits);
   *  the legacy model ignores it. */
  static std::unique_ptr<LayoutCostModel>
  Create(const std::string &name, Target target,
         const tirx::PrimFunc &function = tirx::PrimFunc(),
         const std::vector<ffi::ObjectRef> &statements = {});
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_LAYOUT_INFERENCE_LAYOUT_COST_MODEL_H_
