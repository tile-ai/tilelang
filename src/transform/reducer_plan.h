#ifndef TVM_TL_TRANSFORM_REDUCER_PLAN_H_
#define TVM_TL_TRANSFORM_REDUCER_PLAN_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tvm/tirx/function.h>

#include "../op/reducer.h"

namespace tvm {
namespace tl {

tirx::Stmt MakeReducerUpdateStore(const ReducerUpdateArgs &update,
                                  const tirx::Buffer &target,
                                  ReducerV2OpType op,
                                  const ffi::Optional<tirx::Var> &pack_lane,
                                  bool narrow);

struct ReducerUpdatePlanSite {
  Fragment loop_layout;
  ffi::Array<tirx::Var> loop_vars;
  ffi::Array<PrimExpr> indices;
  PrimExpr value;
  ffi::Array<tirx::Var> serial_vars;
  ffi::Array<PrimExpr> serial_extents;
  tirx::For loop;
  PrimExpr execution_count{1};
};

struct ReducerPlanInfo {
  tirx::Buffer reducer;
  tirx::Buffer dst;
  ReducerV2OpType op;
  Range thread_bounds;
  Fragment storage_layout;
  ffi::Optional<Fragment> packed_layout;
  ffi::Optional<tirx::Var> pack_lane_var;
  std::vector<ReducerUpdatePlanSite> updates;
  std::vector<std::pair<int, int>> steps;
  LayoutMap layout_overrides;
  PrimExpr finalize_count{0};
  int64_t batch{1};
  bool narrow{false};
  bool has_seed{false};
  std::string reason;
};

class ReducerPlanAnalyzer {
public:
  explicit ReducerPlanAnalyzer(const tirx::PrimFunc &function);

  std::optional<std::vector<ReducerPlanInfo>>
  Analyze(const LayoutMap &layouts,
          const ffi::Map<tirx::For, Fragment> &loop_layouts,
          const ffi::Array<tirx::Buffer> &reducers) const;

private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_REDUCER_PLAN_H_
