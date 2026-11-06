/*!
 * \file auto_schedule.cc
 * \brief Generic entrypoint for automatic warp-specialization schedulers.
 *
 * The pass config "tl.enable_auto_warp_specialization" names the scheduler to
 * run (see SchedulerRegistry; currently "role_based"); when unset the pass is a
 * no-op. For each eligible kernel — a native launch region with a known
 * threadIdx.x extent, no existing schedule, and no manual warp specialization —
 * the entrypoint gives every schedulable statement a stable "tl.ws_op_id"
 * marker and checks the schedulability contract (preprocess_ir), then asks the
 * scheduler for a typed WSSchedule, which MaterializeWSSchedule then
 * lowers. The kernel body itself only gains the id markers; any kernel
 * preprocessing or the scheduler declines is left byte-for-byte
 * unchanged, with the reason emitted as an on-site warning (auto warp
 * specialization is opt-in, so the user expects it to fire).
 *
 * TODO: verify dependence coverage of USER-PROVIDED schedules with a real
 * dependence analysis (synthesized schedules cover exactly the cross-role
 * accesses they create; MaterializeWSSchedule does not verify dependence
 * coverage).
 */

#include "layout/expr_utils.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <string>
#include <utility>

#include "./auto_schedule/preprocess_ir.h"
#include "./auto_schedule/role_based_scheduler.h"
#include "./ws_analysis.h"
#include "cuda/op/builtin.h"
#include "transform/common/warp_specialize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace tirx::transform;
using namespace cuda;

namespace {

// Available schedulers, by the name passed in
// tl.enable_auto_warp_specialization.
const std::map<std::string, SchedulerFn> &SchedulerRegistry() {
  static const std::map<std::string, SchedulerFn> registry = {
      {"role_based", RoleBasedSchedule},
  };
  return registry;
}

SchedulerFn FindScheduler(const ffi::String &name) {
  const auto &registry = SchedulerRegistry();
  auto it = registry.find(std::string(name));
  if (it == registry.end()) {
    std::string known;
    for (const auto &[known_name, fn] : registry)
      known += (known.empty() ? "" : ", ") + known_name;
    LOG(FATAL) << "unknown auto-warp-specialization scheduler '" << name
               << "'; available: " << known;
  }
  return it->second;
}

tirx::Function AutoWarpSpecializationImpl(tirx::Function func,
                                          SchedulerFn scheduler) {
  return RewriteWSKernels(
      std::move(func),
      [scheduler](WSKernelAnalysis kernel, PrimVar,
                  Target target) -> Optional<WSKernelRewriteResult> {
        auto threads = TryConstInt(kernel.threads);
        if (!threads || *threads == 0 ||
            kernel.annotations.count(kWSScheduleKey) ||
            HasManualWarpSpecialization(kernel.body))
          return std::nullopt;
        auto body = PreprocessIR(kernel.body, target);
        if (!body)
          return std::nullopt;
        auto schedule =
            scheduler(kernel, *body, static_cast<int>(*threads), target);
        if (!schedule)
          return std::nullopt;
        kernel.annotations.Set(kWSScheduleKey, *schedule);
        return WSKernelRewriteResult{RebuildWSKernel(kernel, *body),
                                     std::nullopt};
      });
}

} // namespace

tvm::transform::Pass AutoWarpSpecialization() {
  auto pass_func = [](tirx::Function func, const IRModule &,
                      const PassContext &ctx) {
    auto scheduler_name =
        ctx->GetConfig<ffi::String>(kEnableAutoWarpSpecialization);
    if (!scheduler_name.has_value())
      return func;
    return AutoWarpSpecializationImpl(std::move(func),
                                      FindScheduler(scheduler_name.value()));
  };
  return CreateFunctionPass(pass_func, 0, "tl.AutoWarpSpecialization");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.AutoWarpSpecialization",
                        AutoWarpSpecialization);
}

} // namespace tl
} // namespace tvm
