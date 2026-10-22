/*!
 * \file auto_schedule.cc
 * \brief Generic entrypoint for automatic warp-specialization schedulers.
 *
 * The pass config "tl.enable_auto_warp_specialization" names the scheduler to
 * run (see SchedulerRegistry; currently "role_based"); when unset the pass is a
 * no-op. For each eligible kernel — a tilelang_root block with a known
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

class AutoWarpSpecializationRewriter : public StmtMutator {
public:
  AutoWarpSpecializationRewriter(SchedulerFn scheduler, Target target)
      : scheduler_(scheduler), target_(std::move(target)) {}

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    // Role ranges are expressed relative to the kernel's original worker
    // count, so carry the enclosing thread extent while visiting its root.
    if (op->attr_key == tirx::attr::thread_extent) {
      const auto *iv = op->node.as<IterVarNode>();
      if (iv && iv->thread_tag == "threadIdx.x") {
        int old_threads = original_threads_;
        original_threads_ = 0;
        if (const auto *value = op->value.as<IntImmNode>())
          original_threads_ = static_cast<int>(value->value);
        Stmt result = StmtMutator::VisitStmt_(op);
        original_threads_ = old_threads;
        return result;
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    // Existing schedules and manually specialized scopes are explicit
    // user decisions and take precedence over automatic planning.
    if (op->name_hint != "tilelang_root" || original_threads_ == 0 ||
        op->annotations.count(kWSScheduleKey) ||
        HasManualWarpSpecialization(op->body)) {
      return StmtMutator::VisitStmt_(op);
    }

    ffi::Optional<Stmt> body = PreprocessIR(op->body, target_);
    if (!body.defined())
      return GetRef<Stmt>(op);
    ffi::Optional<WSSchedule> schedule = scheduler_(
        GetRef<SBlock>(op), body.value(), original_threads_, target_);
    if (!schedule.defined())
      return GetRef<Stmt>(op);

    auto annotations = op->annotations;
    annotations.Set(kWSScheduleKey, schedule.value());
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                  std::move(body).value(), op->init, op->alloc_buffers,
                  op->match_buffers, annotations, op->span);
  }

  SchedulerFn scheduler_;
  Target target_;
  int original_threads_{0};
};

PrimFunc AutoWarpSpecializationImpl(PrimFunc func, SchedulerFn scheduler) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined())
      << "AutoWarpSpecialization: PrimFunc has no bound target "
         "(BindTarget must run before AutoWarpSpecialization)";

  AutoWarpSpecializationRewriter rewriter(scheduler, target.value());
  Stmt body = rewriter(func->body);
  if (body.same_as(func->body))
    return func;
  func.CopyOnWrite()->body = std::move(body);
  return func;
}

} // namespace

tvm::transform::Pass AutoWarpSpecialization() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &ctx) {
    auto scheduler_name = ctx->GetConfig(kEnableAutoWarpSpecialization,
                                         ffi::Optional<ffi::String>());
    if (!scheduler_name.has_value())
      return func;
    return AutoWarpSpecializationImpl(std::move(func),
                                      FindScheduler(scheduler_name.value()));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AutoWarpSpecialization", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cuda.transform.AutoWarpSpecialization",
                        AutoWarpSpecialization);
}

} // namespace tl
} // namespace tvm
