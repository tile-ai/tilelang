#include "auto_schedule/preprocess_ir.h"
#include "auto_schedule/role_based_scheduler.h"
#include "compat/expr.h"
#include "cuda/op/builtin.h"
#include "transform/common/attr.h"
#include "ws_analysis.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class AutoWarpSpecializationRewriter : public StmtExprMutator {
public:
  AutoWarpSpecializationRewriter(cuda::SchedulerFn scheduler, Target target,
                                 Array<Var> readonly)
      : scheduler_(scheduler), target_(std::move(target)),
        readonly_(std::move(readonly)) {}

private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    int saved_threads = original_threads_;
    if (node->op.same_as(launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      original_threads_ = GetConstInt(node->args[1]).value_or(0);
    RegionStmt region = GetRef<RegionStmt>(node);
    if (IsDeviceMainBlock(node) && original_threads_ > 0 &&
        !node->attrs->dict.count(kWSScheduleKey) &&
        !HasManualWarpSpecialization(node->body)) {
      if (auto body = cuda::PreprocessIR(node->body, target_)) {
        RegionStmt scheduling_region = region;
        auto hints = node->attrs->dict;
        hints.Set("tl.ws_readonly_storage", readonly_);
        scheduling_region.CopyOnWrite()->attrs = DictAttrs(hints);
        if (auto schedule = scheduler_(scheduling_region, *body,
                                       original_threads_, target_)) {
          auto attributes = node->attrs->dict;
          attributes.Set(kWSScheduleKey, *schedule);
          region.CopyOnWrite()->attrs = DictAttrs(attributes);
          region.CopyOnWrite()->body = body.value().as_or_throw<SeqStmt>();
        }
      }
    } else {
      region = StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(region)
                   .as_or_throw<RegionStmt>();
    }
    original_threads_ = saved_threads;
    return region;
  }

  cuda::SchedulerFn scheduler_;
  Target target_;
  Array<Var> readonly_;
  int original_threads_ = 0;
};
} // namespace

tvm::transform::Pass AutoWarpSpecialization() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
    auto scheduler = context->GetConfig<String>(kEnableAutoWarpSpecialization);
    if (!scheduler || !function->body)
      return function;
    TVM_FFI_CHECK(*scheduler == "role_based", ValueError)
        << "Unknown auto-warp-specialization scheduler: " << *scheduler;
    auto target = function->GetAttr<Target>("target");
    TVM_FFI_CHECK(target, ValueError)
        << "AutoWarpSpecialization requires a target";
    auto rewriter = make_object<AutoWarpSpecializationRewriter>(
        cuda::RoleBasedSchedule, *target,
        cuda::ScheduleReadOnlyStorage(function));
    SeqStmt body = *function->body;
    function.CopyOnWrite()->body =
        rewriter->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AutoWarpSpecialization");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.cuda.transform.AutoWarpSpecialization",
                              AutoWarpSpecialization);
}
} // namespace tvm::tl
