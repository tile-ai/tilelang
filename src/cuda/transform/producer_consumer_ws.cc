#include "auto_schedule/preprocess_ir.h"
#include "auto_schedule/role_based_scheduler.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "transform/common/attr.h"
#include "ws_analysis.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class ProducerConsumerPlanner : public StmtExprMutator {
public:
  ProducerConsumerPlanner(Target target, Array<Var> readonly)
      : target_(std::move(target)), readonly_(std::move(readonly)) {}
  bool scheduled = false;

private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    int saved_threads = threads_;
    if (node->op.same_as(launch_thread_op()) &&
        node->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      threads_ = GetConstInt(node->args[1]).value_or(0);
    RegionStmt region = GetRef<RegionStmt>(node);
    if (IsDeviceMainBlock(node) && threads_ > 0 &&
        !node->attrs->dict.count(kWSScheduleKey) &&
        !HasManualWarpSpecialization(node->body)) {
      bool candidate = false;
      PostOrderVisit(node->body, [&](const ObjectRef &object) {
        const auto *loop = object.as<ForNode>();
        if (loop && loop->annotations.count("num_stages"))
          PostOrderVisit(loop->body, [&](const ObjectRef &nested) {
            if (auto statement = nested.as<Stmt>())
              candidate |= ClassifyStmt(*statement, target_) ==
                           TileStmtKind::kTmaProducer;
          });
      });
      if (candidate)
        if (auto body = cuda::PreprocessIR(node->body, target_)) {
          RegionStmt scheduling_region = region;
          auto hints = node->attrs->dict;
          hints.Set("tl.ws_readonly_storage", readonly_);
          scheduling_region.CopyOnWrite()->attrs = DictAttrs(hints);
          if (auto schedule = cuda::RoleBasedSchedule(scheduling_region, *body,
                                                      threads_, target_)) {
            auto attributes = node->attrs->dict;
            attributes.Set(kWSScheduleKey, *schedule);
            region.CopyOnWrite()->attrs = DictAttrs(attributes);
            region.CopyOnWrite()->body = body.value().as_or_throw<SeqStmt>();
            scheduled = true;
          }
        }
    } else {
      region = StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(region)
                   .as_or_throw<RegionStmt>();
    }
    threads_ = saved_threads;
    return region;
  }

  Target target_;
  Array<Var> readonly_;
  int threads_ = 0;
};
} // namespace

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    if (!function->body || !target || !TargetHasBulkCopy(*target) ||
        context->GetConfig<bool>(kDisableWarpSpecialized).value_or(false))
      return function;
    auto planner = make_object<ProducerConsumerPlanner>(
        *target, cuda::ScheduleReadOnlyStorage(function));
    SeqStmt body = *function->body;
    SeqStmt rewritten = planner->Mutate(body).ValueOrUnchanged(body);
    if (!planner->scheduled)
      return function;
    function.CopyOnWrite()->body = std::move(rewritten);
    auto materialize =
        ffi::Function::GetGlobal("tl.cuda.native.MaterializeWSSchedule");
    TVM_FFI_CHECK(materialize, RuntimeError)
        << "Import tilelang to register native WS materialization";
    return (*materialize)(function).cast<tirx::Function>();
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.ProducerConsumerWarpSpecialized");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def(
      "tl.cuda.transform.ProducerConsumerWarpSpecialized",
      ProducerConsumerWarpSpecialized);
}
} // namespace tvm::tl
