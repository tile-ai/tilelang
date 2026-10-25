#include "common/pipeline_async.h"
#include "common/pipeline_replay.h"
#include "common/pipeline_utils.h"
#include "cuda/transform/ws_analysis.h"

#include <tvm/tirx/transform.h>

#include <limits>
#include <numeric>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

bool PipelineGuardInterval(const PrimExpr &condition, const PrimVar &variable,
                           int64_t *begin, int64_t *end) {
  if (const auto *conjunction = condition.as<AndNode>())
    return PipelineGuardInterval(conjunction->a, variable, begin, end) &&
           PipelineGuardInterval(conjunction->b, variable, begin, end);
  auto restrict = [&](const auto *comparison, int relation) {
    if (!comparison)
      return false;
    PrimExpr left = comparison->a, right = comparison->b;
    if (right.same_as(variable)) {
      std::swap(left, right);
      relation = -relation;
    }
    auto bound = GetConstInt(right);
    if (!left.same_as(variable) || !bound ||
        *bound == std::numeric_limits<int64_t>::max())
      return false;
    if (relation >= 0)
      *begin = std::max(*begin, *bound + (relation == 2));
    if (relation <= 0)
      *end = std::min(*end, *bound + (relation != -2));
    return true;
  };
  return restrict(condition.as<LTNode>(), -2) ||
         restrict(condition.as<LENode>(), -1) ||
         restrict(condition.as<EQNode>(), 0) ||
         restrict(condition.as<GENode>(), 1) ||
         restrict(condition.as<GTNode>(), 2);
}

For NarrowPipelineGuard(For loop) {
  auto begin = GetConstInt(loop->begin), end = GetConstInt(loop->end);
  if (!begin || !end || !IsConstInt(loop->step, 1) || *begin >= *end)
    return loop;
  while (loop->body->seq.size() == 1) {
    const auto *conditional = loop->body->seq[0].as<IfNode>();
    if (!conditional || conditional->else_case)
      break;
    int64_t lower = *begin, upper = *end;
    if (!PipelineGuardInterval(conditional->condition, loop->var, &lower,
                               &upper))
      break;
    lower = std::clamp(lower, *begin, *end);
    upper = std::clamp(upper, lower, *end);
    SeqStmt body = conditional->then_case;
    loop.CopyOnWrite()->body = body;
    loop.CopyOnWrite()->begin = MakeConst(loop->begin.ty(), lower);
    loop.CopyOnWrite()->end = MakeConst(loop->end.ty(), upper);
    begin = lower;
    end = upper;
  }
  return loop;
}

bool PipelineHasTransitiveTma(const PipelineBody &body, const Target &target) {
  for (size_t root = 0; root < body.statements.size(); ++root) {
    if (ClassifyStmt(body.statements[root].statement, target) !=
            TileStmtKind::kTmaProducer ||
        PipelineAsyncCandidate(body.statements[root], target))
      continue;
    std::vector<bool> reachable(body.statements.size());
    reachable[root] = true;
    for (size_t consumer = root + 1; consumer < body.statements.size();
         ++consumer) {
      for (size_t producer = root; producer < consumer; ++producer)
        reachable[consumer] = reachable[consumer] ||
                              (reachable[producer] &&
                               PipelineDependency(body.statements[producer],
                                                  body.statements[consumer]));
      if (reachable[consumer] &&
          IsProducer(ClassifyStmt(body.statements[consumer].statement, target)))
        return true;
    }
  }
  return false;
}

class PipelineAsyncOccurrences : public StmtExprVisitor {
public:
  std::vector<const CallNode *> calls;
  bool manual_schedule = false;

private:
  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    calls.push_back(node);
    return StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    manual_schedule |= node->attrs->dict.count("tl.ws_schedule") != 0;
    return StmtExprVisitor::Visit_(node);
  }
};

class ClosedPipelineCPAsync : public StmtExprMutator {
public:
  SeqStmt Run(const SeqStmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &object) {
      if (const auto *loop = object.as<ForNode>())
        Inspect(loop);
    });
    auto occurrences = make_object<PipelineAsyncOccurrences>();
    occurrences->Visit(body);
    bool safe = !controls_.empty() && !occurrences->manual_schedule;
    std::unordered_map<const CallNode *, size_t> counts;
    for (const CallNode *call : occurrences->calls) {
      const auto *operation = call->op.as<OpNode>();
      if (!operation)
        continue;
      if (operation->name == "tl.ptx_cp_async" ||
          operation->name.starts_with("tirx.ptx.cp_async") ||
          operation->name.find("ptx_legacy_cp_async") != std::string::npos ||
          operation->name == "tl.tileop.async_copy")
        safe &= ++counts[call] <= members_[call];
      if (auto tile = ParseOperator(GetRef<Call>(call));
          const auto *copy = tile.as<CopyNode>())
        if (auto flag = copy->annotations.Get("is_async_copy"))
          safe &= GetIntAnnotation(*flag).value_or(0) == 0;
    }
    if (!safe)
      return body;
    return Mutate(body).ValueOrUnchanged(body);
  }

private:
  void Inspect(const ForNode *loop) {
    for (const auto &[key, value] : loop->annotations)
      if (key.starts_with("software_pipeline_") || key == "tl.ws_op_id")
        return;
    if (!GetPipelineNumStages(loop))
      return;
    bool pending = false;
    bool committed = false;
    std::vector<const CallNode *> members;
    std::vector<const StmtNode *> controls;
    for (const Stmt &statement : loop->body->seq) {
      const auto *evaluate = statement.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      const auto *operation = call ? call->op.as<OpNode>() : nullptr;
      if (call && call->op.same_as(ptx_cp_async())) {
        if (committed)
          return;
        pending = true;
        members.push_back(call);
      } else if (operation &&
                 operation->name == "tirx.ptx.cp_async_commit_group") {
        if (!pending || committed)
          return;
        committed = true;
        members.push_back(call);
        controls.push_back(statement.get());
      } else if (operation &&
                 operation->name == "tirx.ptx.cp_async_wait_group") {
        if (!pending || !committed || call->args.empty() ||
            GetConstInt(call->args[0]).value_or(-1) != 0)
          return;
        pending = false;
        committed = false;
        members.push_back(call);
        controls.push_back(statement.get());
      } else if (pending) {
        return;
      }
    }
    if (pending || controls.empty())
      return;
    for (const CallNode *call : members)
      ++members_[call];
    controls_.insert(controls.begin(), controls.end());
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    Array<Stmt> statements;
    for (const Stmt &statement : node->seq)
      if (!controls_.count(statement.get()))
        statements.push_back(
            Mutate(statement, mode).ValueOrUnchanged(statement));
    return SeqStmt(statements, node->loc);
  }

  std::unordered_map<const CallNode *, size_t> members_;
  std::unordered_set<const StmtNode *> controls_;
};

class PipelinePlanner : public StmtExprMutator {
public:
  PipelinePlanner(Target target, PipelineReadOnlySet readonly)
      : target_(std::move(target)), readonly_(std::move(readonly)) {}

private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    MemoryAliases saved = aliases_;
    auto saved_threads = threads_;
    auto saved_constraints = constraints_;
    auto result = StmtExprMutator::Mutate_(node, mode);
    aliases_ = std::move(saved);
    threads_ = std::move(saved_threads);
    constraints_ = std::move(saved_constraints);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    aliases_.Bind(node);
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.as<OpNode>() &&
        call->op.as_or_throw<Op>()->name == "tl.launch_thread_idx") {
      int axis = GetConstInt(call->args[0].as_or_throw<PrimExpr>()).value();
      threads_.Set(axis, node->var.as_or_throw<PrimVar>());
      if (axis < static_cast<int>(thread_extents_.size()))
        constraints_.AddConstr(node->var.as_or_throw<PrimVar>(),
                               Range(0, thread_extents_[axis]));
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto previous = constraints_;
    auto previous_extents = thread_extents_;
    if (node->op.same_as(launch_thread_op()))
      constraints_.AddConstr(node->body_params[0].as_or_throw<PrimVar>(),
                             Range(0, node->args[1].as_or_throw<PrimExpr>()));
    if (auto extents =
            node->attrs.GetAttr<Array<PrimExpr>>("tl.launch_threads")) {
      thread_extents_ = *extents;
      for (size_t axis = 0; axis < extents.value().size(); ++axis)
        if (auto variable = threads_.Get(axis))
          constraints_.AddConstr(*variable, Range(0, extents.value()[axis]));
    }
    auto result = StmtExprMutator::Mutate_(node, mode);
    constraints_ = std::move(previous);
    thread_extents_ = std::move(previous_extents);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(GetRef<For>(node))
                   .as_or_throw<For>();
    auto explicit_stage = loop->annotations.Get("tl_pipeline_stage");
    auto explicit_order = loop->annotations.Get("tl_pipeline_order");
    TVM_FFI_CHECK(explicit_stage.has_value() == explicit_order.has_value(),
                  ValueError)
        << "Explicit pipelines require both tl_pipeline_stage and "
           "tl_pipeline_order";
    if (!explicit_stage && !loop->annotations.count("num_stages"))
      return loop;
    TVM_FFI_CHECK(loop->kind == ForKind::kDefault, ValueError)
        << "Software pipelining requires a serial loop";
    if (!explicit_stage)
      loop = NarrowPipelineGuard(loop);
    auto replay = PipelineReplayMask(loop->body, aliases_, readonly_,
                                     loop->annotations, true);
    PipelineBody body =
        AnalyzePipelineBody(loop->body, aliases_, constraints_, replay);
    if (!explicit_stage && PipelineHasTransitiveTma(body, target_)) {
      auto attributes = loop->annotations;
      attributes.erase("num_stages");
      attributes.erase("tl_pipelined_num_stages");
      loop.CopyOnWrite()->annotations = attributes;
      return loop;
    }
    auto attributes =
        FilterPipelineMetadata(loop->annotations, body, loop->body->seq.size());
    if (std::any_of(replay.begin(), replay.end(),
                    [](int flag) { return flag != 0; }))
      attributes.Set(kPipelineReplayableScalarBinds,
                     PipelineIntegerArray(replay));
    if (auto depth = GetPipelineNumStages(loop.get()))
      attributes.Set("tl_pipelined_num_stages", PrimExpr(*depth));
    attributes.erase("num_stages");
    attributes.erase("tl_pipeline_stage");
    attributes.erase("tl_pipeline_order");
    std::vector<int> stages, order(body.statements.size());
    std::iota(order.begin(), order.end(), 0);
    if (explicit_stage) {
      stages = PipelineIntegerArray(*explicit_stage, "tl_pipeline_stage");
      order = PipelineIntegerArray(*explicit_order, "tl_pipeline_order");
      if (stages.size() == loop->body->seq.size() &&
          stages.size() != body.statements.size()) {
        std::vector<int> filtered_stages, filtered_order;
        TVM_FFI_CHECK(order.size() == stages.size(), ValueError)
            << "Pipeline stage and order arrays must have matching lengths";
        for (size_t index : body.source_indices) {
          filtered_stages.push_back(stages[index]);
          filtered_order.push_back(order[index]);
        }
        stages = std::move(filtered_stages);
        order = std::move(filtered_order);
      }
    } else {
      int depth = *GetPipelineNumStages(loop.get());
      stages.resize(body.statements.size(), depth - 1);
      for (size_t index = 0; index < stages.size(); ++index) {
        bool global_read = false, staging_write = false, staging_read = false;
        for (const MemoryAccess &access : body.statements[index].accesses) {
          bool staging = access.location.scope.find("shared") == 0 ||
                         access.location.scope.find("local") == 0;
          global_read |=
              access.location.scope == "global" && (access.mask & kAccessRead);
          staging_write |= staging && (access.mask & kAccessWrite);
          staging_read |= staging && (access.mask & kAccessRead);
        }
        if (IsProducer(
                ClassifyStmt(body.statements[index].statement, target_)) ||
            (global_read && staging_write && !staging_read))
          stages[index] = 0;
      }
      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t before = 0; before < stages.size(); ++before)
          for (size_t after = before + 1; after < stages.size(); ++after)
            if (stages[before] > stages[after] &&
                PipelineDependency(body.statements[before],
                                   body.statements[after])) {
              stages[before] = stages[after];
              changed = true;
            }
        for (size_t first = 0; first < stages.size(); ++first)
          for (const auto &[storage, mask] : body.statements[first].memory) {
            if (PipelineVersionable(body, storage))
              continue;
            int earliest = stages[first];
            bool written = false;
            for (size_t index = 0; index < stages.size(); ++index) {
              auto access = body.statements[index].memory.find(storage);
              if (access != body.statements[index].memory.end()) {
                earliest = std::min(earliest, stages[index]);
                written |= (access->second & kAccessWrite) != 0;
              }
            }
            if (written)
              for (size_t index = 0; index < stages.size(); ++index)
                if (body.statements[index].memory.count(storage) &&
                    stages[index] != earliest) {
                  stages[index] = earliest;
                  changed = true;
                }
          }
      }
    }
    order = NormalizePipelineOrder(std::move(order));
    ValidatePipelineSchedule(body, stages, order);
    auto asynchronous =
        PlanPipelineAsync(body, stages, order, attributes, target_);
    if (!asynchronous.group_last.empty()) {
      attributes.Set(kPipelineAsyncProducers,
                     PipelineIntegerArray(asynchronous.producers));
      attributes.Set(kPipelineAsyncProducerGroups,
                     PipelineIntegerArray(asynchronous.groups));
      std::set<int> unique_stages(asynchronous.group_stage.begin(),
                                  asynchronous.group_stage.end());
      attributes.Set("software_pipeline_async_stages",
                     PipelineIntegerArray(std::vector<int>(
                         unique_stages.begin(), unique_stages.end())));
    }
    attributes.Set(kPipelineStage, PipelineIntegerArray(stages));
    attributes.Set(kPipelineOrder, PipelineIntegerArray(order));
    loop.CopyOnWrite()->annotations = std::move(attributes);
    return loop;
  }

  Target target_;
  MemoryAliases aliases_;
  Map<int64_t, PrimVar> threads_;
  ConstrSet constraints_;
  Array<PrimExpr> thread_extents_;
  PipelineReadOnlySet readonly_;
};
} // namespace

tvm::transform::Pass NormalizePipelineCPAsync() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    if (function->body && target && TargetCudaHasAsyncCopy(*target))
      function.CopyOnWrite()->body =
          make_object<ClosedPipelineCPAsync>()->Run(*function->body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.NormalizePipelineCPAsync");
}

tvm::transform::Pass PipelinePlanning() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> Optional<tirx::Function> {
    if (function->body) {
      auto target = function->GetAttr<Target>("target");
      TVM_FFI_CHECK(target, ValueError) << "PipelinePlanning requires a target";
      auto planner = make_object<PipelinePlanner>(
          *target, PipelineReadOnlyParameters(function));
      SeqStmt body = *function->body;
      function.CopyOnWrite()->body =
          planner->Mutate(body).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.PipelinePlanning");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef()
      .def("tl.transform.NormalizePipelineCPAsync", NormalizePipelineCPAsync)
      .def("tl.transform.PipelinePlanning", PipelinePlanning);
}
} // namespace tvm::tl
