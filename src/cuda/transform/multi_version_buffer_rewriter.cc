/*!
 * \file multi_version_buffer_rewriter.cc
 * \brief Version native pipeline tensors without changing allocation identity
 * in place.
 */

#include "multi_version_buffer_rewriter.h"
#include "cuda/op/builtin.h"
#include "cuda/transform/ws_analysis.h"
#include "cuda/transform/ws_kernel_analysis.h"
#include "ir/annotations.h"
#include "transform/common/pipeline_access.h"
#include "transform/common/pipeline_tensor.h"
#include "ws_analysis.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {
using TensorSet = std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual>;

bool IsBarrier(const TensorVar &tensor) {
  return tensor.scope().name() == "shared.barrier" ||
         tensor.scope().name() == "shared.cluster_barrier";
}

struct VersionPlan {
  int stages = 1;
  Array<TensorVar> tensors;
  PrimExpr linear_iteration{ffi::UnsafeInit{}};
};

class VersionPlanner : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  VersionPlanner(const PipelineMemoryAnalysis *memory, Target target)
      : memory_(memory), target_(std::move(target)) {}

  std::unordered_map<const ForObj *, VersionPlan> plans;
  std::unordered_map<TensorVar, int, ObjectPtrHash, ObjectPtrEqual> capacities;

private:
  Optional<VisitInterrupt> Visit_(const SeqStmtObj *sequence) final {
    size_t size = visible_.size();
    for (const auto &statement : sequence->seq)
      Visit(statement);
    visible_.erase(visible_.begin() + size, visible_.end());
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const BindObj *bind) final {
    Visit(bind->value);
    Visit(bind->var->ty);
    const auto *call = bind->value.as<CallObj>();
    if (call && call->op.same_as(alloc_tensor_op()))
      if (auto tensor = bind->var.as<TensorVar>();
          tensor && (IsSharedBuffer(*tensor) || IsBarrier(*tensor)))
        visible_.push_back(*tensor);
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const ForObj *loop) final {
    loops_.push_back(loop);
    if (auto stages = GetPipelineNumStages(loop); stages && *stages >= 1) {
      TVM_FFI_CHECK_LE(*stages, std::numeric_limits<int>::max(), ValueError);
      VersionPlan plan;
      plan.stages = static_cast<int>(*stages);
      plan.tensors = Select(loop->body, plan.stages);
      // Count actual iterations from each loop origin. The source PC
      // protocol assumes unit-step loops; reject an unsupported step before
      // constructing mismatched buffer slots and barrier phases.
      plan.linear_iteration = WSPipelineIterationModulo(loops_, plan.stages);
      for (const auto &tensor : plan.tensors) {
        auto [entry, inserted] = capacities.emplace(tensor, plan.stages);
        if (!inserted)
          entry->second = std::max(entry->second, plan.stages);
      }
      plans.emplace(loop, std::move(plan));
    }
    auto result = StmtExprVisitor::Visit_(loop);
    loops_.pop_back();
    return result;
  }

  Array<TensorVar> Select(const SeqStmt &body, int stages) {
    Array<Stmt> statements;
    std::function<void(const Stmt &)> flatten = [&](const Stmt &statement) {
      if (const auto *sequence = statement.as<SeqStmtObj>()) {
        for (const auto &part : sequence->seq)
          flatten(part);
      } else if (const auto *region = statement.as<RegionStmtObj>();
                 region && region->op->name == "tl.attr" &&
                 region->body_params.empty() && region->result_vars.empty()) {
        flatten(region->body);
      } else if (const auto *branch = statement.as<IfObj>()) {
        flatten(branch->then_case);
        if (branch->else_case)
          flatten(*branch->else_case);
      } else {
        statements.push_back(statement);
      }
    };
    flatten(body);
    TensorSet producer_used, consumer_used;
    std::unordered_map<TensorVar, size_t, ObjectPtrHash, ObjectPtrEqual>
        first_write, last_read;
    std::vector<bool> copy_stages;
    for (size_t i = 0; i < statements.size(); ++i) {
      auto access = make_object<PipelineAccessCollector>(memory_, target_);
      access->Visit(statements[i]);
      bool global_read = false, shared_write = false;
      for (const auto &region : access->GetReads())
        global_read |= IsGlobalBuffer(RegionTensor(region));
      for (const auto &region : access->GetWrites())
        shared_write |= IsSharedBuffer(RegionTensor(region));
      bool copy = global_read && shared_write;
      copy_stages.push_back(copy);
      const bool producer = IsProducer(ClassifyStmt(statements[i], target_));
      for (const auto &region : access->GetWrites()) {
        auto tensor = RegionTensor(memory_->CanonicalRegion(region));
        first_write.try_emplace(tensor, i);
        if (producer || copy)
          producer_used.insert(tensor);
      }
      for (const auto &region : access->GetReads()) {
        auto tensor = RegionTensor(memory_->CanonicalRegion(region));
        last_read[tensor] = i;
        if (!producer)
          consumer_used.insert(tensor);
      }
      // Opaque accesses cannot justify leaving potentially shared storage
      // unversioned. Keep every visible shared root in the conservative
      // def/use sets; native address rewriting still preserves its alias.
      if (access->HasUnknownAccess()) {
        for (const auto &tensor : visible_) {
          first_write.try_emplace(tensor, i);
          last_read[tensor] = i;
          producer_used.insert(tensor);
          consumer_used.insert(tensor);
        }
      }
    }
    Array<TensorVar> selected;
    TensorSet seen;
    for (const auto &tensor : visible_) {
      if (!seen.insert(tensor).second)
        continue;
      if (IsBarrier(tensor)) {
        selected.push_back(tensor);
        continue;
      }
      if (stages <= 1)
        continue;
      bool needed = producer_used.count(tensor) && consumer_used.count(tensor);
      auto writer = first_write.find(tensor), reader = last_read.find(tensor);
      needed |= writer != first_write.end() && reader != last_read.end() &&
                writer->second < reader->second && copy_stages[writer->second];
      if (needed)
        selected.push_back(tensor);
    }
    return selected;
  }

  const PipelineMemoryAnalysis *memory_;
  Target target_;
  std::vector<TensorVar> visible_;
  std::vector<const ForObj *> loops_;
};

class VersionRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  VersionRewriter(const VersionPlanner *plans,
                  Map<TensorVar, TensorVar> replacements,
                  const PipelineMemoryAnalysis *memory)
      : plans_(plans), replacements_(std::move(replacements)), memory_(memory) {
  }

private:
  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    For rewritten = StmtExprMutator::Mutate_(loop, mode)
                        .ValueOrUnchanged(GetRef<For>(loop))
                        .as_or_throw<For>();
    auto found = plans_->plans.find(loop);
    if (found == plans_->plans.end() || found->second.tensors.empty())
      return rewritten;
    const auto &plan = found->second;
    Map<TensorVar, TensorVar> tensors, barriers;
    for (const auto &tensor : plan.tensors)
      (IsBarrier(tensor) ? barriers : tensors)
          .Set(tensor, replacements_.at(tensor));
    PrimExpr slot = Call(PrimType::Int(32), mvb_stage_index(),
                         {cast(floormod(plan.linear_iteration, plan.stages),
                               PrimType::Int(32))})
                        .as_or_throw<PrimExpr>();
    PrimExpr parity =
        cast(floormod(floordiv(plan.linear_iteration, plan.stages), 2),
             PrimType::Int(32));
    Stmt body = rewritten->body;
    if (!tensors.empty()) {
      auto mutator = make_object<PipelineTensorRewriter>(
          tensors, PipelineTensorRewriter::Mode::kVersion, slot, std::nullopt,
          std::nullopt, std::nullopt, memory_);
      body = mutator->Rewrite(body);
    }
    if (!barriers.empty()) {
      auto mutator = make_object<PipelineTensorRewriter>(
          barriers, PipelineTensorRewriter::Mode::kBarrier, slot, parity,
          loop->var, loop->begin, memory_);
      body = mutator->Rewrite(body);
    }
    return For(rewritten->var, rewritten->begin, rewritten->end,
               rewritten->step, rewritten->kind, body.as_or_throw<SeqStmt>(),
               rewritten->annotations, rewritten->loc);
  }

  const VersionPlanner *plans_;
  Map<TensorVar, TensorVar> replacements_;
  const PipelineMemoryAnalysis *memory_;
};

// A resized tensor is a ring of pipeline slots, not a replacement for an
// arbitrary value used outside that pipeline. Keep the original diagnosed
// boundary. Pure alias definitions may capture its base before the loop;
// versioned uses shift that captured pointer without reevaluating the view.
class EscapedVersionUseChecker : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  EscapedVersionUseChecker(Map<TensorVar, TensorVar> replacements,
                           const PipelineMemoryAnalysis *memory)
      : replacements_(std::move(replacements)), memory_(memory) {}

private:
  void Check(const Expr &value) const {
    auto root = memory_->Root(value);
    if (root)
      if (auto tensor = root.value().as<TensorVar>())
        TVM_FFI_CHECK(!replacements_.count(*tensor), ValueError)
            << "Versioned tensor access escaped pipeline stage context: "
            << tensor.value()->name;
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *load) final {
    Check(load->source);
    return StmtExprVisitor::Visit_(load);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *store) final {
    Check(store->dest);
    return StmtExprVisitor::Visit_(store);
  }

  Optional<VisitInterrupt> Visit_(const OpaqueExprObj *expr) final {
    if (const auto *region = GetRef<OpaqueExpr>(expr).as<TensorRegionObj>())
      Check(region->source);
    return StmtExprVisitor::Visit_(expr);
  }

  Optional<VisitInterrupt> Visit_(const BindObj *bind) final {
    bool old = alias_definition_;
    alias_definition_ =
        bind->var->ty.as<PtrTypeObj>() || bind->var->ty.as<TensorTypeObj>();
    Visit(bind->value);
    Visit(bind->var->ty);
    alias_definition_ = old;
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (IsAnnotationCall(call))
      return std::nullopt;
    return StmtExprVisitor::Visit_(call);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *region) final {
    if (region->op->name == "tl.attr") {
      // The explicit metadata values are reconciled after the stage bodies.
      return Visit(region->body);
    }
    return StmtExprVisitor::Visit_(region);
  }

  Optional<VisitInterrupt> Visit_(const VarObj *var) final {
    if (!alias_definition_)
      Check(GetRef<Var>(var));
    return StmtExprVisitor::Visit_(var);
  }

  Map<TensorVar, TensorVar> replacements_;
  const PipelineMemoryAnalysis *memory_;
  bool alias_definition_ = false;
};
} // namespace

tirx::Function ApplyMultiVersionBufferRewriter(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  PipelineMemoryAnalysis memory(function);
  auto planner = make_object<VersionPlanner>(
      &memory, function->GetAttr<Target>(tvm::attr::kTarget).value());
  planner->Visit(function->body);
  if (planner->capacities.empty())
    return function;

  Map<TensorVar, TensorVar> replacements;
  for (const auto &[tensor, stages] : planner->capacities)
    replacements.Set(tensor,
                     ExpandPipelineTensor(tensor, stages, IsBarrier(tensor)));
  auto rewriter =
      make_object<VersionRewriter>(planner.get(), replacements, &memory);
  auto body = function->body;
  body = rewriter->Mutate(body).ValueOrUnchanged(body);
  auto checker = make_object<EscapedVersionUseChecker>(replacements, &memory);
  checker->Visit(body);

  // Rebuild allocation result types, external aliases, layout maps and
  // barrier initializer vectors only after each pipeline has selected its
  // own slot expression. This also reconciles sibling loop capacities.
  auto reconcile = make_object<PipelineTensorRewriter>(replacements);
  body = reconcile->Mutate(body).ValueOrUnchanged(body);
  function.CopyOnWrite()->body = body;
  return function;
}
} // namespace tvm::tl
