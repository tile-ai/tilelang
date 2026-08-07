/*!
 * \file tl/transform/reducer.cc
 * \brief Verification and materialization passes for deferred reducers.
 */

#include "reducer.h"

#include "../op/deferred_reducer.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "span_utils.h"
#include "support/check.h"

#include <tvm/ir/cast.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using ReducerInfoMap =
    std::unordered_map<Var, ReducerInfo, ObjectPtrHash, ObjectPtrEqual>;
using ReducerBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;

struct ReducerPhysicalPlan {
  // When defined, the destination Fragment layout is also a certificate that
  // every physical destination replica can build its result independently.
  // Otherwise the reducer uses the full-participant collective baseline.
  Optional<Fragment> local_complete_layout;

  bool IsLocalComplete() const { return local_complete_layout.defined(); }
};

using ReducerPhysicalPlanMap =
    std::unordered_map<Var, ReducerPhysicalPlan, ObjectPtrHash, ObjectPtrEqual>;
using ReducerLoopLayoutMap =
    std::unordered_map<For, Fragment, ObjectPtrHash, ObjectPtrEqual>;

struct ReducerPlanningResult {
  ReducerPhysicalPlanMap reducer_plans;
  ReducerLoopLayoutMap loop_layout_overrides;
};

PrimExpr MakeRegionCall(const Buffer &buffer, const Array<PrimExpr> &mins,
                        const Array<PrimExpr> &extents, int access_mask,
                        const Span &span) {
  ICHECK_EQ(mins.size(), extents.size());
  ICHECK_EQ(mins.size(), buffer->shape.size());
  Array<PrimExpr> args = {BufferLoad(buffer, mins), Integer(access_mask)};
  for (const PrimExpr &extent : extents) {
    args.push_back(extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args, {}, span);
}

struct ReducerMetadata {
  ReducerInfoMap info;
  ReducerBufferMap buffers;
};

class ReducerMetadataCollector : public StmtExprVisitor {
public:
  static ReducerMetadata Collect(const PrimFunc &func) {
    ReducerMetadataCollector collector;
    collector(func->body);
    for (const auto &[var, buffer] : collector.metadata_.buffers) {
      ICHECK(collector.metadata_.info.count(var))
          << "Reducer buffer `" << buffer->name
          << "` is missing reducer metadata" << SpanHintSuffix(buffer->span);
    }
    for (const auto &[var, info] : collector.metadata_.info) {
      ICHECK(collector.metadata_.buffers.count(var))
          << "Reducer metadata does not refer to a local.reducer allocation: "
          << var;
      ICHECK(IsBuiltinCommutativeReduceType(info->combine_type));
    }
    return std::move(collector.metadata_);
  }

private:
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      if (buffer.scope() == "local.reducer") {
        ICHECK(!metadata_.buffers.count(buffer->data))
            << "Reducer storage Var is allocated more than once: "
            << buffer->data;
        metadata_.buffers.emplace(buffer->data, buffer);
      }
    }

    if (Optional<Any> annotation = op->annotations.Get(attr::kReducerInfo)) {
      Map<Var, Map<String, Any>> definitions =
          annotation.value().cast<Map<Var, Map<String, Any>>>();
      for (const auto &[var, fields] : definitions) {
        ICHECK(!fields.count("rep"))
            << "Reducer v2 does not support the legacy `replication=` policy";
        Optional<Any> op_field = fields.Get("op");
        ICHECK(op_field.has_value()) << "Reducer metadata is missing `op`";
        std::optional<String> op_name = op_field.value().try_cast<String>();
        ICHECK(op_name.has_value()) << "Reducer metadata `op` must be a string";
        Optional<PrimExpr> seed;
        if (Optional<Any> seed_field = fields.Get("seed")) {
          seed = seed_field.value().cast<PrimExpr>();
        }
        ICHECK(!metadata_.info.count(var))
            << "Reducer metadata is defined more than once for " << var;
        metadata_.info.emplace(var, ReducerInfo(op_name.value(), seed));
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  ReducerMetadata metadata_;
};

enum class ReducerState { kAllocated, kActive, kFinalized };

class ReducerEpochVerifier : public StmtExprVisitor {
public:
  explicit ReducerEpochVerifier(const ReducerMetadata &metadata)
      : metadata_(metadata) {
    for (const auto &[var, _] : metadata.info) {
      states_.emplace(var, ReducerState::kAllocated);
    }
  }

  void Verify(const PrimFunc &func) {
    (*this)(func->body);
    for (const auto &[var, state] : states_) {
      const Buffer &buffer = metadata_.buffers.at(var);
      ICHECK(state == ReducerState::kFinalized)
          << "Reducer `" << buffer->name
          << "` must have exactly one explicit T.reducer_init and one "
             "T.finalize_reducer"
          << SpanHintSuffix(buffer->span);
    }
  }

private:
  void VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    ++nested_control_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --nested_control_depth_;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    ++nested_control_depth_;
    std::function<void()> exit_then = analyzer_.EnterConstraint(op->condition);
    VisitStmt(op->then_case);
    exit_then();
    if (op->else_case.defined()) {
      std::function<void()> exit_else =
          analyzer_.EnterConstraint(Not(op->condition));
      VisitStmt(op->else_case.value());
      exit_else();
    }
    --nested_control_depth_;
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar thread = Downcast<IterVar>(op->node);
      analyzer_.Bind(
          thread->var,
          Range::FromMinExtent(make_zero(thread->var.dtype()), op->value));
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    CheckOrdinaryAccess(op->buffer, "BufferStore", op->span);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    CheckOrdinaryAccess(op->buffer, "BufferLoad", op->span);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get())) {
      ICHECK_EQ(op->args.size(), 1U);
      Var var = ReducerVarFromRegion(op->args[0]);
      RequireTopLevelLifecycleOp("T.reducer_init", op->span);
      ICHECK(states_.at(var) == ReducerState::kAllocated)
          << "T.reducer_init must occur exactly once per reducer allocation"
          << SpanHintSuffix(op->span);
      states_[var] = ReducerState::kActive;
      return;
    }
    if (op->op.same_as(ReducerUpdateOp::Get())) {
      ICHECK_EQ(op->args.size(), 2U);
      AccessRegion access =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      Var var = RequireReducer(access.region->buffer);
      ICHECK(states_.at(var) == ReducerState::kActive)
          << "T.reducer_update must be dominated by T.reducer_init and precede "
             "T.finalize_reducer"
          << SpanHintSuffix(op->span);
      ICHECK_EQ(access.region->region.size(),
                access.region->buffer->shape.size());
      for (size_t i = 0; i < access.region->region.size(); ++i) {
        const Range &range = access.region->region[i];
        ICHECK(is_one(range->extent))
            << "T.reducer_update must name exactly one logical output element"
            << SpanHintSuffix(op->span);
        PrimExpr index = analyzer_.Simplify(range->min);
        ICHECK(analyzer_.CanProve(index >= make_zero(index.dtype())) &&
               analyzer_.CanProve(index < access.region->buffer->shape[i]))
            << "T.reducer_update index " << index
            << " is not provably within [0, " << access.region->buffer->shape[i]
            << ") for reducer dimension " << i << SpanHintSuffix(op->span);
        VisitExpr(range->min);
      }
      ICHECK_LE(SideEffect(op->args[1]), CallEffectKind::kReadState)
          << "T.reducer_update contribution may read state but must not write "
             "state or invoke an opaque effect"
          << SpanHintSuffix(op->span);
      VisitExpr(op->args[1]);
      return;
    }
    if (op->op.same_as(FinalizeReducerOp::Get())) {
      ICHECK_EQ(op->args.size(), 2U);
      Var var = ReducerVarFromRegion(op->args[0]);
      RequireTopLevelLifecycleOp("T.finalize_reducer", op->span);
      ICHECK(states_.at(var) == ReducerState::kActive)
          << "T.finalize_reducer must occur exactly once after initialization"
          << SpanHintSuffix(op->span);
      AccessRegion destination =
          NormalizeToAccessRegion(op->args[1], kAccessWrite);
      ICHECK(!metadata_.info.count(destination.region->buffer->data))
          << "T.finalize_reducer destination must not alias reducer storage"
          << SpanHintSuffix(op->span);
      states_[var] = ReducerState::kFinalized;
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() > 1) {
      if (Optional<Var> var = op->args[1].as<Var>()) {
        CheckOrdinaryVarAccess(var.value(), "access_ptr", op->span);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  Var ReducerVarFromRegion(const PrimExpr &arg) const {
    AccessRegion access = NormalizeToAccessRegion(arg, kAccessReadWrite);
    return RequireReducer(access.region->buffer);
  }

  Var RequireReducer(const Buffer &buffer) const {
    ICHECK(metadata_.info.count(buffer->data))
        << "First-class reducer op expects a local.reducer handle, got buffer `"
        << buffer->name << "` in scope `" << buffer.scope() << "`"
        << SpanHintSuffix(buffer->span);
    return buffer->data;
  }

  void RequireTopLevelLifecycleOp(const char *name, const Span &span) const {
    ICHECK_EQ(nested_control_depth_, 0)
        << name
        << " must be in participant-uniform top-level control flow in reducer "
           "v2"
        << SpanHintSuffix(span);
  }

  void CheckOrdinaryAccess(const Buffer &buffer, const char *kind,
                           const Span &span) const {
    CheckOrdinaryVarAccess(buffer->data, kind, span);
  }

  void CheckOrdinaryVarAccess(const Var &var, const char *kind,
                              const Span &span) const {
    ICHECK(!metadata_.info.count(var))
        << "Reducer v2 forbids ordinary " << kind
        << " access; use T.reducer_init, T.reducer_update, and "
           "T.finalize_reducer"
        << SpanHintSuffix(span);
  }

  const ReducerMetadata &metadata_;
  std::unordered_map<Var, ReducerState, ObjectPtrHash, ObjectPtrEqual> states_;
  arith::Analyzer analyzer_;
  int nested_control_depth_{0};
};

struct ReducerUpdateSite {
  For parallel_root;
  Array<Var> parallel_vars;
  Array<PrimExpr> logical_indices;
  std::vector<std::pair<Var, Range>> loop_domains;
  bool loop_body_safe{false};
  bool supported{true};
};

struct ReducerPlanFacts {
  Optional<Fragment> destination_layout;
  std::vector<ReducerUpdateSite> updates;
  bool supported{true};
};

class LocalCompleteLoopSafetyChecker : public StmtExprVisitor {
public:
  static bool Check(const Stmt &body) {
    LocalCompleteLoopSafetyChecker checker;
    checker(body);
    return checker.safe_;
  }

private:
  // Replacing the inferred loop layout can introduce physical replicas. Keep
  // the first implementation deliberately narrow: loads from shared/global
  // storage are repeatable, while any ordinary store or thread-private load
  // requires a more general effect/ownership proof.
  void VisitStmt_(const BufferStoreNode *op) final {
    safe_ = false;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (!IsGlobalBuffer(op->buffer) && !IsSharedBuffer(op->buffer) &&
        op->buffer.scope() != "local.reducer") {
      safe_ = false;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    bool is_reducer_transport = op->op.same_as(ReducerUpdateOp::Get()) ||
                                op->op.same_as(RegionOp::Get());
    if (!is_reducer_transport &&
        SideEffect(GetRef<Call>(op)) > CallEffectKind::kPure) {
      safe_ = false;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool safe_{true};
};

class ReducerPhysicalPlanner : public arith::IRVisitorWithAnalyzer {
public:
  static ReducerPlanningResult Plan(const PrimFunc &func,
                                    const ReducerMetadata &metadata) {
    ReducerPhysicalPlanner planner(metadata);
    planner(func->body);
    return planner.BuildPlans();
  }

private:
  struct ParallelContext {
    For root;
    Fragment layout;
    Array<Var> vars;
    bool body_safe{false};
    bool supported{true};
    bool parallel_chain_open{true};
  };

  explicit ReducerPhysicalPlanner(const ReducerMetadata &metadata)
      : metadata_(metadata) {
    for (const auto &[var, _] : metadata.info) {
      facts_.emplace(var, ReducerPlanFacts{});
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
    LayoutMap previous_layout_map = current_layout_map_;
    if (Optional<Any> annotation = op->annotations.Get(attr::kLayoutMap)) {
      current_layout_map_ = annotation.value().cast<LayoutMap>();
    }
    arith::IRVisitorWithAnalyzer::VisitStmt_(op);
    current_layout_map_ = std::move(previous_layout_map);
  }

  void VisitStmt_(const ForNode *op) final {
    bool starts_parallel_context =
        op->kind == ForKind::kParallel && !parallel_context_.has_value();
    if (starts_parallel_context) {
      ParallelContext context;
      context.root = GetRef<For>(op);
      context.body_safe = LocalCompleteLoopSafetyChecker::Check(op->body);
      Optional<Any> layout_ref = op->annotations.Get(attr::kParallelLoopLayout);
      if (layout_ref.has_value()) {
        context.layout = layout_ref.value().cast<Fragment>();
      } else {
        context.supported = false;
      }
      parallel_context_ = std::move(context);
    }

    bool previous_parallel_chain_open = false;
    if (parallel_context_.has_value()) {
      previous_parallel_chain_open = parallel_context_->parallel_chain_open;
      if (op->kind == ForKind::kParallel) {
        if (!parallel_context_->parallel_chain_open) {
          parallel_context_->supported = false;
        }
        parallel_context_->vars.push_back(op->loop_var);
        if (!analyzer_.CanProveEqual(op->min, make_zero(op->min.dtype())) ||
            (op->step.defined() &&
             !analyzer_.CanProveEqual(op->step.value(), Integer(1)))) {
          parallel_context_->supported = false;
        }
      } else {
        parallel_context_->parallel_chain_open = false;
      }
    }

    loop_domains_.emplace_back(op->loop_var,
                               Range::FromMinExtent(op->min, op->extent));
    arith::IRVisitorWithAnalyzer::VisitStmt_(op);
    loop_domains_.pop_back();

    if (parallel_context_.has_value()) {
      parallel_context_->parallel_chain_open = previous_parallel_chain_open;
      if (op->kind == ForKind::kParallel) {
        parallel_context_->vars.pop_back();
      }
    }
    if (starts_parallel_context) {
      parallel_context_.reset();
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerUpdateOp::Get())) {
      AccessRegion update =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      Var var = update.region->buffer->data;
      auto facts_it = facts_.find(var);
      ICHECK(facts_it != facts_.end());
      ReducerUpdateSite site;
      site.supported = parallel_context_.has_value() &&
                       parallel_context_->supported &&
                       parallel_context_->layout.defined();
      if (parallel_context_.has_value()) {
        site.parallel_root = parallel_context_->root;
        site.parallel_vars = parallel_context_->vars;
        site.loop_body_safe = parallel_context_->body_safe;
      }
      for (const Range &range : update.region->region) {
        site.logical_indices.push_back(range->min);
      }
      site.loop_domains = loop_domains_;
      facts_it->second.updates.push_back(std::move(site));
    } else if (op->op.same_as(FinalizeReducerOp::Get())) {
      AccessRegion reducer = NormalizeToAccessRegion(op->args[0], kAccessRead);
      AccessRegion destination =
          NormalizeToAccessRegion(op->args[1], kAccessWrite);
      auto facts_it = facts_.find(reducer.region->buffer->data);
      ICHECK(facts_it != facts_.end());
      Optional<Layout> layout =
          current_layout_map_.Get(destination.region->buffer);
      if (!layout.defined()) {
        facts_it->second.supported = false;
      } else if (Optional<Fragment> fragment = layout.value().as<Fragment>()) {
        facts_it->second.destination_layout = fragment.value();
      } else {
        facts_it->second.supported = false;
      }
    }
    arith::IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  struct LocalCompleteCandidate {
    Fragment layout;
    std::vector<For> parallel_roots;
  };

  ReducerPlanningResult BuildPlans() const {
    ReducerPlanningResult result;
    std::unordered_map<Var, LocalCompleteCandidate, ObjectPtrHash,
                       ObjectPtrEqual>
        candidates;
    for (const auto &[var, buffer] : metadata_.buffers) {
      result.reducer_plans.emplace(var, ReducerPhysicalPlan{});
      const ReducerPlanFacts &facts = facts_.at(var);
      std::vector<For> parallel_roots;
      DLOG(INFO) << "[ReducerPhysicalPlanner] reducer=" << buffer->name
                 << " facts_supported=" << facts.supported
                 << " destination_layout=" << facts.destination_layout.defined()
                 << " updates=" << facts.updates.size();
      if (!facts.supported || !facts.destination_layout.defined() ||
          facts.updates.empty() ||
          !CanUseLocalComplete(buffer, facts.destination_layout.value(),
                               facts.updates, &parallel_roots)) {
        continue;
      }
      candidates.emplace(
          var, LocalCompleteCandidate{facts.destination_layout.value(),
                                      std::move(parallel_roots)});
    }

    ReducerLoopLayoutMap requested_layouts;
    std::unordered_set<For, ObjectPtrHash, ObjectPtrEqual> conflicts;
    // One T.Parallel loop can feed multiple reducers. Only rewrite its layout
    // when every selected LocalComplete candidate requests the same physical
    // mapping; a conflict sends all affected reducers back to the baseline.
    for (const auto &[_, candidate] : candidates) {
      for (const For &root : candidate.parallel_roots) {
        auto requested = requested_layouts.find(root);
        if (requested == requested_layouts.end()) {
          requested_layouts.emplace(root, candidate.layout);
        } else if (!LayoutsEqual(requested->second, candidate.layout)) {
          conflicts.insert(root);
        }
      }
    }

    for (const auto &[var, candidate] : candidates) {
      bool has_conflict = false;
      for (const For &root : candidate.parallel_roots) {
        has_conflict |= conflicts.count(root) != 0;
      }
      if (has_conflict) {
        continue;
      }
      result.reducer_plans.at(var).local_complete_layout = candidate.layout;
      for (const For &root : candidate.parallel_roots) {
        result.loop_layout_overrides.emplace(root, candidate.layout);
      }
    }
    return result;
  }

  bool CanUseLocalComplete(const Buffer &logical_buffer,
                           const Fragment &destination_layout,
                           const std::vector<ReducerUpdateSite> &updates,
                           std::vector<For> *parallel_roots) const {
    arith::Analyzer shape_analyzer;
    DLOG(INFO) << "[ReducerPhysicalPlanner] testing LocalComplete for "
               << logical_buffer->name << " with "
               << destination_layout->DebugOutput();
    if (destination_layout->InputDim() != logical_buffer->shape.size()) {
      return false;
    }
    for (size_t i = 0; i < logical_buffer->shape.size(); ++i) {
      if (!shape_analyzer.CanProveEqual(destination_layout->InputShape()[i],
                                        logical_buffer->shape[i])) {
        return false;
      }
    }

    if (destination_layout->ThreadRange().defined()) {
      const int64_t *participant_min = as_const_int(
          shape_analyzer.Simplify(destination_layout->ThreadRange()->min));
      const int64_t *participant_extent = as_const_int(
          shape_analyzer.Simplify(destination_layout->ThreadRange()->extent));
      if (participant_min == nullptr || participant_extent == nullptr ||
          *participant_extent <= 0) {
        return false;
      }
    }
    const int64_t *thread_extent = as_const_int(
        shape_analyzer.Simplify(destination_layout->ThreadExtent()));
    if (thread_extent == nullptr || *thread_extent <= 0) {
      return false;
    }
    const int64_t *replicate_extent = as_const_int(
        shape_analyzer.Simplify(destination_layout->ReplicateExtent()));
    if (replicate_extent == nullptr || *replicate_extent <= 0) {
      return false;
    }
    for (const PrimExpr &extent : destination_layout->OutputShape()) {
      const int64_t *value = as_const_int(shape_analyzer.Simplify(extent));
      if (value == nullptr || *value <= 0) {
        return false;
      }
    }

    for (const ReducerUpdateSite &site : updates) {
      DLOG(INFO) << "[ReducerPhysicalPlanner] update supported="
                 << site.supported << " body_safe=" << site.loop_body_safe
                 << " root=" << site.parallel_root.defined()
                 << " parallel_vars=" << site.parallel_vars
                 << " logical_indices=" << site.logical_indices;
      if (!site.supported || !site.loop_body_safe ||
          !site.parallel_root.defined() ||
          site.parallel_vars.size() != destination_layout->InputDim() ||
          site.logical_indices.size() != destination_layout->InputDim()) {
        return false;
      }

      arith::Analyzer analyzer;
      for (const auto &[loop_var, domain] : site.loop_domains) {
        analyzer.Bind(loop_var, domain);
      }
      for (size_t i = 0; i < site.parallel_vars.size(); ++i) {
        Optional<Range> domain;
        for (const auto &[loop_var, loop_domain] : site.loop_domains) {
          if (loop_var.same_as(site.parallel_vars[i])) {
            domain = loop_domain;
            break;
          }
        }
        if (!domain.defined() ||
            !analyzer.CanProveEqual(domain.value()->extent,
                                    destination_layout->InputShape()[i]) ||
            !analyzer.CanProveEqual(site.logical_indices[i],
                                    site.parallel_vars[i])) {
          return false;
        }
      }
      // This direct identity proof is intentionally the first-version
      // boundary. It covers Parallel(M) plus arbitrary inner serial reduction
      // loops, while Parallel(M, K) -> output[M] falls back to AllReduce.
      bool seen_root = false;
      for (const For &root : *parallel_roots) {
        seen_root |= root.same_as(site.parallel_root);
      }
      if (!seen_root) {
        parallel_roots->push_back(site.parallel_root);
      }
    }
    return true;
  }

  static bool LayoutsEqual(const Fragment &lhs, const Fragment &rhs) {
    if (lhs->InputDim() != rhs->InputDim()) {
      return false;
    }
    arith::Analyzer analyzer;
    for (size_t i = 0; i < lhs->InputDim(); ++i) {
      if (!analyzer.CanProveEqual(lhs->InputShape()[i], rhs->InputShape()[i])) {
        return false;
      }
    }
    return lhs->IsEqual(rhs.get());
  }

  const ReducerMetadata &metadata_;
  std::unordered_map<Var, ReducerPlanFacts, ObjectPtrHash, ObjectPtrEqual>
      facts_;
  LayoutMap current_layout_map_;
  std::optional<ParallelContext> parallel_context_;
  std::vector<std::pair<Var, Range>> loop_domains_;
};

class ReducerMaterializer : public arith::IRMutatorWithAnalyzer {
public:
  ReducerMaterializer(const ReducerMetadata &metadata,
                      const ReducerPhysicalPlanMap &plans,
                      const ReducerLoopLayoutMap &loop_layout_overrides,
                      arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), metadata_(metadata),
        plans_(plans), loop_layout_overrides_(loop_layout_overrides) {
    for (const auto &[var, buffer] : metadata_.buffers) {
      const auto *pointer_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(pointer_type != nullptr)
          << "Reducer buffer data Var must have a PointerType annotation";
      Type local_type = PointerType(pointer_type->element_type, "local");
      Var local_var(buffer->data->name_hint, local_type, buffer->data->span);
      Buffer local_buffer = buffer;
      BufferNode *local_buffer_ptr = local_buffer.CopyOnWrite();
      local_buffer_ptr->data = local_var;
      const ReducerPhysicalPlan &plan = plans_.at(var);
      if (plan.IsLocalComplete()) {
        ICHECK(local_buffer_ptr->strides.empty())
            << "LocalComplete reducer storage requires compact strides";
        local_buffer_ptr->shape =
            plan.local_complete_layout.value()->OutputShape();
      }
      buffer_remap_.emplace(buffer, local_buffer);
    }
  }

  static PrimFunc Rewrite(PrimFunc func, const ReducerMetadata &metadata,
                          const ReducerPlanningResult &planning_result) {
    arith::Analyzer analyzer;
    ReducerMaterializer materializer(metadata, planning_result.reducer_plans,
                                     planning_result.loop_layout_overrides,
                                     &analyzer);
    PrimFuncNode *func_ptr = func.CopyOnWrite();
    func_ptr->body = materializer.VisitStmt(func->body);
    return func;
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block =
        Downcast<SBlock>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    SBlockNode *block_ptr = block.CopyOnWrite();
    for (size_t i = 0; i < block_ptr->alloc_buffers.size(); ++i) {
      auto it = buffer_remap_.find(block_ptr->alloc_buffers[i]);
      if (it != buffer_remap_.end()) {
        block_ptr->alloc_buffers.Set(i, it->second);
      }
    }
    block_ptr->annotations.erase(attr::kReducerInfo);
    return block;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    bool is_parallel = op->kind == ForKind::kParallel;
    bool starts_replicated_parallel = false;
    Optional<Fragment> override_layout;
    if (is_parallel && parallel_depth_ == 0) {
      auto override_it = loop_layout_overrides_.find(GetRef<For>(op));
      Fragment layout;
      if (override_it != loop_layout_overrides_.end()) {
        layout = override_it->second;
        override_layout = layout;
      } else {
        Optional<Any> layout_ref =
            op->annotations.Get(attr::kParallelLoopLayout);
        ICHECK(layout_ref.has_value())
            << "Reducer update is inside a T.Parallel loop without an inferred "
               "loop layout"
            << SpanHintSuffix(op->span);
        layout = layout_ref.value().cast<Fragment>();
      }
      starts_replicated_parallel =
          !analyzer_->CanProveEqual(layout->ReplicateExtent(), Integer(1));
    }
    bool variant_bounds =
        !is_parallel &&
        (IsReplicaVariantValue(op->min) || IsReplicaVariantValue(op->extent) ||
         (op->step.defined() && IsReplicaVariantValue(op->step.value())));
    parallel_depth_ += is_parallel ? 1 : 0;
    replicated_parallel_depth_ += starts_replicated_parallel ? 1 : 0;
    replica_variant_control_depth_ += variant_bounds ? 1 : 0;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    replica_variant_control_depth_ -= variant_bounds ? 1 : 0;
    replicated_parallel_depth_ -= starts_replicated_parallel ? 1 : 0;
    parallel_depth_ -= is_parallel ? 1 : 0;
    if (override_layout.defined()) {
      For result_for = Downcast<For>(result);
      ForNode *result_ptr = result_for.CopyOnWrite();
      result_ptr->annotations.Set(attr::kParallelLoopLayout,
                                  override_layout.value());
      result_ptr->annotations.erase(attr::kParallelLoopPredicate);
      result_ptr->annotations.erase(attr::kParallelLoopRequiresPaddingGuard);
      result = std::move(result_for);
    }
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    bool variant_condition = IsReplicaVariantValue(op->condition);
    replica_variant_control_depth_ += variant_condition ? 1 : 0;
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    replica_variant_control_depth_ -= variant_condition ? 1 : 0;
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool is_physical_thread_scope = false;
    Var physical_thread;
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar thread = Downcast<IterVar>(op->node);
      std::string thread_tag = thread->thread_tag;
      const int64_t *extent = as_const_int(op->value);
      if (thread_tag.rfind("threadIdx.", 0) == 0 &&
          (extent == nullptr || *extent > 1)) {
        physical_thread = thread->var;
        physical_thread_vars_.insert(physical_thread);
        is_physical_thread_scope = true;
      }
    }
    Stmt result = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    if (is_physical_thread_scope) {
      physical_thread_vars_.erase(physical_thread);
    }
    return result;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load =
        Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
    auto it = buffer_remap_.find(op->buffer);
    if (it != buffer_remap_.end()) {
      load.CopyOnWrite()->buffer = it->second;
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store =
        Downcast<BufferStore>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    auto it = buffer_remap_.find(op->buffer);
    if (it != buffer_remap_.end()) {
      store.CopyOnWrite()->buffer = it->second;
    }
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    bool is_init = op->op.same_as(ReducerInitOp::Get());
    bool is_update = op->op.same_as(ReducerUpdateOp::Get());
    bool is_finalize = op->op.same_as(FinalizeReducerOp::Get());
    Optional<ReducerInfo> info;
    std::optional<Var> reducer_var;
    AccessRegion reducer_access;
    Array<PrimExpr> logical_update_indices;
    if (is_init || is_update || is_finalize) {
      reducer_access = NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      auto it = metadata_.info.find(reducer_access.region->buffer->data);
      ICHECK(it != metadata_.info.end());
      info = it->second;
      reducer_var = reducer_access.region->buffer->data;
      if (is_update) {
        for (const Range &range : reducer_access.region->region) {
          logical_update_indices.push_back(range->min);
        }
      }
    }
    if (is_update && replicated_parallel_depth_ > 0) {
      AccessRegion update_access =
          NormalizeToAccessRegion(op->args[0], kAccessReadWrite);
      bool variant_index = false;
      for (const Range &range : update_access.region->region) {
        variant_index |= IsReplicaVariantValue(range->min);
      }
      ICHECK_EQ(replica_variant_control_depth_, 0)
          << "T.reducer_update is controlled by a replica-dependent condition "
             "or loop inside a replicated T.Parallel layout"
          << SpanHintSuffix(op->span);
      ICHECK(!variant_index && !IsReplicaVariantValue(op->args[1]))
          << "T.reducer_update indices and contribution must be invariant "
             "across physical replicas; use logical T.Parallel indices and "
             "fragment/global/shared values instead of threadIdx or ordinary "
             "thread-private local values"
          << SpanHintSuffix(op->span);
    }

    Call call = Downcast<Call>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
    if (!info.defined()) {
      return call;
    }
    CallNode *call_ptr = call.CopyOnWrite();
    call_ptr->annotations.Set(
        attr::kReducerType,
        StringImm(std::string(ReduceTypeName(info.value()->combine_type))));
    if (is_finalize && info.value()->seed.defined()) {
      call_ptr->annotations.Set(attr::kReducerSeed, info.value()->seed.value());
    }
    const ReducerPhysicalPlan &plan = plans_.at(reducer_var.value());
    if (plan.IsLocalComplete()) {
      call_ptr->annotations.Set(attr::kReducerLocalCompleteLayout,
                                plan.local_complete_layout.value());
      // Keep the post-materialization IR self-consistent: reducer call regions
      // name the compact physical buffer, while update logical indices travel
      // separately as planned metadata until ReducerUpdateOp lowers them.
      Buffer logical_buffer = reducer_access.region->buffer;
      auto physical_it = buffer_remap_.find(logical_buffer);
      ICHECK(physical_it != buffer_remap_.end());
      const Buffer &physical_buffer = physical_it->second;
      Array<PrimExpr> physical_mins;
      Array<PrimExpr> physical_extents;
      if (is_update) {
        physical_mins =
            plan.local_complete_layout.value()->Forward(logical_update_indices);
        for (const PrimExpr &index : physical_mins) {
          physical_extents.push_back(make_const(index.dtype(), 1));
        }
        call_ptr->annotations.Set(attr::kReducerLogicalIndices,
                                  logical_update_indices);
      } else {
        for (const PrimExpr &extent : physical_buffer->shape) {
          physical_mins.push_back(make_zero(extent.dtype()));
          physical_extents.push_back(extent);
        }
      }
      call_ptr->args.Set(
          0, MakeRegionCall(physical_buffer, physical_mins, physical_extents,
                            reducer_access.access_mask, op->args[0]->span));
    }
    if (is_update) {
      call_ptr->annotations.Set(
          attr::kReducerParallelOnce,
          Bool(parallel_depth_ > 0 && !plan.IsLocalComplete()));
    }
    return call;
  }

  const ReducerMetadata &metadata_;
  const ReducerPhysicalPlanMap &plans_;
  const ReducerLoopLayoutMap &loop_layout_overrides_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_remap_;
  bool UsesPhysicalThreadVar(const PrimExpr &expr) const {
    return UsesVar(expr, [&](const VarNode *node) {
      return physical_thread_vars_.count(GetRef<Var>(node)) != 0;
    });
  }

  bool IsReplicaVariantValue(const PrimExpr &expr) const {
    if (UsesPhysicalThreadVar(expr)) {
      return true;
    }
    bool has_thread_private_load = false;
    PostOrderVisit(expr, [&](const ObjectRef &object) {
      if (const auto *load = object.as<BufferLoadNode>()) {
        has_thread_private_load |= IsLocalBuffer(load->buffer, true);
      }
    });
    return has_thread_private_load;
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> physical_thread_vars_;
  int parallel_depth_{0};
  int replicated_parallel_depth_{0};
  int replica_variant_control_depth_{0};
};

class ReducerLoweredVerifier : public StmtExprVisitor {
private:
  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      ICHECK_NE(buffer.scope(), "local.reducer")
          << "local.reducer allocation reached backend lowering: " << buffer;
    }
    ICHECK(!op->annotations.count(attr::kReducerInfo))
        << "Reducer metadata reached backend lowering";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK(op->attr_key != attr::kParallelMultiplicity &&
           op->attr_key != attr::kParallelPartitionRequired)
        << "Unconsumed reducer parallel-effect marker reached backend lowering";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(!op->op.same_as(ReducerInitOp::Get()) &&
           !op->op.same_as(ReducerUpdateOp::Get()) &&
           !op->op.same_as(FinalizeReducerOp::Get()))
        << "First-class reducer op reached backend lowering: "
        << GetRef<Call>(op);
    StmtExprVisitor::VisitExpr_(op);
  }
};

} // namespace

tvm::transform::Pass VerifyReducerEpochs() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = ReducerMetadataCollector::Collect(func);
    ReducerEpochVerifier(metadata).Verify(func);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyReducerEpochs", {});
}

tvm::transform::Pass PlanAndMaterializeReducers() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = ReducerMetadataCollector::Collect(func);
    if (metadata.info.empty()) {
      return func;
    }
    // Planning is the final semantic boundary before physical storage is
    // introduced. Re-verify here because earlier pipeline transforms (for
    // example warp specialization or software pipelining) may have changed the
    // execution scope since the frontend verification pass.
    ReducerEpochVerifier(metadata).Verify(func);
    ReducerPlanningResult planning_result =
        ReducerPhysicalPlanner::Plan(func, metadata);
    return ReducerMaterializer::Rewrite(std::move(func), metadata,
                                        planning_result);
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.PlanAndMaterializeReducers", {});
}

tvm::transform::Pass VerifyReducerLowered() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerLoweredVerifier()(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.VerifyReducerLowered", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.transform.VerifyReducerEpochs", VerifyReducerEpochs)
      .def("tl.transform.PlanAndMaterializeReducers",
           PlanAndMaterializeReducers)
      .def("tl.transform.VerifyReducerLowered", VerifyReducerLowered);
}

} // namespace tl
} // namespace tvm
