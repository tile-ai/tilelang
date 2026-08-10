/*!
 * \file tl/transform/reducer/reducer.cc
 * \brief Planning and materialization passes for deferred reducers.
 */

#include "reducer.h"
#include "reducer_metadata.h"

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "layout/utils.h"
#include "op/deferred_reducer.h"
#include "op/reduce_plan.h"
#include "op/region.h"
#include "op/utils.h"
#include "span_utils.h"
#include "support/check.h"

#include <tvm/ir/cast.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

struct ReducerPartialGroupPlan {
  bool canonical{false};
  Optional<Fragment> partial_layout;
  Optional<Fragment> source_loop_layout;
  std::vector<reduction::ThreadReduceStep> thread_steps;
  std::vector<Call> update_sites;
};

struct ReducerEpochPhysicalPlan {
  std::vector<ReducerPartialGroupPlan> groups;
  std::unordered_map<Call, size_t, ObjectPtrHash, ObjectPtrEqual>
      update_to_group;
};

using ReducerEpochPlanMap = std::unordered_map<Var, ReducerEpochPhysicalPlan,
                                               ObjectPtrHash, ObjectPtrEqual>;
using ReducerLoopLayoutMap =
    std::unordered_map<For, Fragment, ObjectPtrHash, ObjectPtrEqual>;

struct ReducerPlanningResult {
  ReducerEpochPlanMap reducer_plans;
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

struct ReducerUpdateSite {
  Call call;
  For parallel_root;
  Fragment loop_layout;
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
      site.call = GetRef<Call>(op);
      site.supported = parallel_context_.has_value() &&
                       parallel_context_->supported &&
                       parallel_context_->layout.defined();
      if (parallel_context_.has_value()) {
        site.parallel_root = parallel_context_->root;
        site.loop_layout = parallel_context_->layout;
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

  struct ProjectedCandidate {
    Fragment source_loop_layout;
    Fragment partial_layout;
    std::vector<reduction::ThreadReduceStep> thread_steps;
  };

  ReducerPlanningResult BuildPlans() const {
    ReducerPlanningResult result;
    std::unordered_map<Var, LocalCompleteCandidate, ObjectPtrHash,
                       ObjectPtrEqual>
        candidates;
    for (const auto &[var, buffer] : metadata_.buffers) {
      const ReducerPlanFacts &facts = facts_.at(var);
      std::vector<For> parallel_roots;
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

    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
        selected_local_complete;
    for (const auto &[var, candidate] : candidates) {
      bool has_conflict = false;
      for (const For &root : candidate.parallel_roots) {
        has_conflict |= conflicts.count(root) != 0;
      }
      if (has_conflict) {
        continue;
      }
      selected_local_complete.insert(var);
      for (const For &root : candidate.parallel_roots) {
        result.loop_layout_overrides.emplace(root, candidate.layout);
      }
    }

    for (const auto &[var, buffer] : metadata_.buffers) {
      const ReducerPlanFacts &facts = facts_.at(var);
      ReducerEpochPhysicalPlan epoch_plan;
      auto local_complete = candidates.find(var);
      if (selected_local_complete.count(var) != 0 &&
          local_complete != candidates.end()) {
        ReducerPartialGroupPlan group;
        group.partial_layout = local_complete->second.layout;
        for (const ReducerUpdateSite &site : facts.updates) {
          group.update_sites.push_back(site.call);
        }
        epoch_plan.groups.push_back(std::move(group));
      } else {
        epoch_plan =
            BuildProjectedGroups(buffer, facts, result.loop_layout_overrides);
      }
      if (epoch_plan.groups.empty()) {
        epoch_plan.groups.push_back(MakeCanonicalGroup(std::vector<Call>{}));
      }
      for (size_t group_index = 0; group_index < epoch_plan.groups.size();
           ++group_index) {
        for (const Call &update : epoch_plan.groups[group_index].update_sites) {
          ICHECK(!epoch_plan.update_to_group.count(update));
          epoch_plan.update_to_group.emplace(update, group_index);
        }
      }
      result.reducer_plans.emplace(var, std::move(epoch_plan));
    }
    return result;
  }

  ReducerEpochPhysicalPlan BuildProjectedGroups(
      const Buffer &logical_buffer, const ReducerPlanFacts &facts,
      const ReducerLoopLayoutMap &loop_layout_overrides) const {
    ReducerEpochPhysicalPlan result;
    std::optional<size_t> canonical_group;
    for (const ReducerUpdateSite &site : facts.updates) {
      std::optional<ProjectedCandidate> candidate;
      if (facts.supported && facts.destination_layout.defined()) {
        Fragment effective_layout = site.loop_layout;
        auto override_it = loop_layout_overrides.find(site.parallel_root);
        if (override_it != loop_layout_overrides.end()) {
          effective_layout = override_it->second;
        }
        candidate = TryBuildProjectedCandidate(logical_buffer,
                                               facts.destination_layout.value(),
                                               site, effective_layout);
      }

      if (!candidate.has_value()) {
        if (!canonical_group.has_value()) {
          canonical_group = result.groups.size();
          result.groups.push_back(MakeCanonicalGroup(std::vector<Call>{}));
        }
        result.groups[canonical_group.value()].update_sites.push_back(
            site.call);
        continue;
      }

      std::optional<size_t> matching_group;
      for (size_t i = 0; i < result.groups.size(); ++i) {
        if (GroupMatches(result.groups[i], candidate.value())) {
          matching_group = i;
          break;
        }
      }
      if (!matching_group.has_value()) {
        ReducerPartialGroupPlan group;
        group.partial_layout = candidate.value().partial_layout;
        group.source_loop_layout = candidate.value().source_loop_layout;
        group.thread_steps = candidate.value().thread_steps;
        result.groups.push_back(std::move(group));
        matching_group = result.groups.size() - 1;
      }
      result.groups[matching_group.value()].update_sites.push_back(site.call);
    }
    return result;
  }

  std::optional<ProjectedCandidate>
  TryBuildProjectedCandidate(const Buffer &logical_buffer,
                             const Fragment &destination_layout,
                             const ReducerUpdateSite &site,
                             const Fragment &effective_loop_layout) const {
    if (!site.supported || !site.parallel_root.defined() ||
        !effective_loop_layout.defined() ||
        effective_loop_layout->InputDim() != site.parallel_vars.size()) {
      return std::nullopt;
    }

    arith::Analyzer analyzer;
    for (const auto &[loop_var, domain] : site.loop_domains) {
      analyzer.Bind(loop_var, domain);
    }
    auto find_domain = [&](const Var &variable) -> Optional<Range> {
      for (const auto &[loop_var, domain] : site.loop_domains) {
        if (loop_var.same_as(variable)) {
          return domain;
        }
      }
      return std::nullopt;
    };

    for (size_t i = 0; i < site.parallel_vars.size(); ++i) {
      Optional<Range> domain = find_domain(site.parallel_vars[i]);
      if (!domain.defined() ||
          !analyzer.CanProveEqual(domain.value()->extent,
                                  effective_loop_layout->InputShape()[i])) {
        return std::nullopt;
      }
    }

    Fragment projection_layout = effective_loop_layout;
    std::optional<int> reduction_dim;
    bool scalar_output =
        logical_buffer->shape.size() == 1 && site.logical_indices.size() == 1 &&
        analyzer.CanProveEqual(logical_buffer->shape[0], Integer(1)) &&
        analyzer.CanProveEqual(site.logical_indices[0], Integer(0));
    if (scalar_output) {
      if (site.parallel_vars.size() != 1) {
        return std::nullopt;
      }
      reduction_dim = 0;
    } else if (site.parallel_vars.size() == logical_buffer->shape.size() + 1) {
      if (site.logical_indices.size() != logical_buffer->shape.size()) {
        return std::nullopt;
      }
      for (size_t candidate_dim = 0; candidate_dim < site.parallel_vars.size();
           ++candidate_dim) {
        bool matches = true;
        for (size_t output_dim = 0; output_dim < logical_buffer->shape.size();
             ++output_dim) {
          size_t parallel_dim =
              output_dim < candidate_dim ? output_dim : output_dim + 1;
          Optional<Range> domain =
              find_domain(site.parallel_vars[parallel_dim]);
          matches &= domain.defined() &&
                     analyzer.CanProveEqual(site.logical_indices[output_dim],
                                            site.parallel_vars[parallel_dim]) &&
                     analyzer.CanProveEqual(domain.value()->extent,
                                            logical_buffer->shape[output_dim]);
        }
        if (matches) {
          if (reduction_dim.has_value()) {
            return std::nullopt;
          }
          reduction_dim = static_cast<int>(candidate_dim);
        }
      }
    } else if (site.parallel_vars.size() == 1 &&
               site.logical_indices.size() == logical_buffer->shape.size()) {
      // Layout inference may fuse Parallel(M, K) into one contiguous logical
      // Range before reducer planning. Recover the compiler-known row-major
      // coordinates so that the omitted trailing coordinate remains an
      // explicit reduction axis for projection and thread-step analysis.
      Optional<Range> fused_domain = find_domain(site.parallel_vars[0]);
      const int64_t *fused_extent =
          fused_domain.defined()
              ? as_const_int(analyzer.Simplify(fused_domain.value()->extent))
              : nullptr;
      int64_t logical_elements = 1;
      std::vector<int64_t> logical_extents;
      logical_extents.reserve(logical_buffer->shape.size());
      for (const PrimExpr &extent : logical_buffer->shape) {
        const int64_t *constant = as_const_int(analyzer.Simplify(extent));
        if (constant == nullptr || *constant <= 0 ||
            logical_elements >
                std::numeric_limits<int64_t>::max() / *constant) {
          return std::nullopt;
        }
        logical_extents.push_back(*constant);
        logical_elements *= *constant;
      }
      if (fused_extent == nullptr || *fused_extent <= logical_elements ||
          *fused_extent % logical_elements != 0) {
        return std::nullopt;
      }
      int64_t reduction_extent = *fused_extent / logical_elements;
      PrimExpr fused = site.parallel_vars[0];
      int64_t stride = reduction_extent;
      for (int output_dim = static_cast<int>(logical_extents.size()) - 1;
           output_dim >= 0; --output_dim) {
        PrimExpr expected =
            FloorMod(FloorDiv(fused, Integer(stride)),
                     Integer(logical_extents[static_cast<size_t>(output_dim)]));
        if (!analyzer.CanProveEqual(
                site.logical_indices[static_cast<size_t>(output_dim)],
                expected)) {
          return std::nullopt;
        }
        stride *= logical_extents[static_cast<size_t>(output_dim)];
      }
      Array<PrimExpr> expanded_shape = logical_buffer->shape;
      expanded_shape.push_back(Integer(reduction_extent));
      projection_layout = Downcast<Fragment>(
          effective_loop_layout->Reshape(expanded_shape, &analyzer));
      reduction_dim = static_cast<int>(logical_buffer->shape.size());
    }
    if (!reduction_dim.has_value()) {
      return std::nullopt;
    }

    Fragment partial_layout = reduction::ComputeReducerLayout(
        projection_layout, reduction_dim.value());
    if (partial_layout->InputDim() != logical_buffer->shape.size()) {
      return std::nullopt;
    }
    for (size_t i = 0; i < logical_buffer->shape.size(); ++i) {
      if (!analyzer.CanProveEqual(partial_layout->InputShape()[i],
                                  logical_buffer->shape[i])) {
        return std::nullopt;
      }
    }
    for (const PrimExpr &extent : partial_layout->OutputShape()) {
      const int64_t *constant_extent = as_const_int(analyzer.Simplify(extent));
      if (constant_extent == nullptr || *constant_extent <= 0) {
        return std::nullopt;
      }
    }

    Array<PrimExpr> output_indices =
        reduction::InputPlaceholders(partial_layout->InputDim());
    for (size_t i = 0; i < output_indices.size(); ++i) {
      analyzer.Bind(Downcast<Var>(output_indices[i]),
                    Range(0, partial_layout->InputShape()[i]));
    }
    if (!ProveFragmentContains(destination_layout, partial_layout,
                               output_indices, output_indices, analyzer)) {
      return std::nullopt;
    }

    Array<IterVar> parallel_iters;
    parallel_iters.reserve(projection_layout->InputDim());
    for (size_t i = 0; i < projection_layout->InputDim(); ++i) {
      Var variable("reducer_projection_" + std::to_string(i));
      parallel_iters.push_back(
          IterVar(Range(0, projection_layout->InputShape()[i]), variable,
                  IterVarType::kDataPar));
    }
    Array<PrimExpr> parallel_indices = parallel_iters.Map(
        [](const IterVar &iter) { return PrimExpr(iter->var); });
    PrimExpr thread = projection_layout->ForwardThread(parallel_indices, {});
    arith::IterSumExpr thread_sum =
        arith::NormalizeToIterSum(thread, ToVMap(parallel_iters), &analyzer);
    std::optional<std::vector<reduction::ThreadReduceStep>> steps =
        reduction::TryCollectThreadReduceSteps(
            thread_sum, parallel_iters[reduction_dim.value()]->var);
    if (!steps.has_value()) {
      return std::nullopt;
    }
    for (const reduction::ThreadReduceStep &step : steps.value()) {
      int logical_width = step.extent;
      int shift = 0;
      if (step.scale <= 0 || logical_width <= 0 ||
          !is_const_power_of_two_integer(Integer(logical_width), &shift)) {
        return std::nullopt;
      }
    }
    return ProjectedCandidate{projection_layout, partial_layout,
                              std::move(steps.value())};
  }

  static ReducerPartialGroupPlan
  MakeCanonicalGroup(std::vector<Call> update_sites) {
    ReducerPartialGroupPlan group;
    group.canonical = true;
    group.update_sites = std::move(update_sites);
    return group;
  }

  static bool GroupMatches(const ReducerPartialGroupPlan &group,
                           const ProjectedCandidate &candidate) {
    return !group.canonical && group.partial_layout.defined() &&
           group.source_loop_layout.defined() &&
           LayoutsEqual(group.partial_layout.value(),
                        candidate.partial_layout) &&
           LayoutsEqual(group.source_loop_layout.value(),
                        candidate.source_loop_layout) &&
           group.thread_steps == candidate.thread_steps;
  }

  bool CanUseLocalComplete(const Buffer &logical_buffer,
                           const Fragment &destination_layout,
                           const std::vector<ReducerUpdateSite> &updates,
                           std::vector<For> *parallel_roots) const {
    arith::Analyzer shape_analyzer;
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
                      const ReducerEpochPlanMap &plans,
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
      const ReducerEpochPhysicalPlan &epoch = plans_.at(var);
      ICHECK(!epoch.groups.empty());
      std::vector<Buffer> partials;
      partials.reserve(epoch.groups.size());
      for (size_t group_index = 0; group_index < epoch.groups.size();
           ++group_index) {
        std::string suffix = "_partial_" + std::to_string(group_index);
        Var local_var(std::string(buffer->data->name_hint) + suffix, local_type,
                      buffer->data->span);
        Buffer local_buffer = buffer;
        BufferNode *local_buffer_ptr = local_buffer.CopyOnWrite();
        local_buffer_ptr->data = local_var;
        local_buffer_ptr->name = std::string(buffer->name) + suffix;
        const ReducerPartialGroupPlan &group = epoch.groups[group_index];
        if (!group.canonical) {
          ICHECK(group.partial_layout.defined());
          ICHECK(local_buffer_ptr->strides.empty())
              << "Projected reducer storage requires compact strides";
          local_buffer_ptr->shape = group.partial_layout.value()->OutputShape();
        }
        partials.push_back(std::move(local_buffer));
      }
      group_buffers_.emplace(buffer, std::move(partials));
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
    Array<Buffer> allocations;
    for (const Buffer &buffer : op->alloc_buffers) {
      auto it = group_buffers_.find(buffer);
      if (it == group_buffers_.end()) {
        allocations.push_back(buffer);
        continue;
      }
      for (const Buffer &partial : it->second) {
        allocations.push_back(partial);
      }
    }
    block_ptr->alloc_buffers = std::move(allocations);
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
    const ReducerEpochPhysicalPlan &epoch = plans_.at(reducer_var.value());
    Buffer logical_buffer = reducer_access.region->buffer;
    auto buffers_it = group_buffers_.find(logical_buffer);
    ICHECK(buffers_it != group_buffers_.end());
    const std::vector<Buffer> &partials = buffers_it->second;
    ICHECK_EQ(partials.size(), epoch.groups.size());

    if (is_init) {
      Array<PrimExpr> args;
      args.reserve(partials.size());
      for (const Buffer &partial : partials) {
        args.push_back(MakeFullRegionCall(partial, reducer_access.access_mask,
                                          op->args[0]->span));
      }
      call_ptr->args = std::move(args);
      return call;
    }

    if (is_update) {
      auto group_it = epoch.update_to_group.find(GetRef<Call>(op));
      ICHECK(group_it != epoch.update_to_group.end())
          << "Reducer update site is missing a physical partial group";
      size_t group_index = group_it->second;
      ICHECK_LT(group_index, epoch.groups.size());
      const ReducerPartialGroupPlan &group = epoch.groups[group_index];
      Array<PrimExpr> physical_indices = logical_update_indices;
      if (!group.canonical) {
        ICHECK(group.partial_layout.defined());
        physical_indices =
            group.partial_layout.value()->Forward(logical_update_indices);
      }
      Array<PrimExpr> physical_extents;
      physical_extents.reserve(physical_indices.size());
      for (const PrimExpr &index : physical_indices) {
        physical_extents.push_back(make_const(index.dtype(), 1));
      }
      Array<PrimExpr> args = {MakeRegionCall(partials[group_index],
                                             physical_indices, physical_extents,
                                             reducer_access.access_mask,
                                             op->args[0]->span),
                              call->args[1]};
      call_ptr->args = std::move(args);
      call_ptr->annotations.Set(attr::kReducerLogicalIndices,
                                logical_update_indices);
      if (group.canonical) {
        call_ptr->annotations.Set(attr::kReducerParallelOnce,
                                  Bool(parallel_depth_ > 0));
      } else {
        call_ptr->annotations.Set(attr::kReducerPartitionRequired, Bool(true));
      }
      return call;
    }

    ICHECK(is_finalize);
    Array<PrimExpr> args;
    Array<ReducerPartialPlan> partial_plans;
    args.reserve(partials.size() + 1);
    partial_plans.reserve(partials.size());
    for (size_t group_index = 0; group_index < epoch.groups.size();
         ++group_index) {
      const ReducerPartialGroupPlan &group = epoch.groups[group_index];
      args.push_back(MakeFullRegionCall(partials[group_index], kAccessRead,
                                        op->args[0]->span));
      Array<Integer> step_extents;
      Array<Integer> step_scales;
      step_extents.reserve(group.thread_steps.size());
      step_scales.reserve(group.thread_steps.size());
      for (const reduction::ThreadReduceStep &step : group.thread_steps) {
        step_extents.push_back(Integer(step.extent));
        step_scales.push_back(Integer(step.scale));
      }
      partial_plans.push_back(
          ReducerPartialPlan(group.canonical, group.partial_layout,
                             std::move(step_extents), std::move(step_scales)));
    }
    args.push_back(call->args[1]);
    call_ptr->args = std::move(args);
    call_ptr->annotations.Set(attr::kReducerPartialPlans,
                              std::move(partial_plans));
    return call;
  }

  const ReducerMetadata &metadata_;
  const ReducerEpochPlanMap &plans_;
  const ReducerLoopLayoutMap &loop_layout_overrides_;
  std::unordered_map<Buffer, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>
      group_buffers_;

  static PrimExpr MakeFullRegionCall(const Buffer &buffer, int access_mask,
                                     const Span &span) {
    Array<PrimExpr> mins;
    Array<PrimExpr> extents;
    mins.reserve(buffer->shape.size());
    extents.reserve(buffer->shape.size());
    for (const PrimExpr &extent : buffer->shape) {
      mins.push_back(make_zero(extent.dtype()));
      extents.push_back(extent);
    }
    return MakeRegionCall(buffer, mins, extents, access_mask, span);
  }

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
           op->attr_key != attr::kParallelPartitionRequired &&
           op->attr_key != attr::kReducerUpdate)
        << "Unconsumed reducer lowering marker reached backend lowering";
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

tvm::transform::Pass PlanAndMaterializeReducers() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext) {
    ReducerMetadata metadata = CollectReducerMetadata(func);
    if (metadata.info.empty()) {
      return func;
    }
    // Planning is the final semantic boundary before physical storage is
    // introduced. Re-verify here because earlier pipeline transforms (for
    // example warp specialization or software pipelining) may have changed the
    // execution scope since the frontend verification pass.
    VerifyReducerEpochSemantics(func, metadata);
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
      .def("tl.transform.PlanAndMaterializeReducers",
           PlanAndMaterializeReducers)
      .def("tl.transform.VerifyReducerLowered", VerifyReducerLowered);
}

} // namespace tl
} // namespace tvm
