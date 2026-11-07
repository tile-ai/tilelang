/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file dependency_analysis.cc
 * \brief Shared schedule and synchronization dependency analysis.
 */

#include "./dependency_analysis.h"
#include "./task_analysis.h"
#include "./task_annotations.h"
#include "ascend/op/utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend/layout/ascend_layouts.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"
#include "transform/common/int64_promoter.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// An undefined storage key groups explicit conflicts between distinct storage.
using CoveredDependencyMap =
    std::unordered_map<ffi::Optional<Var>, std::set<DependencyTaskPair>,
                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// A bare buffer hint matches every region using its storage. A region hint
// matches only the same logical buffer and exact footprint.
static bool HintOperandMatches(const Any &hint, const BufferRegion &query) {
  if (auto var = hint.as<Var>())
    return var.value().same_as(query->buffer->data);
  BufferRegion hint_region = Downcast<BufferRegion>(hint);
  return hint_region->buffer.same_as(query->buffer) &&
         RegionsEqual(hint_region->region, query->region);
}

static ConflictHintList
ActiveConflictHints(ControlNode *loop, const ConflictHintList &root_hints) {
  if (loop == nullptr)
    return root_hints;
  auto ann = loop->control->annotations.Get("conflict_hint");
  return ann.has_value() ? ann.value().cast<ConflictHintList>()
                         : ConflictHintList{};
}

static bool IsConflictHint(const Array<Any> &hint) {
  ICHECK_EQ(hint.size(), 4u)
      << "conflict_hint annotation expects 4-element entries";
  const auto *is_conflict = hint[3].as<IntImmNode>();
  ICHECK(is_conflict != nullptr && is_conflict->dtype.is_bool())
      << "conflict_hint annotation expects a constant boolean polarity";
  return is_conflict->value;
}

// Return the suffix of the enclosing loop nest that forms one storage-local
// iteration space. Walk outward from `loop` until the parent contains another
// access to `storage` outside the child on this path. That parent is the scope
// where separate access sequences meet, so only its child-to-`loop` suffix is
// renamed when comparing two iterations. If no such parent exists, use the
// complete enclosing loop nest.
std::vector<const ControlNode *> StorageIterationNest(ControlNode *loop,
                                                      const Var &storage) {
  std::vector<const ControlNode *> controls = loop->GetAncestorControls();
  size_t first = 0;
  for (size_t i = controls.size(); i > 1; --i) {
    const ControlNode *parent = controls[i - 2];
    const IRStructure *path_child = controls[i - 1];
    ICHECK_EQ(path_child->GetParent(), parent);

    bool has_external_access =
        parent->task && parent->task->TouchesStorage(storage);
    for (const auto &child : parent->children) {
      if (child.get() != path_child && child->TouchesStorage(storage)) {
        has_external_access = true;
        break;
      }
    }
    if (has_external_access) {
      first = i - 1;
      break;
    }
  }
  return {controls.begin() + first, controls.end()};
}

static std::optional<bool>
DeclaredRegionConflict(const BufferRegion &a_region,
                       const BufferRegion &b_region, ControlNode *loop,
                       int offset, const ConflictHintList &root_conflicts) {
  bool declared_no_conflict = false;
  for (const Any &entry : ActiveConflictHints(loop, root_conflicts)) {
    Array<Any> hint = Downcast<Array<Any>>(entry);
    bool is_conflict = IsConflictHint(hint);
    int64_t cross_code = Downcast<IntImm>(hint[2])->value;
    if (cross_code != -1 && (cross_code == 1) != (offset != 0))
      continue;
    bool matches = (HintOperandMatches(hint[0], a_region) &&
                    HintOperandMatches(hint[1], b_region)) ||
                   (HintOperandMatches(hint[0], b_region) &&
                    HintOperandMatches(hint[1], a_region));
    if (!matches)
      continue;
    if (is_conflict)
      return true;
    declared_no_conflict = true;
  }
  // A forced conflict wins if contradictory declarations match.
  if (declared_no_conflict)
    return false;
  return std::nullopt;
}

static bool
RegionsMayOverlap(const ConstrSet &a_ctx, const BufferRegion &a_region,
                  const ConstrSet &b_ctx, const BufferRegion &b_region,
                  ControlNode *loop, int offset, size_t num_storage_owners,
                  Optional<PrimExpr> a_pitch, Optional<PrimExpr> b_pitch) {
  const bool cross = offset != 0;
  if (cross && loop == nullptr)
    return false;

  if (!a_region->buffer->data.same_as(b_region->buffer->data))
    return false;

  // Same storage but different logical buffer (T.view/T.reshape): regions may
  // differ in rank/dtype/strides, so keep the dependency conservatively.
  if (!a_region->buffer.same_as(b_region->buffer))
    return true;

  const Array<Range> &a_rng = a_region->region;
  const Array<Range> &b_rng = b_region->region;
  if (a_rng.size() != b_rng.size())
    return true; // shape mismatch — be safe
  const size_t ndim = a_rng.size();

  arith::Analyzer ana;
  ana.z3_prover.SetRLimit(50000);
  Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet a_local_ctx = a_ctx;
  ConstrSet b_local_ctx = b_ctx;
  ConstrSet loop_ctx;
  // Scheduled tasks can define condition snapshots outside the constraint
  // visitor's Bind stack. Rename all variables written in the compared
  // iteration scope, including these otherwise-free predicates. Sharing one
  // predicate between producer and consumer would incorrectly make opposite
  // branches mutually exclusive across different iterations.
  auto seed_iteration_vars = [&](const ControlNode *scope) {
    for (const Var &var : scope->GetWriteVars()) {
      if (!a_sub.count(var))
        a_sub.Set(var, var.copy_with_suffix("_p"));
      if (!b_sub.count(var))
        b_sub.Set(var, var.copy_with_suffix("_c"));
    }
  };
  if (cross) {
    if (offset < 0) {
      std::vector<const ControlNode *> iteration_nest =
          num_storage_owners > 1
              ? loop->GetAncestorControls()
              : StorageIterationNest(loop, a_region->buffer->data);
      ICHECK(!iteration_nest.empty());
      seed_iteration_vars(iteration_nest.front());
      Var pivot_lv = iteration_nest.front()->control->loop_var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot_lv);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot_lv);
      if (num_storage_owners <= 1) {
        PrimExpr different_iteration = Bool(false);
        for (const ControlNode *control : iteration_nest) {
          Var iteration = control->control->loop_var;
          PrimExpr p = Substitute(iteration, a_sub);
          PrimExpr c = Substitute(iteration, b_sub);
          different_iteration = different_iteration || (p != c);
        }
        loop_ctx.AddConstr(different_iteration);
      }
    } else {
      seed_iteration_vars(loop);
      Var iteration = loop->control->loop_var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, iteration);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, iteration);
      PrimExpr p = Substitute(iteration, a_sub);
      PrimExpr c = Substitute(iteration, b_sub);
      loop_ctx.AddConstr(c == p + IntImm(iteration.dtype(), offset));
    }
  }
  loop_ctx.Merge(a_local_ctx).Merge(b_local_ctx).Populate(ana);

  FreshenMutableReads freshen_a(FreshenMutableReads::Mode::kSnapshot);
  FreshenMutableReads freshen_b(FreshenMutableReads::Mode::kSnapshot);
  size_t compared_dims = ndim;
  const Buffer &buffer = a_region->buffer;
  if (IsL0ABuffer(buffer) || IsL0BBuffer(buffer) || IsL0CBuffer(buffer)) {
    bool same_pitch = false;
    if (a_pitch.defined() && b_pitch.defined()) {
      PrimExpr equal = freshen_a(Substitute(a_pitch.value(), a_sub)) ==
                       freshen_b(Substitute(b_pitch.value(), b_sub));
      same_pitch = ana.CanProve(equal) || ana.z3_prover.CanProve(equal);
    }
    // L0 matrix regions may overlap unless their pitches are provably equal.
    if (!same_pitch)
      compared_dims = ndim > 2 ? ndim - 2 : 0;
  }
  PrimExpr any_dim_disjoint = Bool(false);
  Int64Promoter promoter;
  auto widen_index = [&](const PrimExpr &value) {
    return cast(DataType::Int(64), promoter(value));
  };
  for (size_t i = 0; i < compared_dims; ++i) {
    // Widen the arithmetic before forming endpoints. Casting a whole int32
    // expression hides min/max tail bounds from the arithmetic simplifier.
    PrimExpr a_min = widen_index(freshen_a(Substitute(a_rng[i]->min, a_sub)));
    PrimExpr a_extent =
        widen_index(freshen_a(Substitute(a_rng[i]->extent, a_sub)));
    PrimExpr b_min = widen_index(freshen_b(Substitute(b_rng[i]->min, b_sub)));
    PrimExpr b_extent =
        widen_index(freshen_b(Substitute(b_rng[i]->extent, b_sub)));
    // A half-open endpoint can exceed the int32 index range by one even
    // for an ordinary single-element access.
    PrimExpr a_end = a_min + a_extent;
    PrimExpr b_end = b_min + b_extent;
    any_dim_disjoint = any_dim_disjoint || (a_extent <= 0) || (b_extent <= 0) ||
                       (a_end <= b_min) || (b_end <= a_min);
  }
  any_dim_disjoint = ana.Simplify(any_dim_disjoint);
  // Raw Z3 as a fallback catches the infeasible-context case that Analyzer's
  // constant-fold short-circuit misses.
  return !(ana.CanProve(any_dim_disjoint) ||
           ana.z3_prover.CanProve(any_dim_disjoint));
}

bool RegionsMayConflict(const ConstrSet &a_ctx, const BufferRegion &a_region,
                        const ConstrSet &b_ctx, const BufferRegion &b_region,
                        ControlNode *loop, int offset,
                        size_t num_storage_owners,
                        const ConflictHintList &root_conflicts,
                        Optional<PrimExpr> a_pitch,
                        Optional<PrimExpr> b_pitch) {
  if (offset != 0 && loop == nullptr)
    return false;
  if (auto declared = DeclaredRegionConflict(a_region, b_region, loop, offset,
                                             root_conflicts))
    return *declared;
  return RegionsMayOverlap(a_ctx, a_region, b_ctx, b_region, loop, offset,
                           num_storage_owners, a_pitch, b_pitch);
}

struct CompoundTaskAccesses {
  bool has_loop{false};
  bool complete{true};
  std::vector<Var> local_vars;
  std::vector<BufferRegion> reads;
  std::vector<BufferRegion> writes;
};

// Keep loop-dependent addresses symbolic instead of taking the rectangular
// hull of an entire atomic task. Ignoring internal bounds, guards and Bind
// relations overestimates execution, so an inconclusive query stays a conflict.
// Only complete, ordinary GM/UB copy footprints with stable dense layouts can
// refine dependencies; other accesses are retained solely for hint matching.
class CompoundTaskAccessCollector : public StmtVisitor {
public:
  static CompoundTaskAccesses Collect(const TaskNode *task) {
    CompoundTaskAccessCollector collector;
    collector.outer_ctx_ = task->outer_ctx;
    PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>()) {
        collector.result_.has_loop = true;
        collector.result_.local_vars.push_back(loop->loop_var);
      } else if (const auto *bind = node.as<BindNode>()) {
        collector.result_.local_vars.push_back(bind->var);
      }
    });
    uint16_t pipe = task->GetPipeMask();
    if (pipe == 0 || (pipe & (pipe - 1)) != 0)
      collector.result_.complete = false;
    // A task's footprint also depends on enclosing control. Do not refine
    // accesses whose indices, layout or execution read mutable state.
    for (const IRStructure *node = task; node; node = node->GetParent()) {
      for (const auto &guard : node->GetGuards()) {
        if (guard->IsCondition()) {
          collector.CheckPure(
              static_cast<const ConditionGuard *>(guard.get())->condition);
        } else {
          const auto *attribute =
              static_cast<const AttributeGuard *>(guard.get());
          if (auto expr = attribute->node.as<PrimExpr>())
            collector.CheckPure(expr.value());
          if (attribute->value.defined())
            collector.CheckAttributeValue(attribute->value);
        }
      }
      if (node->IsControl()) {
        const auto &control = static_cast<const ControlNode *>(node)->control;
        collector.CheckPure(control->min);
        collector.CheckPure(control->extent);
        if (control->step.has_value())
          collector.CheckPure(control->step.value());
      }
    }
    for (const Constr &constraint : task->outer_ctx.constrs_) {
      if (constraint.kind == Constr::kBindRange) {
        collector.CheckPure(constraint.range->min);
        collector.CheckPure(constraint.range->extent);
      } else {
        collector.CheckPure(constraint.value);
      }
    }
    collector(task->stmt);
    // Scheduled task regions also include reads from their detached guards.
    for (const auto &guard : task->GetGuards()) {
      if (guard->IsCondition()) {
        collector.RecordControl(
            static_cast<const ConditionGuard *>(guard.get())->condition);
      } else {
        const auto *attribute =
            static_cast<const AttributeGuard *>(guard.get());
        if (const auto *expr = attribute->node.as<PrimExprNode>())
          collector.RecordControl(GetRef<PrimExpr>(expr));
      }
    }
    return std::move(collector.result_);
  }

private:
  void CheckAttributeValue(const PrimExpr &expr) {
    // Assumption guards carry a constant diagnostic string in their value;
    // their predicate is the node and still requires the strict pure check.
    if (!expr.as<StringImmNode>())
      CheckPure(expr);
  }

  void CheckPure(const PrimExpr &expr) {
    result_.complete &= expr.dtype().is_scalar() &&
                        (expr.dtype().is_int() || expr.dtype().is_uint() ||
                         expr.dtype().is_bool()) &&
                        SideEffect(expr) <= CallEffectKind::kPure;
  }

  void CheckLayout(const PrimExpr &expr) {
    CheckPure(expr);
    // Function parameters describe one shared layout. Internal or enclosing
    // loop/Bind values can differ between the compared access instances.
    for (const Var &var : result_.local_vars) {
      if (UsesVar(expr, [&](const VarNode *v) { return v == var.get(); }))
        result_.complete = false;
    }
    for (const Constr &constraint : outer_ctx_.constrs_) {
      if (constraint.kind != Constr::kConstr &&
          UsesVar(expr,
                  [&](const VarNode *v) { return v == constraint.var.get(); }))
        result_.complete = false;
    }
  }

  void CheckBuffer(const Buffer &buffer, const Region &ranges) {
    if (ranges.size() != buffer->shape.size() ||
        !buffer->axis_separators.empty() ||
        (!buffer->strides.empty() &&
         buffer->strides.size() != buffer->shape.size())) {
      result_.complete = false;
      return;
    }
    CheckLayout(buffer->elem_offset);
    arith::Analyzer analyzer;
    Int64Promoter promoter;
    PrimExpr dense_stride = IntImm(DataType::Int(64), 1);
    for (size_t i = buffer->shape.size(); i > 0; --i) {
      CheckLayout(buffer->shape[i - 1]);
      if (!buffer->strides.empty()) {
        CheckLayout(buffer->strides[i - 1]);
        if (!analyzer.CanProveEqual(
                cast(DataType::Int(64), promoter(buffer->strides[i - 1])),
                dense_stride))
          result_.complete = false;
      }
      dense_stride = dense_stride *
                     cast(DataType::Int(64), promoter(buffer->shape[i - 1]));
    }
    for (const Range &range : ranges) {
      CheckPure(range->min);
      CheckPure(range->extent);
    }
  }

  void CheckCopy(const CallNode *call) {
    AscendCopy copy(call->args, call->annotations);
    bool ordinary = (IsGlobalBuffer(copy->src) && IsSharedBuffer(copy->dst)) ||
                    (IsSharedBuffer(copy->src) && IsGlobalBuffer(copy->dst));
    if (!ordinary || copy->nd2nz || copy->transpose || copy->dual_dst_ctl ||
        copy->data_select || copy->pad_value.defined() ||
        copy->mx_sf_data.defined() || copy->l0_pitch.defined() ||
        !is_zero(copy->unit_flag_ctl) || !is_zero(copy->sub_blockid) ||
        ++copies_ > 16) {
      result_.complete = false;
      return;
    }
    for (const auto &[key, value] : call->annotations) {
      if (auto expr = value.as<PrimExpr>())
        CheckPure(expr.value());
      else
        result_.complete = false;
    }
    CheckBuffer(copy->src, copy->src_range);
    CheckBuffer(copy->dst, copy->dst_range);
  }

  void Record(const Stmt &stmt) {
    // Keep even unsupported accesses for explicit hint matching. They cannot
    // justify eliminating a dependency: a region list need not describe an
    // opaque pointer use or a special transfer's full physical footprint.
    TaskAccessInfo accesses = AnalyzeTaskAccesses(stmt);
    result_.reads.insert(result_.reads.end(), accesses.read_regions.begin(),
                         accesses.read_regions.end());
    result_.writes.insert(result_.writes.end(), accesses.write_regions.begin(),
                          accesses.write_regions.end());
  }

  void RecordControl(const PrimExpr &expr) {
    CheckPure(expr);
    Record(Evaluate(expr));
  }

  void VisitStmt(const Stmt &stmt) final {
    if (stmt.as<SeqStmtNode>() || stmt.as<ForNode>() ||
        stmt.as<IfThenElseNode>() || stmt.as<AttrStmtNode>()) {
      StmtVisitor::VisitStmt(stmt);
    } else {
      Record(stmt);
      if (const auto *bind = stmt.as<BindNode>()) {
        CheckPure(bind->value);
        return;
      }
      if (const auto *eval = stmt.as<EvaluateNode>()) {
        if (is_zero(eval->value))
          return;
        const auto *call = eval->value.as<CallNode>();
        if (call && IsAscendCopyCall(call) && call->args.size() == 2) {
          CheckCopy(call);
          return;
        }
        if (call && task_analysis_detail::IsCrossCoreSyncCall(call) &&
            call->args.size() == 3 && call->args[0].as<IntImmNode>() &&
            call->args[1].as<StringImmNode>()) {
          // These intrinsics order their pipe without accessing tensor storage.
          // Keep the calls and their resource effects intact; only their scalar
          // flag operand contributes accesses to this footprint analysis. A
          // mutable flag expression still prevents refinement.
          CheckPure(call->args[2]);
          return;
        }
      }
      result_.complete = false;
    }
  }

  void VisitStmt_(const ForNode *op) final {
    if ((op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) ||
        (op->step.has_value() && !is_one(op->step.value())))
      result_.complete = false;
    RecordControl(op->min);
    RecordControl(op->extent);
    if (op->step.has_value())
      RecordControl(op->step.value());
    VisitStmt(op->body);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    RecordControl(op->condition);
    VisitStmt(op->then_case);
    if (op->else_case.defined())
      VisitStmt(op->else_case.value());
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tl::attr::kAscendTask &&
        op->attr_key != tl::attr::kAscendPerCoreTask &&
        op->attr_key != tirx::attr::tilelang_assume &&
        op->attr_key != tl::attr::kAssumeRequiresRuntimeCheck)
      result_.complete = false;
    if (const auto *expr = op->node.as<PrimExprNode>())
      RecordControl(GetRef<PrimExpr>(expr));
    CheckAttributeValue(op->value);
    Record(Evaluate(op->value));
    VisitStmt(op->body);
  }

  CompoundTaskAccesses result_;
  ConstrSet outer_ctx_;
  size_t copies_{0};
};

std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes, ControlNode *loop,
                    const BufferVersionMap &manual_buffer_versions,
                    const MultiBufferOwnerMap &multi_buffer_owners,
                    DependencyCache *dependency_cache,
                    const ConflictHintList &root_conflicts) {
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  if (loop) {
    auto cache_it = cache.find(loop);
    if (cache_it != cache.end())
      return cache_it->second;
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const IRStructure *lhs, const IRStructure *rhs) {
                       return lhs->GetStage() < rhs->GetStage();
                     });
  }

  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  struct RegionAccess {
    TaskNode *task;
    BufferRegion region;
    bool is_write;
  };
  struct NodeAccesses {
    std::vector<RegionAccess> reads;
    std::vector<RegionAccess> writes;
  };
  std::vector<NodeAccesses> accesses(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(nodes[i], tasks);
    for (TaskNode *task : tasks) {
      for (const BufferRegion &region : task->GetReadRegions())
        accesses[i].reads.push_back({task, region, false});
      for (const BufferRegion &region : task->GetWriteRegions())
        accesses[i].writes.push_back({task, region, true});
    }
  }

  std::unordered_map<TaskNode *, CompoundTaskAccesses> compound_accesses;
  size_t refinement_budget = 64;
  ConflictHintList positive_hints;
  for (const Any &entry : ActiveConflictHints(loop, root_conflicts)) {
    if (IsConflictHint(Downcast<Array<Any>>(entry)))
      positive_hints.push_back(entry);
  }
  auto get_compound_accesses =
      [&](TaskNode *task) -> const CompoundTaskAccesses & {
    auto it = compound_accesses.find(task);
    if (it == compound_accesses.end())
      it = compound_accesses
               .emplace(task, CompoundTaskAccessCollector::Collect(task))
               .first;
    return it->second;
  };
  auto may_conflict = [&](const RegionAccess &lhs, const RegionAccess &rhs,
                          int distance, size_t owners) {
    if (distance != 0 && loop == nullptr)
      return false;
    auto declared = DeclaredRegionConflict(lhs.region, rhs.region, loop,
                                           distance, root_conflicts);
    if (declared == std::optional<bool>(true))
      return true;
    // Match original leaf regions before freshening, and before a storage-wide
    // no-conflict or a disjoint hull can hide an explicit positive declaration.
    // Match each operand separately to avoid a quadratic leaf-pair hint scan.
    for (const Any &entry : positive_hints) {
      Array<Any> hint = Downcast<Array<Any>>(entry);
      int64_t cross_code = Downcast<IntImm>(hint[2])->value;
      if (cross_code != -1 && (cross_code == 1) != (distance != 0))
        continue;
      auto matches = [&](const Any &operand, const RegionAccess &access) {
        if (HintOperandMatches(operand, access.region))
          return true;
        const auto &compound = get_compound_accesses(access.task);
        for (const BufferRegion &region :
             access.is_write ? compound.writes : compound.reads) {
          if (region->buffer.same_as(access.region->buffer) &&
              HintOperandMatches(operand, region))
            return true;
        }
        return false;
      };
      if ((matches(hint[0], lhs) && matches(hint[1], rhs)) ||
          (matches(hint[1], lhs) && matches(hint[0], rhs)))
        return true;
    }
    if (declared.has_value())
      return *declared;
    const Var &lhs_storage = lhs.region->buffer->data;
    const Var &rhs_storage = rhs.region->buffer->data;
    Optional<PrimExpr> lhs_pitch = lhs.task->GetL0Pitch(lhs_storage);
    Optional<PrimExpr> rhs_pitch = rhs.task->GetL0Pitch(rhs_storage);
    if (!RegionsMayOverlap(lhs.task->outer_ctx, lhs.region, rhs.task->outer_ctx,
                           rhs.region, loop, distance, owners, lhs_pitch,
                           rhs_pitch))
      return false;
    if (!lhs.region->buffer.same_as(rhs.region->buffer) ||
        !IsGlobalBuffer(lhs.region->buffer) || refinement_budget == 0)
      return true;

    const auto &a = get_compound_accesses(lhs.task);
    const auto &b = get_compound_accesses(rhs.task);
    if (!a.complete || !b.complete || (!a.has_loop && !b.has_loop))
      return true;
    // Two invocations of the same atomic task can use different internal
    // iterations, even at logical distance zero or when comparing it to itself.
    Map<Var, PrimExpr> a_sub, b_sub;
    for (const Var &var : a.local_vars)
      a_sub.Set(var, var.copy_with_suffix("_a"));
    for (const Var &var : b.local_vars)
      b_sub.Set(var, var.copy_with_suffix("_b"));
    bool found_a = false, found_b = false;
    for (const BufferRegion &a_region : lhs.is_write ? a.writes : a.reads) {
      if (!a_region->buffer.same_as(lhs.region->buffer))
        continue;
      found_a = true;
      for (const BufferRegion &b_region : rhs.is_write ? b.writes : b.reads) {
        if (!b_region->buffer.same_as(rhs.region->buffer))
          continue;
        found_b = true;
        if (refinement_budget == 0)
          return true;
        --refinement_budget;
        if (RegionsMayOverlap(lhs.task->outer_ctx.Substitute(a_sub),
                              BufferRegion(a_region->buffer,
                                           Substitute(a_region->region, a_sub)),
                              rhs.task->outer_ctx.Substitute(b_sub),
                              BufferRegion(b_region->buffer,
                                           Substitute(b_region->region, b_sub)),
                              loop, distance, owners, lhs_pitch, rhs_pitch))
          return true;
      }
    }
    // Missing access metadata is never evidence of disjointness.
    return !found_a || !found_b;
  };

  std::function<void(IRStructure *, CoveredDependencyMap &)>
      collect_subtree_deps = [&](IRStructure *node, CoveredDependencyMap &out) {
        if (!node || !node->IsControl())
          return;
        auto *ctrl = static_cast<ControlNode *>(node);
        std::vector<IRStructure *> children;
        for (const auto &child : ctrl->children)
          children.push_back(child.get());
        auto child_deps =
            AnalyzeDependencies(children, ctrl, manual_buffer_versions,
                                multi_buffer_owners, &cache, root_conflicts);
        for (const auto &cdep : child_deps) {
          auto &covered_pairs = out[cdep.storage];
          covered_pairs.insert(cdep.task_pairs.begin(), cdep.task_pairs.end());
        }
        for (auto *child : children)
          collect_subtree_deps(child, out);
      };
  auto subtree_covered = [&](IRStructure *node) {
    CoveredDependencyMap covered;
    collect_subtree_deps(node, covered);
    return covered;
  };

  auto get_manual_versions = [&](const ffi::Optional<Var> &storage) {
    if (!storage.has_value())
      return 0;
    auto it = manual_buffer_versions.find(storage.value());
    return it == manual_buffer_versions.end() ? 0 : (*it).second;
  };
  auto get_num_storage_owners =
      [&](const ffi::Optional<Var> &storage) -> size_t {
    if (loop == nullptr || !storage.has_value())
      return 0;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() ||
        std::find(owners->second.begin(), owners->second.end(), loop) ==
            owners->second.end()) {
      return 0;
    }
    return owners->second.size();
  };
  auto find_owner = [](const std::vector<ControlNode *> &owners,
                       const TaskNode *task) -> ControlNode * {
    ControlNode *result = nullptr;
    for (ControlNode *owner : owners) {
      if (!task->IsWithin(owner))
        continue;
      // Nested owners deliberately remain data dependencies here;
      // PrepareMultiBuffer diagnoses their overlapping ownership.
      if (result != nullptr)
        return nullptr;
      result = owner;
    }
    return result;
  };
  auto is_owner_exclusion = [&](const ffi::Optional<Var> &storage,
                                const TaskNode *lhs, const TaskNode *rhs) {
    if (!storage.has_value())
      return false;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() || owners->second.size() <= 1)
      return false;
    ControlNode *lhs_owner = find_owner(owners->second, lhs);
    ControlNode *rhs_owner = find_owner(owners->second, rhs);
    return lhs_owner != nullptr && rhs_owner != nullptr &&
           lhs_owner != rhs_owner;
  };

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (loop == nullptr && i >= j)
        continue;
      struct PendingDependency {
        ffi::Optional<Var> storage;
        int distance;
        DependencyKind kind;
        std::set<DependencyTaskPair> task_pairs;
      };
      std::vector<PendingDependency> pending;

      auto add_pair = [&](const ffi::Optional<Var> &storage, int distance,
                          TaskNode *producer, TaskNode *consumer,
                          DependencyKind kind = DependencyKind::kData) {
        auto existing = std::find_if(
            pending.begin(), pending.end(), [&](const PendingDependency &dep) {
              return dep.storage.same_as(storage) && dep.distance == distance &&
                     dep.kind == kind;
            });
        if (existing == pending.end()) {
          pending.push_back({storage, distance, kind, {}});
          existing = std::prev(pending.end());
        }
        existing->task_pairs.emplace(producer, consumer);
      };

      auto consider = [&](const std::vector<RegionAccess> &lhs_accesses,
                          const std::vector<RegionAccess> &rhs_accesses) {
        for (const RegionAccess &lhs : lhs_accesses) {
          for (const RegionAccess &rhs : rhs_accesses) {
            if (IsSharedBuffer(lhs.region->buffer) &&
                IsSharedBuffer(rhs.region->buffer)) {
              // A common UB offset (e.g. a hardware dual-copy destination)
              // still names different physical memory on the two AIVs.
              auto local_sid = [](const TaskNode *task) -> std::optional<int> {
                if (IsVectorCore(task->GetCoreMask()))
                  return VectorSid(task->GetCoreMask());
                return GetTaskSubcore(task->stmt);
              };
              auto lhs_sid = local_sid(lhs.task), rhs_sid = local_sid(rhs.task);
              if (lhs_sid && rhs_sid && *lhs_sid != *rhs_sid)
                continue;
            }
            const Var &lhs_storage = lhs.region->buffer->data;
            const Var &rhs_storage = rhs.region->buffer->data;
            ffi::Optional<Var> storage;
            if (lhs_storage.same_as(rhs_storage))
              storage = lhs_storage;
            bool owner_exclusion =
                is_owner_exclusion(storage, lhs.task, rhs.task);
            // The i<j visit sees every cross-owner write/read, write/write,
            // and read/write conflict once. Such conflicts are undirected:
            // the scheduler chooses one physical owner order.
            if (owner_exclusion && i >= j)
              continue;
            int manual_versions = get_manual_versions(storage);
            if (manual_versions && !owner_exclusion) {
              for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                if (may_conflict(lhs, rhs, d, /*owners=*/0)) {
                  add_pair(storage, d, lhs.task, rhs.task);
                  break;
                }
              }
              continue;
            }
            for (int distance : {0, -1}) {
              if (distance == 0 && i >= j)
                continue;
              if (distance < 0 && loop == nullptr)
                continue;
              if (may_conflict(lhs, rhs, distance,
                               get_num_storage_owners(storage))) {
                add_pair(storage, owner_exclusion ? 0 : distance, lhs.task,
                         rhs.task,
                         owner_exclusion ? DependencyKind::kOwnerExclusion
                                         : DependencyKind::kData);
                break;
              }
            }
          }
        }
      };
      consider(accesses[i].writes, accesses[j].reads);
      consider(accesses[i].writes, accesses[j].writes);
      consider(accesses[i].reads, accesses[j].writes);

      if (i == j && nodes[i]->IsControl()) {
        CoveredDependencyMap covered = subtree_covered(nodes[i]);
        for (PendingDependency &dep : pending) {
          auto covered_it = covered.find(dep.storage);
          if (covered_it == covered.end())
            continue;
          for (auto it = dep.task_pairs.begin(); it != dep.task_pairs.end();) {
            if (covered_it->second.count(*it)) {
              it = dep.task_pairs.erase(it);
            } else {
              ++it;
            }
          }
        }
      }

      for (PendingDependency &dep : pending) {
        if (dep.task_pairs.empty())
          continue;
        deps.push_back({nodes[i], nodes[j], std::move(dep.storage),
                        std::vector<DependencyTaskPair>(dep.task_pairs.begin(),
                                                        dep.task_pairs.end()),
                        dep.distance, dep.kind});
      }
    }
  }
  if (loop)
    cache.emplace(loop, deps);
  return deps;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
