/*!
 * \file annotate_multi_buffer_eligible.cc
 * \brief Scheduled-TIR analysis: for each For loop, decide which on-chip
 *        buffers can be multi-buffered and record the set on the For's
 *        annotations under "multi_buffer_eligible".
 *
 *  Runs after NormalizeControlFlowForSchedule and MaterializeScheduleUnits,
 *  then decodes the shared IRStructure. Buffer-dependent loop bounds are
 *  therefore explicit tasks, while manual T.Stage requests are available when
 *  recovering the logical write-before-read order. Materialization flattens
 *  IfThenElse into guarded sibling nodes, so write-first analysis reasons
 *  about guard equivalence and implication instead of relying on nested
 *  statement shape.
 *
 *  Automatic claims form the deepest disjoint loop frontier that completely
 *  covers a storage's ordinary accesses; owner-external fills may initialize
 *  every physical version. Every claimed loop must write the storage before
 *  any read. If descendant loops cover only part of an epoch (for example, a
 *  row-writing loop followed by a whole-buffer consumer), the claim is
 *  promoted to a write-first ancestor. Independent sibling loops may
 *  therefore become multiple owners of one storage.
 *
 *  Frontend may pre-set the annotation. A storage named by any explicit claim
 *  is excluded from automatic owner inference across the whole kernel; the
 *  explicit claim is preserved unless the storage is already manually
 *  versioned or pinned to one version without an explicit mode. Other buffers
 *  proven eligible are added automatically.
 *  T.annotate_buffer_versions({buf: 1}) removes the storage from eligibility;
 *  (1, mode) retains eligibility.
 */
#include "ascend/transform/source_scopes.h"
#include "ir/loop_annotations.h"
#include "ir/stmt_functor.h"
#include "transform/common/constr_visitor.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/multi_buffer.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/buffer_version.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using namespace ffi;
using namespace prim;

namespace {

using StorageSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using ControlStorageClaims =
    std::unordered_map<ControlNode *, std::vector<Var>>;

Var CanonicalStorage(const Var &storage, const StorageProvenance &context) {
  auto address = context.Address(storage);
  return address.exact && address.root.as<VarObj>()
             ? address.root.as_or_throw<Var>()
             : storage;
}

template <typename T>
Map<Var, T> CanonicalVersions(const Map<Var, T> &versions,
                              const StorageProvenance &context) {
  Map<Var, T> result;
  for (const auto &[storage, value] : versions) {
    Var root = CanonicalStorage(storage, context);
    if (auto prior = result.Get(root))
      TVM_FFI_CHECK(*prior == value, ValueError)
          << "Conflicting buffer version policies for native aliases of "
          << root;
    result.Set(root, value);
  }
  return result;
}

Array<Var> NormalizeEligibleStorages(const Any &annotation,
                                     const StorageProvenance &context) {
  Array<Var> result;
  StorageSet seen;
  for (const Any &item : annotation.cast<Array<Any>>()) {
    Var storage = item.cast<Var>();
    TVM_FFI_CHECK(storage->ty.as<TensorTypeObj>() ||
                      storage->ty.as<PtrTypeObj>(),
                  ValueError)
        << "multi_buffer_eligible entries must name native tensor or pointer "
           "storage";
    storage = CanonicalStorage(storage, context);
    if (seen.insert(storage).second)
      result.push_back(storage);
  }
  return result;
}

// ---------------------------------------------------------------------------
// WriteFirstClassifier: program-order classification of a storage's access
// pattern inside a Stmt. All Buffer aliases sharing buffer->data participate.
//
// Four-state lattice:
//   kUntouched         — subtree provably never accesses the buffer
//   kMaybeWriteFirst   — either kUntouched or kWriteFirst (write not certain
//                        but no read-before-write risk)
//   kWriteFirst        — first access is a guaranteed write
//   kReadFirst         — a read may happen before any guaranteed write
//                        (absorbing/unsafe state)
//
// Both kUntouched and kMaybeWriteFirst and kWriteFirst are "safe" for
// multi-buffering (prior content is irrelevant). kReadFirst is not. This is
// deliberately a storage-granular heuristic: a write to any region is treated
// as making the storage write-first; it does not prove that every later-read
// region was overwritten. Kernels carrying untouched regions across iterations
// must opt out via T.annotate_buffer_versions({buf: 1}) or explicitly choose an
// owner whose epoch overwrites the complete read footprint.
// ---------------------------------------------------------------------------
enum class AccessOrder {
  kUntouched,
  kMaybeWriteFirst,
  kWriteFirst,
  kReadFirst,
};

// Terminal states determine the answer; further sub-events cannot change it.
bool IsTerminal(AccessOrder order) {
  return order == AccessOrder::kReadFirst || order == AccessOrder::kWriteFirst;
}

// Sequential composition of two sub-events in program order. Terminal states
// absorb the suffix; a conditional write followed by a read remains read-first
// because the write may have been skipped.
AccessOrder SeqCompose(AccessOrder first, AccessOrder second) {
  if (IsTerminal(first))
    return first;
  if (first == AccessOrder::kUntouched)
    return second;
  if (second == AccessOrder::kUntouched)
    return AccessOrder::kMaybeWriteFirst;
  return second;
}

// Merge mutually exclusive then/else branch classifications.
AccessOrder MergeIfThenElse(AccessOrder then_order, AccessOrder else_order) {
  if (then_order == AccessOrder::kReadFirst ||
      else_order == AccessOrder::kReadFirst) {
    return AccessOrder::kReadFirst;
  }
  if (then_order == AccessOrder::kUntouched &&
      else_order == AccessOrder::kUntouched) {
    return AccessOrder::kUntouched;
  }
  if (then_order == AccessOrder::kWriteFirst &&
      else_order == AccessOrder::kWriteFirst) {
    return AccessOrder::kWriteFirst;
  }
  return AccessOrder::kMaybeWriteFirst;
}

// Merge a single then branch with an implicit untouched else branch.
AccessOrder MergeIfThenOnly(AccessOrder then_order) {
  if (then_order == AccessOrder::kReadFirst)
    return AccessOrder::kReadFirst;
  if (then_order == AccessOrder::kUntouched)
    return AccessOrder::kUntouched;
  return AccessOrder::kMaybeWriteFirst;
}

// A possibly skipped scope cannot provide a guaranteed first write.
AccessOrder DemoteIfNotMustExecute(AccessOrder order, bool must_execute) {
  if (!must_execute && order == AccessOrder::kWriteFirst)
    return AccessOrder::kMaybeWriteFirst;
  return order;
}

AccessOrder AssumeForExecutes(AccessOrder body) {
  // NOTE: intentionally aggressive. Treat every For as executing at least
  // once, and assume a maybe-write-first body takes a writing path in some
  // iteration. A zero-trip or fully skipped loop may therefore make a later
  // read appear to have a preceding write.
  return body == AccessOrder::kMaybeWriteFirst ? AccessOrder::kWriteFirst
                                               : body;
}

class WriteFirstClassifier : public StmtExprVisitor {
public:
  using Result = Optional<VisitInterrupt>;
  explicit WriteFirstClassifier(StorageProvenance storage)
      : storage_(std::move(storage)) {}
  AccessOrder Classify(const Stmt &stmt, const Var &storage) {
    target_ = storage;
    result_ = AccessOrder::kUntouched;
    Visit(stmt);
    return result_;
  }
  AccessOrder Classify(const Expr &expr, const Var &storage) {
    target_ = storage;
    result_ = AccessOrder::kUntouched;
    Visit(expr);
    return result_;
  }

private:
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  Var target_{UnsafeInit{}};
  AccessOrder result_{AccessOrder::kUntouched};
  template <typename F> AccessOrder Local(F function) {
    auto old = result_;
    result_ = AccessOrder::kUntouched;
    function();
    auto result = result_;
    result_ = old;
    return result;
  }
  bool Touches(const Expr &value) const {
    auto address = storage_.Address(value), target = storage_.Address(target_);
    if (address.exact && target.exact)
      return address.root.same_as(target.root);
    // Opaque pointer producers cannot justify a write-first or no-read claim.
    return value->ty.as<TensorTypeObj>() || value->ty.as<PtrTypeObj>();
  }
  void Access(const Expr &value, int mask) {
    if (IsTerminal(result_) || !Touches(value))
      return;
    if (mask & kAccessRead)
      result_ = AccessOrder::kReadFirst;
    else if (mask & kAccessWrite)
      result_ = AccessOrder::kWriteFirst;
  }
  void Bounds(const TensorRegion &region) {
    for (const auto &range : region->region) {
      Visit(range->begin);
      Visit(range->extent);
    }
    Visit(region->source->ty);
  }
  Result Visit_(const VarObj *op) final {
    // A bare native tensor/pointer consumed by an unknown expression can keep
    // its contents live. Definitions, known address ops and metadata skip it.
    Access(GetRef<Var>(op), kAccessRead);
    return std::nullopt;
  }
  Result Visit_(const TensorLoadObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    for (const auto &index : op->indices)
      Visit(index);
    Visit(op->source->ty);
    Access(op->source, kAccessRead);
    return std::nullopt;
  }
  Result Visit_(const TensorStoreObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    Visit(op->value);
    for (const auto &index : op->indices)
      Visit(index);
    Visit(op->dest->ty);
    Access(op->dest, kAccessWrite);
    return std::nullopt;
  }
  Result Visit_(const SeqStmtObj *op) final {
    for (const auto &stmt : op->seq) {
      if (IsTerminal(result_))
        break;
      Visit(stmt);
    }
    return std::nullopt;
  }
  Result Visit_(const BindObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    Expr value = op->value;
    if (const auto *call = value.as<CallObj>(); IsScheduledBind(call))
      value = ReadScheduledBinding(call).value;
    const auto *call = value.as<CallObj>();
    if (call && (call->op.same_as(alloc_tensor_op()) ||
                 call->op.same_as(decl_tensor_op()))) {
      Visit(op->var->ty);
      // Pointer computation is not a read of the storage being described.
      if (call->op.same_as(decl_tensor_op()))
        AddressOperands(call->args[0]);
      return std::nullopt;
    }
    return Visit(value);
  }
  Result Visit_(const ForObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    Visit(op->begin);
    Visit(op->end);
    Visit(op->step);
    if (IsTerminal(result_))
      return std::nullopt;
    auto body = Local([&] { Visit(op->body); });
    result_ = SeqCompose(result_, AssumeForExecutes(body));
    return std::nullopt;
  }
  Result Visit_(const IfObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    Visit(op->condition);
    if (IsTerminal(result_))
      return std::nullopt;
    auto yes = Local([&] { Visit(op->then_case); });
    auto branch =
        op->else_case
            ? MergeIfThenElse(yes, Local([&] { Visit(*op->else_case); }))
            : MergeIfThenOnly(yes);
    result_ = SeqCompose(result_, branch);
    return std::nullopt;
  }
  Result Visit_(const WhileObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    Visit(op->condition);
    if (!IsTerminal(result_))
      result_ = SeqCompose(
          result_,
          DemoteIfNotMustExecute(Local([&] { Visit(op->body); }), false));
    return std::nullopt;
  }
  Result Visit_(const RegionStmtObj *op) final {
    if (IsTerminal(result_))
      return std::nullopt;
    if (IsAscendBlock(op) || IsSemanticTask(op) || IsScheduleUnit(op) ||
        IsStageScope(op))
      return Visit(op->body);
    if (auto attr = ReadSourceAttribute(op)) {
      if (CanRewriteMultiBufferAttrNode(attr->key, attr->node))
        Visit(attr->node.cast<Expr>());
      if (!IsTerminal(result_))
        Visit(op->body);
      return std::nullopt;
    }
    for (const auto &arg : op->args)
      Visit(arg);
    if (!IsTerminal(result_))
      Visit(op->body);
    return std::nullopt;
  }
  void AddressOperands(const Expr &value) {
    if (value.as<VarObj>())
      return;
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tensor_data_ptr_op())) {
        Visit(call->args[0]->ty);
        return;
      }
      if (call->op.same_as(reinterpret_op())) {
        AddressOperands(call->args[0]);
        return;
      }
      if (call->op.same_as(ptr_byte_offset_op())) {
        AddressOperands(call->args[0]);
        Visit(call->args[1]);
        return;
      }
    }
    Visit(value);
  }
  Result Visit_(const CallObj *op) final {
    if (IsTerminal(result_) || IsAnnotationCall(op))
      return std::nullopt;
    if (IsScheduledBind(op))
      return Visit(ReadScheduledBinding(op).value);
    if (op->op.same_as(tl::access_ptr())) {
      const auto *load = op->args[0].as<TensorLoadObj>();
      TVM_FFI_CHECK(load && op->args.size() == 3, ValueError);
      for (const auto &index : load->indices)
        Visit(index);
      Visit(op->args[1]);
      Visit(op->args[2]);
      Visit(load->source->ty);
      Access(load->source,
             GetConservativeAccessMask(op->args[2].as_or_throw<PrimExpr>()));
      return std::nullopt;
    }
    if (op->op.same_as(tl::ptr_access())) {
      TVM_FFI_CHECK_EQ(op->args.size(), 4U, ValueError);
      AddressOperands(op->args[0]);
      for (size_t i = 1; i < op->args.size(); ++i)
        Visit(op->args[i]);
      Access(op->args[0],
             GetConservativeAccessMask(op->args[3].as_or_throw<PrimExpr>()));
      return std::nullopt;
    }
    if (auto tile = ParseOperator(GetRef<Call>(op)); tile.defined()) {
      auto access = tile->GetAccessRegions();
      for (const auto &region : access.reads)
        Bounds(region);
      for (const auto &region : access.writes)
        Bounds(region);
      // Only scalar/value operands execute as ordinary expressions. The
      // tensor regions themselves are addresses described by the op effects.
      for (const auto &arg : DecodeTileOpArguments(op).operands)
        if (!IsTensorLikeExpr(arg))
          Visit(arg);
      for (const auto &region : access.reads)
        Access(region->source, kAccessRead);
      for (const auto &region : access.writes)
        Access(region->source, kAccessWrite);
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }
};

int RequestedStage(const IRStructure *node) {
  int stage = node->GetStage();
  return stage == kUnscheduledStage ? 0 : stage;
}

std::vector<const IRStructure *>
GetWriteFirstOrder(const std::vector<std::shared_ptr<IRStructure>> &nodes) {
  std::vector<const IRStructure *> result;
  result.reserve(nodes.size());
  bool has_requested_stage = false;
  for (const auto &node : nodes) {
    result.push_back(node.get());
    has_requested_stage |= node->GetStage() != kUnscheduledStage;
  }
  if (has_requested_stage) {
    std::stable_sort(result.begin(), result.end(),
                     [](const IRStructure *lhs, const IRStructure *rhs) {
                       return RequestedStage(lhs) < RequestedStage(rhs);
                     });
  }
  return result;
}

// MaterializeScheduleUnits turns structured branches into guarded siblings.
// Track the condition under which an earlier guaranteed write has occurred,
// and prove that every later read executes only inside that condition. This
// recovers both equal guards and exhaustive opposite guards without rebuilding
// an IfThenElse tree.
class IRWriteFirstClassifier {
public:
  static AccessOrder
  Classify(const std::vector<std::shared_ptr<IRStructure>> &nodes,
           const Var &storage, const ConstrSet &outer_ctx) {
    State state;
    for (const IRStructure *node : GetWriteFirstOrder(nodes)) {
      ProcessNode(node, storage, outer_ctx, &state);
      if (state.unsafe)
        return AccessOrder::kReadFirst;
    }
    if (!state.touched)
      return AccessOrder::kUntouched;
    return IsCovered(IntImm::Bool(true), state.written_guard, outer_ctx)
               ? AccessOrder::kWriteFirst
               : AccessOrder::kMaybeWriteFirst;
  }

private:
  struct State {
    bool touched{false};
    bool unsafe{false};
    PrimExpr written_guard{IntImm::Bool(false)};
  };

  static bool IsCovered(const PrimExpr &access_guard,
                        const PrimExpr &written_guard,
                        const ConstrSet &outer_ctx) {
    return GuardsEquivalent(access_guard, written_guard, outer_ctx) ||
           GuardImplies(access_guard, written_guard, outer_ctx);
  }

  static PrimExpr MergeWriteGuard(const PrimExpr &known, const PrimExpr &added,
                                  const ConstrSet &outer_ctx) {
    if (IsZero(known))
      return added;
    if (IsZero(added) || GuardsEquivalent(known, added, outer_ctx) ||
        GuardImplies(added, known, outer_ctx)) {
      return known;
    }
    if (GuardImplies(known, added, outer_ctx))
      return added;
    return known || added;
  }

  static void ProcessAccess(AccessOrder order, const PrimExpr &active_guard,
                            const ConstrSet &outer_ctx, State *state) {
    ICHECK(state != nullptr);
    if (order == AccessOrder::kUntouched || state->unsafe)
      return;
    state->touched = true;
    if (order == AccessOrder::kReadFirst) {
      state->unsafe = !IsCovered(active_guard, state->written_guard, outer_ctx);
      return;
    }
    if (order == AccessOrder::kWriteFirst) {
      state->written_guard =
          MergeWriteGuard(state->written_guard, active_guard, outer_ctx);
    }
  }

  static AccessOrder ClassifyPayload(const IRStructure *node,
                                     const Var &storage) {
    auto classifier = make_object<WriteFirstClassifier>(node->StorageContext());
    if (node->IsTask()) {
      return classifier->Classify(static_cast<const TaskNode *>(node)->stmt,
                                  storage);
    }

    TVM_FFI_ICHECK(node->IsControl());
    const auto *control = static_cast<const ControlNode *>(node);
    AccessOrder header = classifier->Classify(control->task->stmt, storage);
    if (header == AccessOrder::kReadFirst ||
        header == AccessOrder::kWriteFirst) {
      return header;
    }

    AccessOrder body =
        Classify(control->children, storage, control->GetLoopBodyContext());
    return SeqCompose(header, AssumeForExecutes(body));
  }

  static void ProcessNode(const IRStructure *node, const Var &storage,
                          const ConstrSet &outer_ctx, State *state,
                          PrimExpr active_guard = IntImm::Bool(true)) {
    auto classifier = make_object<WriteFirstClassifier>(node->StorageContext());
    for (const auto &guard : node->GetGuards()) {
      if (guard->IsCondition()) {
        const PrimExpr &condition =
            static_cast<const ConditionGuard *>(guard.get())->condition;
        ProcessAccess(classifier->Classify(condition, storage), active_guard,
                      outer_ctx, state);
        active_guard =
            IsOne(active_guard) ? condition : active_guard && condition;
        continue;
      }

      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto node_expr = attribute->node.try_cast<PrimExpr>()) {
        ProcessAccess(classifier->Classify(node_expr.value(), storage),
                      active_guard, outer_ctx, state);
      }
    }
    if (node->IsScope()) {
      const auto *scope = static_cast<const ScopeNode *>(node);
      if (scope->task)
        ProcessAccess(classifier->Classify(scope->task->stmt, storage),
                      active_guard, outer_ctx, state);
      if (state->unsafe)
        return;
      PrimExpr condition = IntImm::Bool(true);
      bool branch = scope->shell.as<IfObj>() != nullptr;
      if (const auto *statement = scope->shell.as<IfObj>()) {
        condition = statement->condition;
        // Two separate evaluations of a mutable condition are not the same
        // guard. The branches of this one scope share a fresh proof value.
        if (SideEffect(condition) > CallEffectKind::kPure)
          condition = PrimVar("eligibility_condition", PrimType::Bool());
      }
      for (const IRStructure *child : GetWriteFirstOrder(scope->children)) {
        ProcessNode(child, storage, outer_ctx, state,
                    branch ? active_guard && condition : active_guard);
        if (state->unsafe)
          return;
      }
      for (const IRStructure *child :
           GetWriteFirstOrder(scope->else_children)) {
        ProcessNode(child, storage, outer_ctx, state,
                    active_guard && !condition);
        if (state->unsafe)
          return;
      }
      return;
    }
    ProcessAccess(ClassifyPayload(node, storage), active_guard, outer_ctx,
                  state);
  }
};

// ---------------------------------------------------------------------------
// MultiBufferOwnerPlanner: for each storage, choose the deepest set of
// disjoint structural loops that covers every ordinary access. Independently
// rewritable fills may remain outside the frontier. A loop is selected only
// when its descendants do not already provide complete coverage and its
// stage-ordered body is write-first.
// ---------------------------------------------------------------------------
class MultiBufferOwnerPlanner {
public:
  static ControlStorageClaims
  Plan(const std::vector<std::shared_ptr<IRStructure>> &root,
       const StorageSet &excluded_storages, const L0StorageGroups &groups) {
    MultiBufferOwnerPlanner planner;
    StorageSet seen;
    for (const auto &node : root) {
      for (const Var &storage : node->GetOnChipStorages()) {
        Var representative = groups.Representative(storage);
        if (!seen.insert(representative).second ||
            excluded_storages.count(representative))
          continue;
        Array<Var> members = groups.Members(storage);
        CoveragePlan plan = planner.PlanList(root, members);
        if (!plan.IsComplete())
          continue;
        for (ControlNode *owner : plan.owners)
          for (const Var &member : members)
            planner.claims_[owner].push_back(member);
      }
    }
    return std::move(planner.claims_);
  }

private:
  // Ordered from least to most restrictive; sibling composition takes max.
  enum class Coverage {
    kUntouched,
    kCovered,
    kNeedsOwner,
  };

  struct CoveragePlan {
    Coverage coverage{Coverage::kUntouched};
    std::vector<ControlNode *> owners;

    bool IsComplete() const {
      return coverage == Coverage::kUntouched || coverage == Coverage::kCovered;
    }

    bool CanClaimHere() const {
      // An ownerless covered subtree contains only independently broadcastable
      // fills. Keep those fills outside the owner frontier so they initialize
      // every physical version instead of advancing one version per loop
      // iteration.
      return coverage == Coverage::kNeedsOwner;
    }

    void AddUncoveredAccess() { coverage = Coverage::kNeedsOwner; }

    void Merge(const CoveragePlan &other) {
      coverage = std::max(coverage, other.coverage);
      owners.insert(owners.end(), other.owners.begin(), other.owners.end());
    }
  };

  ControlStorageClaims claims_;

  static CoveragePlan PlanLeaf(const TaskNode *task,
                               const Array<Var> &members) {
    CoveragePlan result;
    for (const Var &storage : members) {
      if (!task->TouchesStorage(storage))
        continue;
      bool broadcast_fill = members.size() == 1 &&
                            !task->GuardsTouchStorage(storage) &&
                            CanBroadcastFillToStorage(task->stmt, storage,
                                                      task->StorageContext());
      if (!broadcast_fill)
        result.AddUncoveredAccess();
      else if (result.coverage == Coverage::kUntouched)
        result.coverage = Coverage::kCovered;
    }
    return result;
  }

  CoveragePlan PlanList(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                        const Array<Var> &members) {
    CoveragePlan result;
    for (const auto &node : nodes)
      result.Merge(PlanNode(node.get(), members));
    return result;
  }

  CoveragePlan PlanNode(IRStructure *node, const Array<Var> &members) {
    if (node->IsTask())
      return PlanLeaf(static_cast<const TaskNode *>(node), members);

    if (node->IsScope()) {
      auto *scope = static_cast<ScopeNode *>(node);
      CoveragePlan result;
      if (scope->task)
        result.Merge(PlanLeaf(scope->task.get(), members));
      for (const auto *children : scope->ChildLists())
        result.Merge(PlanList(*children, members));
      return result;
    }

    TVM_FFI_ICHECK(node->IsControl());
    auto *control = static_cast<ControlNode *>(node);
    CoveragePlan result = PlanList(control->children, members);
    if (result.CanClaimHere()) {
      // Each plane must be write-first independently. In particular, a data
      // reload does not initialize or kill sticky SF contents.
      bool write_first = true;
      for (const Var &storage : members) {
        AccessOrder order = IRWriteFirstClassifier::Classify(
            control->children, storage, control->GetLoopBodyContext());
        write_first &= order != AccessOrder::kReadFirst;
      }
      if (write_first) {
        result.coverage = Coverage::kCovered;
        result.owners = {control};
      }
    }
    for (const Var &storage : members) {
      if (control->task->TouchesStorage(storage))
        result.AddUncoveredAccess();
    }
    return result;
  }
};

// Normalize explicit claims and add the automatic owner frontier directly to
// the decoded ControlNodes. Encoding preserves schedule-unit stages and guards.
struct LoopEligibilityPlan {
  Var variable;
  Array<Var> eligible;
};

class MultiBufferAnnotator {
public:
  static std::vector<LoopEligibilityPlan>
  Rewrite(ScheduledTIR *scheduled_tir, StorageSet manual_buffers,
          const L0StorageGroups &groups) {
    ICHECK(scheduled_tir != nullptr);
    // This pass precedes AutoSchedule: these counts are frontend overrides,
    // not solver-selected versions. Only a bare frontend 1 means opt-out.
    StorageSet single_version_buffers;
    StorageSet excluded_storages = manual_buffers;
    for (const auto &[storage, versions] :
         scheduled_tir->metadata.buffer_versions) {
      if (versions == 1 &&
          !scheduled_tir->metadata.buffer_version_modes.count(storage)) {
        single_version_buffers.insert(storage);
        excluded_storages.insert(storage);
      }
    }
    CollectExplicitStorages(scheduled_tir->tree, &excluded_storages);
    StorageSet expanded;
    for (const Var &storage : excluded_storages)
      for (const Var &member : groups.Members(storage))
        expanded.insert(member);
    excluded_storages = std::move(expanded);
    ControlStorageClaims claims = MultiBufferOwnerPlanner::Plan(
        scheduled_tir->tree, excluded_storages, groups);
    MultiBufferAnnotator annotator(std::move(claims), std::move(manual_buffers),
                                   std::move(single_version_buffers), groups);
    annotator.RewriteNodes(scheduled_tir->tree);
    return std::move(annotator.plans_);
  }

private:
  MultiBufferAnnotator(ControlStorageClaims claims, StorageSet manual_buffers,
                       StorageSet single_version_buffers,
                       L0StorageGroups groups)
      : claims_(std::move(claims)), groups_(std::move(groups)),
        manual_buffers_(std::move(manual_buffers)),
        single_version_buffers_(std::move(single_version_buffers)) {}

  ControlStorageClaims claims_;
  L0StorageGroups groups_;
  StorageSet manual_buffers_;
  StorageSet single_version_buffers_;
  StorageSet warned_excluded_storages_;
  std::vector<LoopEligibilityPlan> plans_;

  static void CollectExplicitStorages(
      const std::vector<std::shared_ptr<IRStructure>> &nodes,
      StorageSet *storages) {
    ICHECK(storages != nullptr);
    for (const auto &node : nodes) {
      if (node->IsControl()) {
        const auto *control = static_cast<const ControlNode *>(node.get());
        if (auto annotation = ReadLoopAnnotations(control->control.get())
                                  .Get(kMultiBufferEligible)) {
          for (const Var &storage : NormalizeEligibleStorages(
                   annotation.value(), control->StorageContext())) {
            storages->insert(storage);
          }
        }
      }
      for (const auto *children : node->ChildLists())
        CollectExplicitStorages(*children, storages);
    }
  }

  void RewriteNodes(std::vector<std::shared_ptr<IRStructure>> &nodes) {
    for (const auto &node : nodes) {
      if (node->IsControl())
        RewriteControl(static_cast<ControlNode *>(node.get()));
      for (auto *children : node->ChildLists())
        RewriteNodes(*children);
    }
  }

  void RewriteControl(ControlNode *control) {
    For loop = control->control;
    Array<Var> eligible;
    StorageSet already;
    if (auto annotation =
            ReadLoopAnnotations(loop.get()).Get(kMultiBufferEligible)) {
      for (const Var &storage : NormalizeEligibleStorages(
               annotation.value(), control->StorageContext())) {
        if (manual_buffers_.count(storage)) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING) << "Ignoring explicit '" << kMultiBufferEligible
                         << "' claim for storage " << storage->name
                         << " because it is already manually multi-buffered";
          }
          continue;
        }
        if (single_version_buffers_.count(storage)) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING) << "Ignoring explicit '" << kMultiBufferEligible
                         << "' claim for storage " << storage->name
                         << " because T.annotate_buffer_versions({buf: 1}) "
                            "disables multi-buffer eligibility; use "
                            "{buf: (1, \"auto\")} to retain eligibility with "
                            "one version";
          }
          continue;
        }
        for (const Var &member : groups_.Members(storage))
          if (already.insert(member).second)
            eligible.push_back(member);
      }
    }

    auto planned = claims_.find(control);
    if (planned != claims_.end()) {
      for (const Var &storage : planned->second) {
        for (const Var &member : groups_.Members(storage))
          if (already.insert(member).second)
            eligible.push_back(member);
      }
    }

    plans_.push_back({loop->var, std::move(eligible)});
  }
};

// The schedule encoder rebuilds a control's body from its child nodes. Apply
// native annotation edits to that final body, otherwise old marker children
// would overwrite them. Match the scheduled-control traversal, including
// lexical branches; atomic task-internal loops are not scheduled controls.
// This also distinguishes repeated occurrences with the same Var identity.
class EmptyEligibilityCarrierRemover : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (IsAnnotationCall(call) && ReadAnnotations(call).empty())
      return SeqStmt({});
    return StmtExprMutator::Mutate_(op, mode);
  }
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    SeqStmt sequence = StmtExprMutator::Mutate_(op, mode)
                           .ValueOrUnchanged(GetRef<SeqStmt>(op))
                           .as_or_throw<SeqStmt>();
    Array<Stmt> statements;
    for (const Stmt &statement : sequence->seq) {
      const auto *empty = statement.as<SeqStmtObj>();
      if (!empty || !empty->seq.empty())
        statements.push_back(statement);
    }
    return statements.size() == sequence->seq.size()
               ? Stmt(sequence)
               : Stmt(SeqStmt(statements, sequence->loc));
  }
};

class EligibilityCarrierWriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit EligibilityCarrierWriter(std::vector<LoopEligibilityPlan> plans)
      : plans_(std::move(plans)) {}

  SeqStmt Run(const SeqStmt &root) {
    SeqStmt result = Mutate(root, InplaceMode::kDisallow)
                         .ValueOrUnchanged(root)
                         .as_or_throw<SeqStmt>();
    TVM_FFI_ICHECK_EQ(cursor_, plans_.size());
    return result;
  }

private:
  std::vector<LoopEligibilityPlan> plans_;
  size_t cursor_{0};

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (IsSemanticTask(op))
      return Unchanged();
    const auto *loop =
        IsScheduleUnit(op) ? SingleStatement(op->body).as<ForObj>() : nullptr;
    if (!loop)
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_ICHECK_LT(cursor_, plans_.size());
    const size_t index = cursor_++;
    TVM_FFI_ICHECK(loop->var.same_as(plans_[index].variable));
    RegionStmt region = StmtExprMutator::Mutate_(op, mode)
                            .ValueOrUnchanged(GetRef<RegionStmt>(op))
                            .as_or_throw<RegionStmt>();
    For updated = SingleStatement(region->body).as_or_throw<For>();
    updated =
        RewriteLoopAnnotationCarriers(updated, [](Map<String, Any> fields) {
          fields.erase(kMultiBufferEligible);
          return fields;
        });
    updated.CopyOnWrite()->annotations.erase(kMultiBufferEligible);
    auto remover = make_object<EmptyEligibilityCarrierRemover>();
    auto body = updated->body;
    updated.CopyOnWrite()->body =
        remover->Mutate(body, mode).ValueOrUnchanged(body);
    updated = AppendLoopAnnotation(
        updated, {{kMultiBufferEligible, plans_[index].eligible}});
    region.CopyOnWrite()->body = SeqStmt(updated);
    return region;
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](tirx::Function f, const IRModule &,
                       const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "AnnotateMultiBufferEligible",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.body, context.outer_ctx, nullptr,
              context.kernel_annotations, context.parameter_alias_contract);
          StorageProvenance storage = MakeAscendStorageProvenance();
          storage.Collect(context.body);
          L0SFBindings bindings;
          for (const auto &[sf, data] : CollectL0SFBindings(context.body)) {
            Var scale_root = CanonicalStorage(sf, storage);
            Var data_root = CanonicalStorage(data, storage);
            if (auto previous = bindings.Get(scale_root))
              TVM_FFI_CHECK(previous->same_as(data_root), ValueError)
                  << "Conflicting data owners for an aliased L0 scale tensor";
            bindings.Set(scale_root, data_root);
          }
          L0StorageGroups groups(bindings);
          auto &metadata = scheduled_tir.metadata;
          metadata.buffer_versions = ExpandL0StorageGroupValues(
              CanonicalVersions(metadata.buffer_versions, storage), groups,
              "version counts");
          metadata.manual_buffer_versions = ExpandL0StorageGroupValues(
              CanonicalVersions(metadata.manual_buffer_versions, storage),
              groups, "manual version counts");
          metadata.buffer_version_modes = ExpandL0StorageGroupValues(
              CanonicalVersions(metadata.buffer_version_modes, storage), groups,
              "version modes");
          StorageSet manual_buffers;
          for (const auto &[storage, _] :
               scheduled_tir.metadata.manual_buffer_versions) {
            manual_buffers.insert(storage);
          }
          auto plans = MultiBufferAnnotator::Rewrite(
              &scheduled_tir, std::move(manual_buffers), groups);
          SeqStmt encoded = EncodeScheduledTIR(std::move(scheduled_tir));
          return make_object<EligibilityCarrierWriter>(std::move(plans))
              ->Run(encoded);
        },
        /*require_kernel=*/false);
  };
  return CreateFunctionPass(pass_func, 0, "tl.AnnotateMultiBufferEligible");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMultiBufferEligible",
                        AnnotateMultiBufferEligible);
}

} // namespace tl
} // namespace tvm
