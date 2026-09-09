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
 *  versioned. Other buffers proven eligible are added automatically. A buffer
 *  the user does NOT want multi-buffered can be pinned to a single version via
 *  T.annotate_buffer_versions({buf: 1}) instead of being omitted here.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
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
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using StorageSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using ControlStorageClaims =
    std::unordered_map<ControlNode *, std::vector<Var>>;

Array<Var> NormalizeEligibleStorages(const Any &annotation) {
  Array<Var> result;
  StorageSet seen;
  for (const Any &item : Downcast<Array<Any>>(annotation)) {
    Var storage;
    if (auto buffer = item.try_cast<Buffer>()) {
      storage = buffer.value()->data;
    } else if (auto var = item.try_cast<Var>()) {
      storage = var.value();
    } else {
      LOG(FATAL) << "'" << kMultiBufferEligible
                 << "' entries must be Buffer or Var objects, got " << item;
    }
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
// must pin the storage to one version or explicitly choose an owner whose epoch
// overwrites the complete read footprint.
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
  AccessOrder Classify(const Stmt &s, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(s);
    return result_;
  }

  AccessOrder Classify(const PrimExpr &e, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitExpr(e);
    return result_;
  }

private:
  Var target_storage_;
  AccessOrder result_ = AccessOrder::kUntouched;

  // Dispatch short-circuit: skip further work once a terminal is reached.
  void VisitStmt(const Stmt &s) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitStmt(s);
  }
  void VisitExpr(const PrimExpr &e) final {
    if (IsTerminal(result_))
      return;
    StmtExprVisitor::VisitExpr(e);
  }

  template <typename F> AccessOrder ClassifyLocal(F f) {
    AccessOrder saved = result_;
    result_ = AccessOrder::kUntouched;
    f();
    AccessOrder local = result_;
    result_ = saved;
    return local;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(op->buffer->data))
      result_ = AccessOrder::kReadFirst;
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // value first, then indices, then the store itself.
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    for (const auto &idx : op->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(op->buffer->data))
      result_ = AccessOrder::kWriteFirst;
  }

  void VisitStmt_(const SeqStmtNode *op) final {
    for (const auto &s : op->seq) {
      VisitStmt(s);
      if (IsTerminal(result_))
        return;
    }
  }

  void VisitStmt_(const EvaluateNode *op) final { VisitExpr(op->value); }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    if (IsTerminal(result_))
      return;
    VisitExpr(op->extent);
    if (IsTerminal(result_))
      return;
    if (op->step.has_value()) {
      VisitExpr(op->step.value());
      if (IsTerminal(result_))
        return;
    }
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_, AssumeForExecutes(body));
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto t = ClassifyLocal([&] { VisitStmt(op->then_case); });
    AccessOrder branch;
    if (op->else_case) {
      auto e = ClassifyLocal([&] { VisitStmt(op->else_case.value()); });
      branch = MergeIfThenElse(t, e);
    } else {
      branch = MergeIfThenOnly(t);
    }
    result_ = SeqCompose(result_, branch);
  }

  void VisitStmt_(const BindNode *op) final { VisitExpr(op->value); }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (CanRewriteMultiBufferAttrNode(op->attr_key, op->node)) {
      VisitExpr(Downcast<PrimExpr>(op->node));
      if (IsTerminal(result_))
        return;
    }
    VisitExpr(op->value);
    if (IsTerminal(result_))
      return;
    VisitStmt(op->body);
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    if (IsTerminal(result_))
      return;
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    result_ = SeqCompose(result_,
                         DemoteIfNotMustExecute(body, /*must_execute=*/false));
  }

  void VisitStmt_(const SBlockNode *op) final {
    // Reduce-init runs only on the first iteration of the surrounding loop;
    // its writes are not guaranteed across iterations -> kMaybeWriteFirst.
    if (op->init.defined()) {
      auto init = ClassifyLocal([&] { VisitStmt(op->init.value()); });
      AccessOrder init_eff = init;
      if (init == AccessOrder::kWriteFirst)
        init_eff = AccessOrder::kMaybeWriteFirst;
      result_ = SeqCompose(result_, init_eff);
      if (IsTerminal(result_))
        return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const SBlockRealizeNode *op) final {
    VisitExpr(op->predicate);
    if (IsTerminal(result_))
      return;
    bool must_execute = false;
    if (const auto *imm = op->predicate.as<IntImmNode>())
      must_execute = (imm->value != 0);
    auto body = ClassifyLocal([&] { VisitStmt(op->block); });
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_execute));
  }

  void VisitExpr_(const CallNode *op) final {
    static const Op &region_op = region();
    static const auto access_ptr_op = Op::Get("tl.access_ptr");

    if (op->op.same_as(region_op)) {
      HandleRegionCall(op);
      return;
    }
    if (op->op.same_as(access_ptr_op)) {
      HandleAccessPtrCall(op);
      return;
    }
    for (const auto &arg : op->args) {
      VisitExpr(arg);
      if (IsTerminal(result_))
        return;
    }
  }

  // tl.region(BufferLoad(buf, indices...), access_type, extents...)
  //   access_type: bit 0 = read, bit 1 = write.
  void HandleRegionCall(const CallNode *op) {
    if (op->args.size() < 2) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mode = op->args[1].as<IntImmNode>();
    if (!bl || !mode) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      VisitExpr(op->args[i]);
      if (IsTerminal(result_))
        return;
    }
    if (target_storage_.same_as(bl->buffer->data)) {
      int m = mode->value;
      if (m & 1)
        result_ = AccessOrder::kReadFirst;
      else if (m & 2)
        result_ = AccessOrder::kWriteFirst;
    }
  }

  // tl.access_ptr(BufferLoad(buf, indices...), extent, rw_mask)
  //   rw_mask: bit 0 = read, bit 1 = write.
  void HandleAccessPtrCall(const CallNode *op) {
    if (op->args.size() < 3) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    const auto *bl = op->args[0].as<BufferLoadNode>();
    const auto *mask = op->args[2].as<IntImmNode>();
    if (!bl || !mask) {
      for (const auto &arg : op->args) {
        VisitExpr(arg);
        if (IsTerminal(result_))
          return;
      }
      return;
    }
    for (const auto &idx : bl->indices) {
      VisitExpr(idx);
      if (IsTerminal(result_))
        return;
    }
    VisitExpr(op->args[1]);
    if (IsTerminal(result_))
      return;
    if (target_storage_.same_as(bl->buffer->data)) {
      int rw = mask->value;
      if (rw & 1)
        result_ = AccessOrder::kReadFirst;
      else if (rw & 2)
        result_ = AccessOrder::kWriteFirst;
    }
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
    return IsCovered(Bool(true), state.written_guard, outer_ctx)
               ? AccessOrder::kWriteFirst
               : AccessOrder::kMaybeWriteFirst;
  }

private:
  struct State {
    bool touched{false};
    bool unsafe{false};
    PrimExpr written_guard{Bool(false)};
  };

  static bool IsCovered(const PrimExpr &access_guard,
                        const PrimExpr &written_guard,
                        const ConstrSet &outer_ctx) {
    return GuardsEquivalent(access_guard, written_guard, outer_ctx) ||
           GuardImplies(access_guard, written_guard, outer_ctx);
  }

  static PrimExpr MergeWriteGuard(const PrimExpr &known, const PrimExpr &added,
                                  const ConstrSet &outer_ctx) {
    if (is_zero(known))
      return added;
    if (is_zero(added) || GuardsEquivalent(known, added, outer_ctx) ||
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
    WriteFirstClassifier classifier;
    if (node->IsTask()) {
      return classifier.Classify(static_cast<const TaskNode *>(node)->stmt,
                                 storage);
    }

    const auto *control = static_cast<const ControlNode *>(node);
    AccessOrder header = classifier.Classify(control->task->stmt, storage);
    if (header == AccessOrder::kReadFirst ||
        header == AccessOrder::kWriteFirst) {
      return header;
    }

    AccessOrder body =
        Classify(control->children, storage, control->GetLoopBodyContext());
    return SeqCompose(header, AssumeForExecutes(body));
  }

  static void ProcessNode(const IRStructure *node, const Var &storage,
                          const ConstrSet &outer_ctx, State *state) {
    PrimExpr active_guard = Bool(true);
    WriteFirstClassifier classifier;
    for (const auto &guard : node->GetGuards()) {
      if (guard->IsCondition()) {
        const PrimExpr &condition =
            static_cast<const ConditionGuard *>(guard.get())->condition;
        ProcessAccess(classifier.Classify(condition, storage), active_guard,
                      outer_ctx, state);
        active_guard =
            is_one(active_guard) ? condition : active_guard && condition;
        continue;
      }

      const auto *attribute = static_cast<const AttributeGuard *>(guard.get());
      if (auto node_expr = attribute->node.try_cast<PrimExpr>()) {
        ProcessAccess(classifier.Classify(node_expr.value(), storage),
                      active_guard, outer_ctx, state);
      }
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
       const StorageSet &excluded_storages) {
    MultiBufferOwnerPlanner planner;
    StorageSet seen;
    for (const auto &node : root) {
      for (const Var &storage : node->GetOnChipStorages()) {
        if (!seen.insert(storage).second || excluded_storages.count(storage))
          continue;
        CoveragePlan plan = planner.PlanList(root, storage);
        if (!plan.IsComplete())
          continue;
        for (ControlNode *owner : plan.owners)
          planner.claims_[owner].push_back(storage);
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

  static CoveragePlan PlanLeaf(const TaskNode *task, const Var &storage) {
    if (!task->TouchesStorage(storage))
      return {};
    CoveragePlan result;
    bool broadcast_fill = !task->GuardsTouchStorage(storage) &&
                          CanBroadcastFillToStorage(task->stmt, storage);
    result.coverage =
        broadcast_fill ? Coverage::kCovered : Coverage::kNeedsOwner;
    return result;
  }

  CoveragePlan PlanList(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                        const Var &storage) {
    CoveragePlan result;
    for (const auto &node : nodes)
      result.Merge(PlanNode(node.get(), storage));
    return result;
  }

  CoveragePlan PlanNode(IRStructure *node, const Var &storage) {
    if (node->IsTask()) {
      return PlanLeaf(static_cast<const TaskNode *>(node), storage);
    }

    auto *control = static_cast<ControlNode *>(node);
    CoveragePlan result = PlanList(control->children, storage);
    if (result.CanClaimHere()) {
      AccessOrder order = IRWriteFirstClassifier::Classify(
          control->children, storage, control->GetLoopBodyContext());
      if (order == AccessOrder::kWriteFirst ||
          order == AccessOrder::kMaybeWriteFirst) {
        result.coverage = Coverage::kCovered;
        result.owners = {control};
      }
    }

    // A ControlNode's task describes accesses that execute before entering
    // the loop body, including guards and loop-header expressions. They cannot
    // be covered by this loop's version epoch, but remain ordinary accesses
    // that an enclosing loop may cover.
    if (control->task->TouchesStorage(storage))
      result.AddUncoveredAccess();
    return result;
  }
};

// Normalize explicit claims and add the automatic owner frontier directly to
// the decoded ControlNodes. Encoding preserves schedule-unit stages and guards.
class MultiBufferAnnotator {
public:
  static void Rewrite(ScheduledTIR *scheduled_tir, StorageSet manual_buffers) {
    ICHECK(scheduled_tir != nullptr);
    StorageSet excluded_storages = manual_buffers;
    CollectExplicitStorages(scheduled_tir->tree, &excluded_storages);
    ControlStorageClaims claims =
        MultiBufferOwnerPlanner::Plan(scheduled_tir->tree, excluded_storages);
    MultiBufferAnnotator annotator(std::move(claims),
                                   std::move(manual_buffers));
    annotator.RewriteNodes(scheduled_tir->tree);
  }

private:
  MultiBufferAnnotator(ControlStorageClaims claims, StorageSet manual_buffers)
      : claims_(std::move(claims)), manual_buffers_(std::move(manual_buffers)) {
  }

  ControlStorageClaims claims_;
  StorageSet manual_buffers_;
  StorageSet warned_excluded_storages_;

  static void CollectExplicitStorages(
      const std::vector<std::shared_ptr<IRStructure>> &nodes,
      StorageSet *storages) {
    ICHECK(storages != nullptr);
    for (const auto &node : nodes) {
      if (!node->IsControl())
        continue;
      const auto *control = static_cast<const ControlNode *>(node.get());
      if (auto annotation =
              control->control->annotations.Get(kMultiBufferEligible)) {
        for (const Var &storage :
             NormalizeEligibleStorages(annotation.value())) {
          storages->insert(storage);
        }
      }
      CollectExplicitStorages(control->children, storages);
    }
  }

  void RewriteNodes(std::vector<std::shared_ptr<IRStructure>> &nodes) {
    for (const auto &node : nodes) {
      if (!node->IsControl())
        continue;
      auto *control = static_cast<ControlNode *>(node.get());
      RewriteNodes(control->children);
      RewriteControl(control);
    }
  }

  void RewriteControl(ControlNode *control) {
    For loop = control->control;
    Array<Var> eligible;
    StorageSet already;
    if (auto annotation = loop->annotations.Get(kMultiBufferEligible)) {
      for (const Var &storage : NormalizeEligibleStorages(annotation.value())) {
        if (manual_buffers_.count(storage)) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING) << "Ignoring explicit '" << kMultiBufferEligible
                         << "' claim for storage " << storage->name_hint
                         << " because it is already manually multi-buffered";
          }
          continue;
        }
        if (already.insert(storage).second)
          eligible.push_back(storage);
      }
    }

    auto planned = claims_.find(control);
    if (planned != claims_.end()) {
      for (const Var &storage : planned->second) {
        if (already.insert(storage).second)
          eligible.push_back(storage);
      }
    }

    loop.CopyOnWrite()->annotations.Set(kMultiBufferEligible, eligible);
    control->control = std::move(loop);
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "AnnotateMultiBufferEligible",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          StorageSet manual_buffers;
          for (const auto &[storage, _] :
               scheduled_tir.metadata.manual_buffer_versions) {
            manual_buffers.insert(storage);
          }
          MultiBufferAnnotator::Rewrite(&scheduled_tir,
                                        std::move(manual_buffers));
          return EncodeScheduledTIR(std::move(scheduled_tir));
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateMultiBufferEligible", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMultiBufferEligible",
                        AnnotateMultiBufferEligible);
}

} // namespace tl
} // namespace tvm
