/*!
 * \file annotate_multi_buffer_eligible.cc
 * \brief Pre-AutoSchedule analysis: for each For loop, decide which on-chip
 *        buffers can be multi-buffered and record the set on the For's
 *        annotations under "multi_buffer_eligible".
 *
 *  Runs after NormalizeControlFlowForSchedule (which rewrites `while` loops
 *  into bounded serial for loops and hoists complex if conditions into Bind
 *  variables). Operating after that rewrite lets this pass mark buffers inside
 * a former while body as multi-buffer eligible; the hoisted Bind conditions
 * keep the if-then-else intact so the write-first analysis is unaffected.
 *
 *  Automatic claims form the deepest disjoint loop frontier that completely
 *  covers a storage's ordinary accesses; owner-external fills may initialize
 *  every physical version. Every claimed loop must write the storage before
 *  any read. If descendant loops cover only part of an epoch (for example, a
 *  row-writing loop followed by a whole-buffer consumer), the claim is
 *  promoted to a write-first ancestor. Independent sibling loops may
 *  therefore become multiple owners of one storage.
 *
 *  Frontend may pre-set the annotation. We treat it as a seed rather than a
 *  veto: frontend-listed buffers are kept unless the storage is already
 *  manually versioned or is used by an unsupported AttrStmt::node expression.
 *  Known assume attributes carry a PrimExpr node that is analyzed and rewritten
 *  with the guarded task. Any buffer the analysis additionally proves eligible
 *  is unioned in. A buffer the user does NOT want
 *  multi-buffered can be pinned to a single version via
 *  T.annotate_buffer_versions({buf: 1}) instead of being omitted here.
 */
#include <tvm/arith/analyzer.h>
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
#include "ascend/transform/buffer_version.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using StorageSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using LoopStorageClaims =
    std::unordered_map<For, std::vector<Var>, ObjectPtrHash, ObjectPtrEqual>;

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

class WriteFirstClassifier : public StmtExprVisitor {
public:
  AccessOrder Classify(const Stmt &s, const Var &storage) {
    target_storage_ = storage;
    result_ = AccessOrder::kUntouched;
    StmtExprVisitor::VisitStmt(s);
    return result_;
  }

private:
  Var target_storage_;
  AccessOrder result_ = AccessOrder::kUntouched;
  arith::Analyzer analyzer_;

  // Terminal states determine the answer; further sub-events can't change it.
  static bool IsTerminal(AccessOrder o) {
    return o == AccessOrder::kReadFirst || o == AccessOrder::kWriteFirst;
  }

  // Sequential composition: order of two sub-events in program order.
  // (kReadFirst / kWriteFirst absorb; kMaybeWriteFirst+kReadFirst is
  //  conservatively kReadFirst since the prefix's skip case would leak.)
  static AccessOrder SeqCompose(AccessOrder a, AccessOrder b) {
    if (a == AccessOrder::kReadFirst || a == AccessOrder::kWriteFirst)
      return a;
    if (a == AccessOrder::kUntouched)
      return b;
    // a == kMaybeWriteFirst
    if (b == AccessOrder::kUntouched)
      return AccessOrder::kMaybeWriteFirst;
    return b;
  }

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

  // Merge of an if/then/else (both branches taken under mutually exclusive
  // conditions). Returns the safest classification consistent with both.
  static AccessOrder MergeIfThenElse(AccessOrder t, AccessOrder e) {
    if (t == AccessOrder::kReadFirst || e == AccessOrder::kReadFirst)
      return AccessOrder::kReadFirst;
    if (t == AccessOrder::kUntouched && e == AccessOrder::kUntouched)
      return AccessOrder::kUntouched;
    if (t == AccessOrder::kWriteFirst && e == AccessOrder::kWriteFirst)
      return AccessOrder::kWriteFirst;
    // Mix among {kUntouched, kWriteFirst, kMaybeWriteFirst}: maybe-write.
    return AccessOrder::kMaybeWriteFirst;
  }
  // Single-branch if (no else); implicit else is kUntouched.
  static AccessOrder MergeIfThenOnly(AccessOrder t) {
    if (t == AccessOrder::kReadFirst)
      return AccessOrder::kReadFirst;
    if (t == AccessOrder::kUntouched)
      return AccessOrder::kUntouched;
    // kWriteFirst or kMaybeWriteFirst: either the branch ran and wrote, or
    // didn't run at all — never a leaked read.
    return AccessOrder::kMaybeWriteFirst;
  }
  // If the surrounding loop / block may not execute, demote write-first to
  // maybe-write-first (write not guaranteed). kReadFirst and kUntouched and
  // kMaybeWriteFirst pass through unchanged.
  static AccessOrder DemoteIfNotMustExecute(AccessOrder r, bool must_execute) {
    if (must_execute)
      return r;
    if (r == AccessOrder::kWriteFirst)
      return AccessOrder::kMaybeWriteFirst;
    return r;
  }

  bool ForMustExecute(const ForNode *op) {
    // step >= 1 (default is 1) and extent > 0.
    if (op->step.has_value()) {
      if (!analyzer_.CanProve(op->step.value() >= 1))
        return false;
    }
    return analyzer_.CanProve(op->extent > 0);
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
    bool must_exec = ForMustExecute(op);
    auto body = ClassifyLocal([&] { VisitStmt(op->body); });
    // NOTE: technically unsafe heuristic. kMaybeWriteFirst body means each
    // iteration is independently safe (skip or write-first), but says
    // nothing about whether *any* iteration takes the write branch. If every
    // iteration happens to skip, the loop overall leaves the buffer
    // untouched — yet we report kWriteFirst, which can mislead an outer
    // classifier that sees a subsequent read in the same scope as "safe
    // because the For wrote first". We accept this risk because the typical
    // pattern (conditional write inside a hot loop) almost always has the
    // condition true at least once, and being conservative here was
    // empirically too restrictive for multi-buffer eligibility.
    if (must_exec && body == AccessOrder::kMaybeWriteFirst)
      body = AccessOrder::kWriteFirst;
    result_ = SeqCompose(result_, DemoteIfNotMustExecute(body, must_exec));
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

// ---------------------------------------------------------------------------
// StorageAccessCollector: one-shot scan of a Stmt to collect every on-chip
// storage that is read or written anywhere inside it, preserving first-access
// order.
// ---------------------------------------------------------------------------
class StorageAccessCollector : public StmtExprVisitor {
public:
  static std::vector<Var> Collect(const Stmt &stmt) {
    StorageAccessCollector collector;
    collector(stmt);
    return std::move(collector.touched_);
  }

  static std::vector<Var> Collect(const PrimExpr &expr) {
    StorageAccessCollector collector;
    collector(expr);
    return std::move(collector.touched_);
  }

private:
  std::vector<Var> touched_;
  StorageSet seen_;

  void Record(const Buffer &buffer) {
    if (!IsAscendOnChipBuffer(buffer))
      return;
    const Var &storage = buffer->data;
    if (seen_.insert(storage).second)
      touched_.push_back(storage);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &r : op->reads)
      Record(r->buffer);
    for (const auto &w : op->writes)
      Record(w->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (CanRewriteMultiBufferAttrNode(op->attr_key, op->node))
      VisitExpr(Downcast<PrimExpr>(op->node));
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    static const Op &region_op = region();
    static const auto access_ptr_op = Op::Get("tl.access_ptr");
    if (op->op.same_as(region_op) && !op->args.empty()) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        Record(bl->buffer);
    } else if (op->op.same_as(access_ptr_op) && !op->args.empty()) {
      if (const auto *bl = op->args[0].as<BufferLoadNode>())
        Record(bl->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

// AttrStmt::node is an Any and generic TIR visitors do not traverse it. Known
// assume attributes have an explicit PrimExpr contract and follow the guarded
// task's physical version. Other node protocols remain unsupported, so storage
// referenced from them is excluded from automatic and explicit claims.
class UnsupportedAttrNodeStorageCollector : public StmtExprVisitor {
public:
  static StorageSet Collect(const Stmt &stmt) {
    UnsupportedAttrNodeStorageCollector collector;
    collector(stmt);
    return std::move(collector.storages_);
  }

private:
  StorageSet storages_;

  void VisitStmt_(const AttrStmtNode *op) final {
    if (!CanRewriteMultiBufferAttrNode(op->attr_key, op->node)) {
      auto node = op->node.try_cast<PrimExpr>();
      if (node.has_value()) {
        std::vector<Var> storages =
            StorageAccessCollector::Collect(node.value());
        storages_.insert(storages.begin(), storages.end());
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

bool IsTaskBoundaryAttribute(const String &attr_key) {
  return attr_key == tl::attr::kAscendTask ||
         attr_key == tl::attr::kAscendPerCoreTask ||
         attr_key == attr::kScheduleUnit;
}

// ---------------------------------------------------------------------------
// MultiBufferOwnerPlanner: for each storage, choose the deepest set of
// disjoint structural loops that covers every ordinary access. Independently
// rewritable fills may remain outside the frontier. A loop is selected only
// when its descendants do not already provide complete coverage and the whole
// loop body is write-first. This keeps a row-writing inner loop from claiming
// a buffer consumed after that loop, while allowing independent sibling loops
// to become multiple owners.
//
// Recursion mirrors MaterializeScheduleUnits:
//   - SeqStmt / IfThenElse / scheduling AttrStmt -> transparent structure
//   - For (serial/unrolled)            -> candidate ControlNode
//   - all other statements             -> opaque TaskNode leaf
// ---------------------------------------------------------------------------
class MultiBufferOwnerPlanner {
public:
  static LoopStorageClaims Plan(const Stmt &body,
                                const StorageSet &excluded_storages) {
    MultiBufferOwnerPlanner planner;
    for (const Var &storage : StorageAccessCollector::Collect(body)) {
      if (excluded_storages.count(storage))
        continue;
      CoveragePlan plan = planner.PlanStmt(body, storage);
      if (!plan.IsComplete())
        continue;
      for (const For &owner : plan.owners)
        planner.claims_[owner].push_back(storage);
    }
    return std::move(planner.claims_);
  }

private:
  // Ordered from least to most restrictive; sibling composition takes max.
  enum class Coverage {
    // No access in this subtree.
    kUntouched,
    // Every access is covered by descendant owners, an explicit claim, or a
    // broadcast fill.
    kCovered,
    // An ancestor loop may still claim the uncovered accesses.
    kNeedsOwner,
    // A control expression prevents any ancestor from claiming the storage.
    kBlocked,
  };

  struct CoveragePlan {
    Coverage coverage{Coverage::kUntouched};
    // Explicit claims are fixed seeds. An automatic ancestor must not subsume
    // one, because the explicit nested claim will remain in the IR.
    bool has_explicit_claim{false};
    std::vector<For> owners;

    bool IsComplete() const {
      return coverage == Coverage::kUntouched || coverage == Coverage::kCovered;
    }

    bool CanClaimHere() const {
      // An ownerless covered subtree contains only independently broadcastable
      // fills. Keep those fills outside the owner frontier so they initialize
      // every physical version instead of advancing one version per loop
      // iteration.
      return !has_explicit_claim && coverage == Coverage::kNeedsOwner;
    }

    void AddUncoveredAccess() {
      if (coverage != Coverage::kBlocked)
        coverage = Coverage::kNeedsOwner;
    }

    void BlockAncestorClaim() { coverage = Coverage::kBlocked; }

    void Merge(const CoveragePlan &other) {
      coverage = std::max(coverage, other.coverage);
      has_explicit_claim |= other.has_explicit_claim;
      owners.insert(owners.end(), other.owners.begin(), other.owners.end());
    }
  };

  LoopStorageClaims claims_;

  template <typename Container>
  static bool ContainsStorage(const Container &storages, const Var &storage) {
    return std::any_of(storages.begin(), storages.end(), [&](const Var &other) {
      return other.same_as(storage);
    });
  }

  static bool TouchesStorage(const Stmt &stmt, const Var &storage) {
    return ContainsStorage(StorageAccessCollector::Collect(stmt), storage);
  }

  static bool TouchesStorage(const PrimExpr &expr, const Var &storage) {
    return ContainsStorage(StorageAccessCollector::Collect(expr), storage);
  }

  static bool HasExplicitClaim(const For &loop, const Var &storage) {
    auto annotation = loop->annotations.Get(kMultiBufferEligible);
    return annotation.has_value() &&
           ContainsStorage(NormalizeEligibleStorages(annotation.value()),
                           storage);
  }

  static CoveragePlan PlanLeaf(const Stmt &stmt, const Var &storage) {
    if (!TouchesStorage(stmt, storage))
      return {};
    CoveragePlan result;
    result.coverage = CanBroadcastFillToStorage(stmt, storage)
                          ? Coverage::kCovered
                          : Coverage::kNeedsOwner;
    return result;
  }

  CoveragePlan PlanStmt(const Stmt &stmt, const Var &storage) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      CoveragePlan result;
      for (const Stmt &child : seq->seq)
        result.Merge(PlanStmt(child, storage));
      return result;
    }

    if (const auto *loop_node = stmt.as<ForNode>()) {
      For loop = GetRef<For>(loop_node);
      if (loop_node->kind != ForKind::kSerial &&
          loop_node->kind != ForKind::kUnrolled) {
        return PlanLeaf(stmt, storage);
      }

      if (HasExplicitClaim(loop, storage)) {
        CoveragePlan result;
        result.coverage = TouchesStorage(stmt, storage) ? Coverage::kCovered
                                                        : Coverage::kUntouched;
        result.has_explicit_claim = true;
        return result;
      }

      CoveragePlan body = PlanStmt(loop_node->body, storage);
      bool control_touched = TouchesStorage(loop_node->min, storage) ||
                             TouchesStorage(loop_node->extent, storage) ||
                             (loop_node->step.has_value() &&
                              TouchesStorage(loop_node->step.value(), storage));
      if (control_touched) {
        body.BlockAncestorClaim();
        return body;
      }
      if (!body.CanClaimHere())
        return body;

      WriteFirstClassifier classifier;
      AccessOrder order = classifier.Classify(loop_node->body, storage);
      if (order == AccessOrder::kWriteFirst ||
          order == AccessOrder::kMaybeWriteFirst) {
        CoveragePlan result;
        result.coverage = Coverage::kCovered;
        result.owners.push_back(loop);
        return result;
      }
      return body;
    }

    if (const auto *condition = stmt.as<IfThenElseNode>()) {
      CoveragePlan result = PlanStmt(condition->then_case, storage);
      if (condition->else_case.has_value())
        result.Merge(PlanStmt(condition->else_case.value(), storage));
      if (TouchesStorage(condition->condition, storage))
        result.BlockAncestorClaim();
      return result;
    }

    if (const auto *attribute = stmt.as<AttrStmtNode>()) {
      if (IsTaskBoundaryAttribute(attribute->attr_key)) {
        return PlanLeaf(stmt, storage);
      }
      CoveragePlan result = PlanStmt(attribute->body, storage);
      if (TouchesStorage(attribute->value, storage))
        result.BlockAncestorClaim();
      if (auto node = attribute->node.try_cast<PrimExpr>();
          node.has_value() && TouchesStorage(node.value(), storage)) {
        if (CanRewriteMultiBufferAttrNode(attribute->attr_key,
                                          attribute->node)) {
          result.AddUncoveredAccess();
        } else {
          result.BlockAncestorClaim();
        }
      }
      return result;
    }

    return PlanLeaf(stmt, storage);
  }
};

// ---------------------------------------------------------------------------
// MultiBufferAnnotator: normalize explicit claims and add the automatic owner
// frontier selected above. Traversal follows the same structural loop spine.
// ---------------------------------------------------------------------------
class MultiBufferAnnotator : public StmtMutator {
public:
  static Stmt Rewrite(const Stmt &body, StorageSet manual_buffers) {
    StorageSet excluded_storages = manual_buffers;
    StorageSet attr_node_storages =
        UnsupportedAttrNodeStorageCollector::Collect(body);
    excluded_storages.insert(attr_node_storages.begin(),
                             attr_node_storages.end());
    LoopStorageClaims claims =
        MultiBufferOwnerPlanner::Plan(body, excluded_storages);
    return MultiBufferAnnotator(std::move(claims), std::move(manual_buffers),
                                std::move(attr_node_storages))(body);
  }

private:
  MultiBufferAnnotator(LoopStorageClaims claims, StorageSet manual_buffers,
                       StorageSet attr_node_storages)
      : claims_(std::move(claims)), manual_buffers_(std::move(manual_buffers)),
        attr_node_storages_(std::move(attr_node_storages)) {}

  LoopStorageClaims claims_;
  StorageSet manual_buffers_;
  StorageSet attr_node_storages_;
  StorageSet warned_excluded_storages_;

  Stmt VisitStmt_(const ForNode *op) final {
    // Serial and unrolled loops are structural ControlNodes and may own an
    // automatic version ring. Parallel/vectorized loops remain opaque tasks.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      return GetRef<For>(op);
    }

    For for_node = GetRef<For>(op);
    Stmt new_body = VisitStmt(op->body);

    // Seed the eligible set with any frontend-provided annotation, then union
    // in everything the analysis proves safe. The frontend list is a hint of
    // buffers the user definitely wants multi-buffered; it does not suppress
    // additional candidates (the user pins unwanted ones to a single version
    // via T.annotate_buffer_versions instead).
    Array<Var> eligible;
    StorageSet already;
    if (auto v = op->annotations.Get(kMultiBufferEligible)) {
      for (const Var &storage : NormalizeEligibleStorages(v.value())) {
        bool is_manual = manual_buffers_.count(storage);
        bool has_unsupported_attr_node = attr_node_storages_.count(storage);
        if (is_manual || has_unsupported_attr_node) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING)
                << "Ignoring explicit '" << kMultiBufferEligible
                << "' claim for storage " << storage->name_hint
                << (is_manual ? " because it is already manually multi-buffered"
                              : " because it is referenced by an unsupported "
                                "AttrStmt::node expression");
          }
          continue;
        }
        if (already.insert(storage).second)
          eligible.push_back(storage);
      }
    }

    auto planned = claims_.find(for_node);
    if (planned != claims_.end()) {
      for (const Var &storage : planned->second) {
        if (already.insert(storage).second)
          eligible.push_back(storage);
      }
    }

    auto *n = for_node.CopyOnWrite();
    n->body = new_body;
    n->annotations.Set(kMultiBufferEligible, eligible);
    return for_node;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsTaskBoundaryAttribute(op->attr_key)) {
      return GetRef<AttrStmt>(op);
    }
    return StmtMutator::VisitStmt_(op);
  }

  // Leaves in the IR structure: do not descend (matches AutoSchedule).
  //
  // The callback runs this visitor on tilelang_root's body. Every block below
  // that root is an opaque TaskNode leaf.
  Stmt VisitStmt_(const SBlockNode *op) final { return GetRef<SBlock>(op); }
  Stmt VisitStmt_(const WhileNode *op) final { return GetRef<While>(op); }
};

} // namespace

using namespace tirx::transform;

// Collect the data Vars named in the `tl.manual_multi_buffer` annotation on any
// SBlock in the function body.
static StorageSet CollectManualMultiBuffers(const Stmt &body) {
  struct Visitor : public StmtVisitor {
    StorageSet vars;
    void VisitStmt_(const SBlockNode *op) final {
      if (auto annotation = op->annotations.Get(kManualMultiBuffer)) {
        for (const auto &[data, _] :
             annotation.value().cast<BufferVersionMap>()) {
          vars.insert(data);
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
  } v;
  v(body);
  return std::move(v.vars);
}

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "AnnotateMultiBufferEligible",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          auto manual = CollectManualMultiBuffers(root);
          root.CopyOnWrite()->body =
              MultiBufferAnnotator::Rewrite(root->body, std::move(manual));
          return root;
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
