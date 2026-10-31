/*!
 * \file legalize_parallel_to_pto.cc
 * \brief LegalizeParallelToPTO: fold lane-varying control flow and the remaining
 * expression forms into value choices, so the vectorizer only sees `BufferStore`
 * with a `Select` value.
 *
 * Pipeline: VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO (PTO targets only).
 *
 * Per conditional, the post-order visitor aligns both arms into one destination
 * sequence (`AlignArmDestinations`: same-destination stores share a position, a
 * lone store gets an identity pad `A = Load(A)`) and emits one unconditional
 * `Select` per position (`EmitAlignedPairs`). Min/Max become comparisons plus
 * `Select` and the tl/tir.infinity() sentinel becomes a FloatImm, in the same
 * pass.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "parallel_to_pto_utils.h"

#include "../target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

[[noreturn]] void Fail(const std::string &msg) {
  LOG(FATAL) << "[LegalizeParallelToPTO] " << msg;
}

// Collects the BufferLoad locations of an expression: the ordering step needs to
// know which destination's entry value a value reads.
class LoadCollector : public tirx::StmtExprVisitor {
 public:
  struct Location {
    const BufferNode *buffer;
    Array<PrimExpr> indices;
  };

  std::vector<Location> loads;

  void VisitExpr_(const BufferLoadNode *op) final {
    loads.push_back({op->buffer.get(), op->indices});
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
};

// Task set of one arm of a conditional: one bit per store. The bits are packed
// into 64-bit words so the pairing DP carries no arm-size limit -- the search is
// exponential in the arm size, but the lower-bound pruning collapses the shapes
// that appear in practice (distinct destinations need no branching at all).
class TaskMask {
 public:
  TaskMask() = default;
  explicit TaskMask(size_t bits) : words_((bits + 63) / 64, 0) {}

  static TaskMask Full(size_t bits) {
    TaskMask mask(bits);
    for (size_t i = 0; i < bits; ++i) {
      mask.Set(i);
    }
    return mask;
  }

  // Bits set in `full` but not in `done`.
  static TaskMask Remaining(const TaskMask &full, const TaskMask &done) {
    TaskMask out(full);
    for (size_t w = 0; w < out.words_.size(); ++w) {
      out.words_[w] = full.words_[w] & ~done.words_[w];
    }
    return out;
  }

  static int CountIntersection(const TaskMask &lhs, const TaskMask &rhs) {
    int total = 0;
    for (size_t w = 0; w < lhs.words_.size(); ++w) {
      total += __builtin_popcountll(lhs.words_[w] & rhs.words_[w]);
    }
    return total;
  }

  bool Test(size_t i) const { return ((words_[i / 64] >> (i % 64)) & 1u) != 0; }

  void Set(size_t i) { words_[i / 64] |= (uint64_t{1} << (i % 64)); }

  TaskMask With(size_t i) const {
    TaskMask out = *this;
    out.Set(i);
    return out;
  }

  // True when some bit of *this is not set in `other`.
  bool AnyOutside(const TaskMask &other) const {
    for (size_t w = 0; w < words_.size(); ++w) {
      if ((words_[w] & ~other.words_[w]) != 0) {
        return true;
      }
    }
    return false;
  }

  bool operator==(const TaskMask &other) const { return words_ == other.words_; }
  bool operator!=(const TaskMask &other) const { return !(*this == other); }
  bool operator<(const TaskMask &other) const { return words_ < other.words_; }

 private:
  std::vector<uint64_t> words_;
};

class ControlFlowMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      // Enclosing blocks are traversed to reach the SIMD_VF region; nothing
      // outside it is rewritten.
      return StmtExprMutator::VisitStmt_(op);
    }
    // Clearing must precede the fold: folding builds stores without the field.
    Stmt normalized =
        HasParallelUnit(op->body) ? NormalizePredicates(op->body) : op->body;
    // No region-level gate: every rewrite below is gated on `in_parallel_`, so
    // a hand-written VMI region (SIMD_VF without a unit) stays untouched.
    bool saved = in_simd_vf_;
    in_simd_vf_ = true;
    Stmt body = StmtExprMutator::VisitStmt(normalized);
    in_simd_vf_ = saved;
    SBlock block = GetRef<SBlock>(op);
    SBlockNode *node = block.CopyOnWrite();
    node->body = body;
    return block;
  }

  static bool HasParallelUnit(const Stmt &body) {
    bool found = false;
    PostOrderVisit(body, [&found](const ObjectRef &node) {
      if (const auto *loop = node.as<ForNode>()) {
        if (loop->kind == ForKind::kParallel) {
          found = true;
        }
      }
    });
    return found;
  }

  // Clears an access predicate that provably holds. Verify admitted the field
  // only when it is provably true, so anything else here means the two passes
  // disagree; that is reported instead of ignored, because dropping a
  // conditional access makes it unconditional. The judgment is the same helper
  // Verify uses, with the same loop-range and scalar-Bind context.
  Stmt NormalizePredicates(const Stmt &body) {
    class Normalizer : public StmtExprMutator {
     public:
      explicit Normalizer(ControlFlowMutator *parent) : parent_(parent) {}

      // Facts are scoped exactly like WalkRegion / the unit visitor: restoring
      // only the definition set would leave a branch-local Bind value usable
      // after its scope ends.
      void EnterScope(Map<Var, PrimExpr> *saved) {
        *saved = parent_->predicate_env_;
      }
      void LeaveScope(Map<Var, PrimExpr> *saved) {
        parent_->predicate_env_ = std::move(*saved);
      }
      Stmt VisitStmt_(const ForNode *op) final {
        parent_->predicate_analyzer_.Bind(
            op->loop_var, Range::FromMinExtent(op->min, op->extent),
            /*allow_override=*/true);
        Map<Var, PrimExpr> saved;
        EnterScope(&saved);
        Stmt result = StmtExprMutator::VisitStmt_(op);
        LeaveScope(&saved);
        return result;
      }
      Stmt VisitStmt_(const IfThenElseNode *op) final {
        Map<Var, PrimExpr> saved;
        EnterScope(&saved);
        Stmt result = StmtExprMutator::VisitStmt_(op);
        LeaveScope(&saved);
        return result;
      }
      Stmt VisitStmt_(const WhileNode *op) final {
        Map<Var, PrimExpr> saved;
        EnterScope(&saved);
        Stmt result = StmtExprMutator::VisitStmt_(op);
        LeaveScope(&saved);
        return result;
      }
      Stmt VisitStmt_(const AttrStmtNode *op) final {
        Map<Var, PrimExpr> saved;
        EnterScope(&saved);
        Stmt result = StmtExprMutator::VisitStmt_(op);
        LeaveScope(&saved);
        return result;
      }
      Stmt VisitStmt_(const BindNode *op) final {
        Stmt result = StmtExprMutator::VisitStmt_(op);
        if (const auto *bind = result.as<BindNode>()) {
          if (!bind->value.dtype().is_handle() &&
              pto::IsSafelyReevaluableValue(bind->value)) {
            parent_->predicate_env_.Set(bind->var, bind->value);
          }
        }
        return result;
      }
      Stmt VisitStmt_(const BufferStoreNode *op) final {
        BufferStore store =
            Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
        if (store->predicate.defined()) {
          parent_->CheckAdmittedPredicate(store->predicate.value(),
                                          store->buffer->name);
          store.CopyOnWrite()->predicate = std::nullopt;
        }
        return store;
      }
      PrimExpr VisitExpr_(const BufferLoadNode *op) final {
        BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
        if (load->predicate.defined()) {
          parent_->CheckAdmittedPredicate(load->predicate.value(),
                                          load->buffer->name);
          load.CopyOnWrite()->predicate = std::nullopt;
        }
        return load;
      }

     private:
      ControlFlowMutator *parent_;
    };
    // Seed the proof context with the facts that stay valid around this
    // region: enclosing loop ranges and the scalar Binds defined outside it.
    for (const auto &entry : outer_ranges_) {
      predicate_analyzer_.Bind(entry.first, entry.second,
                               /*allow_override=*/true);
    }
    predicate_env_ = outer_binds_;
    Normalizer normalizer(this);
    return normalizer(body);
  }

  void CheckAdmittedPredicate(const PrimExpr &predicate,
                              const String &buffer_name) {
    if (!pto::IsProvablyTruePredicate(predicate, &predicate_analyzer_,
                                      predicate_env_)) {
      Fail("buffer `" + std::string(buffer_name) +
           "` carries an access predicate that is not provably true; "
           "VerifyParallelToPTO admits only compile-time-true predicates, so "
           "the two passes disagree");
    }
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (!in_simd_vf_ || op->kind != ForKind::kParallel) {
      // An enclosing loop range is a fact a predicate inside a nested
      // converting region may rest on; keep it for that region's normaliser.
      outer_ranges_.push_back(
          {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
      Stmt result = StmtExprMutator::VisitStmt_(op);
      outer_ranges_.pop_back();
      return result;
    }

    bool saved_in = in_parallel_;
    in_parallel_ = true;

    Stmt body = StmtExprMutator::VisitStmt(op->body);

    For loop = GetRef<For>(op);
    if (!body.same_as(op->body)) {
      loop.CopyOnWrite()->body = body;
    }

    in_parallel_ = saved_in;
    return loop;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    Stmt result = StmtExprMutator::VisitStmt_(op);
    if (const auto *bind = result.as<BindNode>()) {
      if (!bind->value.dtype().is_handle() &&
          pto::IsSafelyReevaluableValue(bind->value)) {
        outer_binds_.Set(bind->var, bind->value);
      }
    }
    return result;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // Post-order: a nested `if` is already folded into a store, so both arms
    // are flat store lists here.
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    // The condition travels into the value choices, so it needs the same
    // rewriting as the arms (Min/Max -> Select, infinity -> FloatImm);
    // otherwise Vectorize would receive a node it cannot lower.
    // A condition that reads a destination this `if` writes would observe the
    // newly written value in the folded sequence, so that shape is rejected.
    PrimExpr condition = VisitExpr(op->condition);
    {
      LoadCollector condition_loads;
      condition_loads(condition);
      std::vector<const BufferStoreNode *> written;
      pto::CollectStores(then_case, &written);
      if (else_case.defined()) {
        pto::CollectStores(else_case.value(), &written);
      }
      for (const LoadCollector::Location &load : condition_loads.loads) {
        for (const BufferStoreNode *store : written) {
          if (MayConflict(store->buffer.get(), load.buffer)) {
            Fail("the if condition reads a destination this branch writes; the "
                 "folded value choices would see the new value, so this branch "
                 "cannot be rewritten (compute the condition before the if)");
          }
        }
      }
    }
    Stmt aligned_then;
    Optional<Stmt> aligned_else;
    if (AlignArmDestinations(then_case, else_case, &aligned_then,
                             &aligned_else)) {
      then_case = aligned_then;
      else_case = aligned_else;
    }
    return EmitAlignedPairs(condition, then_case, else_case);
  }

  // Min/Max -> Select so Vectorize only needs cmp + vsel. Rewritten only
  // inside a T.Parallel unit in a converting region: scalar expressions in
  // sibling serial code within the same SIMD_VF keep their Min/Max shape.
  PrimExpr VisitExpr_(const MinNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a < b, a, b);
  }
  PrimExpr VisitExpr_(const MaxNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    return Select(a > b, a, b);
  }

  // The frontend infinity() sentinel is a compile-time +inf constant of its
  // dtype; rewrite it to FloatImm so Vectorize's constant handling covers it
  // without a frontend-op-name case.
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (in_parallel_) {
      if (const auto *op_node = op->op.as<OpNode>()) {
        const std::string &name = op_node->name;
        if (name == "tl.infinity" || name == "tir.infinity") {
          return FloatImm(op->dtype,
                          std::numeric_limits<double>::infinity());
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  // Conservative storage test for *ordering*: two accesses may conflict unless
  // they name different known origins (a parameter or an allocation). Views over
  // the same storage conflict even when their logical indices differ, because
  // A[1,i] and Av[0,i] can denote the same physical address. The judgment is the
  // same one Verify applies (pto::MayConflictStorage), so the two passes cannot
  // classify a pair differently.
  bool MayConflict(const BufferNode *lhs_buffer, const BufferNode *rhs_buffer) {
    return pto::MayConflictStorage(lhs_buffer, rhs_buffer, storage_origins_);
  }

  bool SameArmLocation(const BufferNode *lhs_buffer,
                       const Array<PrimExpr> &lhs_indices,
                       const BufferNode *rhs_buffer,
                       const Array<PrimExpr> &rhs_indices) {
    // Compare the *physical* address: two Buffer views (A and Av) can share
    // the same UB data while being distinct BufferNodes, and views that start
    // at a different element offset are different destinations even though
    // their logical indices match.
    // Two views with different element types cannot share a Select: both
    // value branches must have the same dtype.
    if (lhs_buffer->dtype != rhs_buffer->dtype) {
      return false;
    }
    if (lhs_buffer->data.get() != rhs_buffer->data.get() ||
        lhs_indices.size() != rhs_indices.size() ||
        !ffi::StructuralEqual()(lhs_buffer->elem_offset,
                                rhs_buffer->elem_offset) ||
        !ffi::StructuralEqual()(lhs_buffer->strides, rhs_buffer->strides) ||
        !ffi::StructuralEqual()(lhs_buffer->shape, rhs_buffer->shape)) {
      return false;
    }
    // Compare the destination as a value
    arith::Analyzer analyzer;
    for (size_t i = 0; i < lhs_indices.size(); ++i) {
      if (!analyzer.CanProveEqual(lhs_indices[i], rhs_indices[i])) {
        return false;
      }
    }
    return true;
  }

  static bool CollectPlanTasks(const Stmt &stmt, std::vector<PlanTask> *out) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (!CollectPlanTasks(s, out)) {
          return false;
        }
      }
      return true;
    }
    if (pto::IsNoOpStmt(stmt)) {
      return true;
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      out->push_back({stmt, store});
      return true;
    }
    if (stmt.as<BindNode>() != nullptr) {
      out->push_back({stmt, nullptr});
      return true;
    }
    return false;
  }

  bool ConditionConflictsWithWrites(const PrimExpr &condition,
                                    const Stmt &then_case,
                                    const Optional<Stmt> &else_case) {
    LoadCollector condition_loads;
    condition_loads(condition);
    std::vector<PlanTask> tasks;
    if (!CollectPlanTasks(then_case, &tasks) ||
        (else_case.defined() && !CollectPlanTasks(else_case.value(), &tasks))) {
      Fail("conditional branches must lower to stores, Binds or no-ops");
    }
    for (const PlanTask &task : tasks) {
      if (task.store == nullptr) {
        continue;
      }
      for (const LoadCollector::Location &load : condition_loads.loads) {
        if (MayConflict(task.store->buffer.get(), load.buffer)) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<AlignedTask> IdentityAlignment(const Stmt &then_case,
                                             const Optional<Stmt> &else_case) {
    std::vector<PlanTask> then_tasks;
    std::vector<PlanTask> else_tasks;
    if (!CollectPlanTasks(then_case, &then_tasks) ||
        (else_case.defined() && !CollectPlanTasks(else_case.value(), &else_tasks))) {
      Fail("conditional branches must lower to stores, Binds or no-ops");
    }
    std::vector<AlignedTask> result;
    result.reserve(then_tasks.size() + else_tasks.size());
    for (const PlanTask &task : then_tasks) {
      result.emplace_back(task.stmt, Optional<Stmt>());
    }
    for (const PlanTask &task : else_tasks) {
      result.emplace_back(Optional<Stmt>(), task.stmt);
    }
    return result;
  }

  // Both arms are aligned position-wise. A pair of stores becomes one value
  // choice; a condition Bind remains a task of its own and is never dropped or
  // treated as a freely movable expression alias.
  bool AlignArmDestinations(const Stmt &then_case,
                            const Optional<Stmt> &else_case, Stmt *then_out,
                            Optional<Stmt> *else_out) {
    std::vector<const BufferStoreNode *> then_stores;
    std::vector<const BufferStoreNode *> else_stores;
    if (!pto::CollectStores(then_case, &then_stores)) {
      return false;
    }
    if (else_case.defined() &&
        !pto::CollectStores(else_case.value(), &else_stores)) {
      return false;
    }
    if (then_stores.empty() && else_stores.empty()) {
      return false;
    }

    const size_t n = then_stores.size();
    const size_t m = else_stores.size();

    // Precedence masks: within one arm, a pair that touches the same address
    // keeps its order (RAW: write before the read, WAR: read before the
    // overwrite, WAW: the later write survives).
    const auto build_predecessors =
        [this](const std::vector<const BufferStoreNode *> &seq) {
          const size_t k = seq.size();
          std::vector<TaskMask> pred(k, TaskMask(k));
          for (size_t i = 0; i < k; ++i) {
            LoadCollector reads_i;
            reads_i(seq[i]->value);
            for (size_t j = i + 1; j < k; ++j) {
              LoadCollector reads_j;
              reads_j(seq[j]->value);
              bool dep = MayConflict(seq[i]->buffer.get(), seq[j]->buffer.get());
              for (const LoadCollector::Location &load : reads_j.loads) {
                if (!dep && MayConflict(seq[i]->buffer.get(), load.buffer)) {
                  dep = true;
                }
              }
              for (const LoadCollector::Location &load : reads_i.loads) {
                if (!dep && MayConflict(seq[j]->buffer.get(), load.buffer)) {
                  dep = true;
                }
              }
              if (dep) {
                pred[j].Set(i);
              }
            }
          }
          return pred;
        };
    const std::vector<TaskMask> pred_t = build_predecessors(then_stores);
    const std::vector<TaskMask> pred_e = build_predecessors(else_stores);

    // Per-label masks: one position carries exactly one label, so a label with
    // k tasks left on either side needs at least k more positions. Summing this
    // over labels is a provable lower bound on the remaining steps, so the
    // search can stop as soon as it reaches it.
    const TaskMask full_t = TaskMask::Full(n);
    const TaskMask full_e = TaskMask::Full(m);
    std::vector<std::pair<TaskMask, TaskMask>> label_masks;
    std::vector<std::pair<const BufferNode *, Array<PrimExpr>>> representatives;
    const auto label_index = [this, &label_masks, &representatives, n, m](
                                 const BufferStoreNode *store) -> size_t {
      for (size_t k = 0; k < label_masks.size(); ++k) {
        if (SameArmLocation(representatives[k].first, representatives[k].second,
                            store->buffer.get(), store->indices)) {
          return k;
        }
      }
      label_masks.push_back({TaskMask(n), TaskMask(m)});
      representatives.push_back({store->buffer.get(), store->indices});
      return label_masks.size() - 1;
    };
    std::vector<size_t> label_of_t(then_stores.size(), 0);
    std::vector<size_t> label_of_e(else_stores.size(), 0);
    for (size_t i = 0; i < n; ++i) {
      label_of_t[i] = label_index(then_stores[i]);
      label_masks[label_of_t[i]].first.Set(i);
    }
    for (size_t j = 0; j < m; ++j) {
      label_of_e[j] = label_index(else_stores[j]);
      label_masks[label_of_e[j]].second.Set(j);
    }
    const auto lower_bound = [&label_masks, full_t, full_e](
                                 const TaskMask &done_t,
                                 const TaskMask &done_e) {
      const TaskMask rem_t = TaskMask::Remaining(full_t, done_t);
      const TaskMask rem_e = TaskMask::Remaining(full_e, done_e);
      int bound = 0;
      for (const auto &masks : label_masks) {
        const int a = TaskMask::CountIntersection(masks.first, rem_t);
        const int b = TaskMask::CountIntersection(masks.second, rem_e);
        bound += a > b ? a : b;
      }
      return bound;
    };

    std::map<std::pair<TaskMask, TaskMask>, int> memo;
    std::function<int(const TaskMask &, const TaskMask &)> best_steps =
        [&](const TaskMask &done_t, const TaskMask &done_e) -> int {
      if (done_t == full_t && done_e == full_e) {
        return 0;
      }
      const auto key = std::make_pair(done_t, done_e);
      const auto found = memo.find(key);
      if (found != memo.end()) {
        return found->second;
      }
      const int lb = lower_bound(done_t, done_e);
      int best = 1 << 20;
      for (size_t i = 0; i < n; ++i) {
        if (done_t.Test(i)) {
          continue;
        }
        if (pred_t[i].AnyOutside(done_t)) {
          continue;
        }
        for (size_t j = 0; j < m; ++j) {
          if (done_e.Test(j)) {
            continue;
          }
          if (pred_e[j].AnyOutside(done_e)) {
            continue;
          }
          if (!SameArmLocation(then_stores[i]->buffer.get(),
                               then_stores[i]->indices,
                               else_stores[j]->buffer.get(),
                               else_stores[j]->indices)) {
            continue;
          }
          best = std::min(best, 1 + best_steps(done_t.With(i), done_e.With(j)));
          if (best == lb) {
            memo[key] = best;
            return best;
          }
        }
      }
      for (size_t i = 0; i < n; ++i) {
        if (!done_t.Test(i) && !pred_t[i].AnyOutside(done_t)) {
          best = std::min(best, 1 + best_steps(done_t.With(i), done_e));
          if (best == lb) {
            memo[key] = best;
            return best;
          }
        }
      }
      for (size_t j = 0; j < m; ++j) {
        if (!done_e.Test(j) && !pred_e[j].AnyOutside(done_e)) {
          best = std::min(best, 1 + best_steps(done_t, done_e.With(j)));
          if (best == lb) {
            memo[key] = best;
            return best;
          }
        }
      }
      memo[key] = best;
      return best;
    };

    const auto pad = [](const BufferStoreNode *shape) {
      return BufferStore(shape->buffer, BufferLoad(shape->buffer, shape->indices),
                         shape->indices);
    };

    Array<Stmt> then_seq;
    Array<Stmt> else_seq;
    TaskMask done_t(n);
    TaskMask done_e(m);
    while (done_t != full_t || done_e != full_e) {
      const int total = best_steps(done_t, done_e);
      bool emitted = false;
      // Prefer a pair: it keeps the position count minimal and avoids a padding
      // store (and its read-back load).
      for (size_t i = 0; i < n && !emitted; ++i) {
        if (done_t.Test(i)) {
          continue;
        }
        if (pred_t[i].AnyOutside(done_t)) {
          continue;
        }
        for (size_t j = 0; j < m && !emitted; ++j) {
          if (done_e.Test(j)) {
            continue;
          }
          if (pred_e[j].AnyOutside(done_e)) {
            continue;
          }
          if (!SameArmLocation(then_stores[i]->buffer.get(),
                               then_stores[i]->indices,
                               else_stores[j]->buffer.get(),
                               else_stores[j]->indices)) {
            continue;
          }
          if (1 + best_steps(done_t.With(i), done_e.With(j)) == total) {
            then_seq.push_back(GetRef<Stmt>(then_stores[i]));
            else_seq.push_back(GetRef<Stmt>(else_stores[j]));
            done_t.Set(i);
            done_e.Set(j);
            emitted = true;
          }
        }
      }
      for (size_t i = 0; i < n && !emitted; ++i) {
        if (!done_t.Test(i) && !pred_t[i].AnyOutside(done_t) &&
            1 + best_steps(done_t.With(i), done_e) == total) {
          then_seq.push_back(GetRef<Stmt>(then_stores[i]));
          else_seq.push_back(pad(then_stores[i]));
          done_t.Set(i);
          emitted = true;
        }
      }
      for (size_t j = 0; j < m && !emitted; ++j) {
        if (!done_e.Test(j) && !pred_e[j].AnyOutside(done_e) &&
            1 + best_steps(done_t, done_e.With(j)) == total) {
          else_seq.push_back(GetRef<Stmt>(else_stores[j]));
          then_seq.push_back(pad(else_stores[j]));
          done_e.Set(j);
          emitted = true;
        }
      }
      if (!emitted) {
        return false;  // unreachable: a single store move always exists
      }
    }

    *then_out =
        then_seq.empty() ? Stmt(Evaluate(0)) : Stmt(SeqStmt::Flatten(then_seq));
    if (else_out != nullptr) {
      *else_out = else_seq.empty()
                      ? Optional<Stmt>()
                      : Optional<Stmt>(SeqStmt::Flatten(else_seq));
    }
    return true;
  }

  Stmt EmitAlignedPairs(const PrimExpr &condition, const Stmt &then_case,
                        const Optional<Stmt> &else_case) {
    std::vector<const BufferStoreNode *> then_stores;
    std::vector<const BufferStoreNode *> else_stores;
    if (!pto::CollectStores(then_case, &then_stores)) {
      Fail("If then-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
      return Stmt();
    }
    if (else_case.defined() &&
        !pto::CollectStores(else_case.value(), &else_stores)) {
      Fail("If else-branch under T.Parallel must be BufferStore(s) or no-op "
           "(Select-rewritable); unsupported stmt shape (Verify admits the "
           "same shapes)");
      return Stmt();
    }
    if (then_stores.size() != else_stores.size()) {
      Fail("both arms of this `if` must write the same destination addresses to "
           "be folded into value choices; the alignment step could not pad them "
           "(non lane-private address form)");
      return Stmt();
    }
    Array<Stmt> flat;
    for (size_t i = 0; i < then_stores.size(); ++i) {
      if (!SameArmLocation(then_stores[i]->buffer.get(),
                           then_stores[i]->indices,
                           else_stores[i]->buffer.get(),
                           else_stores[i]->indices)) {
        Fail("the two arms of this `if` do not write the same destination at the "
             "same position after alignment; the branch cannot be folded into "
             "value choices");
        return Stmt();
      }
      flat.push_back(BufferStore(
          then_stores[i]->buffer,
          Select(condition, then_stores[i]->value, else_stores[i]->value),
          then_stores[i]->indices));
    }
    return flat.empty() ? Stmt(Evaluate(0)) : Stmt(SeqStmt::Flatten(flat));
  }
  bool in_simd_vf_ = false;
  bool in_parallel_ = false;
  int condition_serial_ = 0;
  // Storage origins: which data Vars are real allocations and which are
  // parameters, plus the parameter no-alias facts.
  pto::StorageOrigins storage_origins_;
  // Predicate context for the whole-region normalisation: the loop ranges the
  // region imposes and the scalar `Bind` values in scope, exactly the context
  // Verify used to admit the predicate.
  arith::Analyzer predicate_analyzer_;
  Map<Var, PrimExpr> predicate_env_;
  // Facts that stay valid around a converting region: enclosing loop ranges and
  // safely re-evaluable scalar Binds defined outside it.
  Map<Var, PrimExpr> outer_binds_;
  std::vector<std::pair<Var, Range>> outer_ranges_;
};

PrimFunc LowerControlFlow(PrimFunc f) {
  auto *n = f.CopyOnWrite();
  ControlFlowMutator mutator;
  // Storage origins are collected before the fold so the ordering analysis can
  // tell two independent allocations apart from two views of one storage.
  const bool no_alias = [&] {
    auto attr = n->GetAttr<Integer>(tirx::attr::kNoAlias);
    return attr.defined() && attr.value()->value != 0;
  }();
  Array<Var> non_restrict;
  if (auto attr = n->GetAttr<Array<Var>>(tl::attr::kNonRestrictParams)) {
    non_restrict = attr.value();
  }
  mutator.storage_origins_ = pto::CollectStorageOrigins(
      n->body, n->buffer_map, no_alias, non_restrict);
  n->body = mutator(std::move(n->body));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return LowerControlFlow(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.LegalizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                        LegalizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm
