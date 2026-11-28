/*!
 * \file legalize_parallel_to_pto.cc
 * \brief LegalizeParallelToPTO: fold lane-varying control flow and the
 * remaining expression forms into value choices, so the vectorizer only sees
 * `BufferStore` with a `Select` value.
 *
 * Pipeline: VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO (PTO targets only).
 *
 * Per conditional, the post-order visitor aligns both arms into one destination
 * sequence (`AlignArmDestinations`: same-destination stores share a position, a
 * lone store gets an identity pad `A = Load(A)`) and emits one unconditional
 * `Select` per position (`EmitAlignedPairs`).
 *
 * Min/Max take one of two paths in the same pass: an integer Min/Max, or a
 * floating-point one under `tl.pto.relaxed_minmax`, keeps its node so Vectorize
 * can pick the direct VMI instruction; every other admitted Min/Max becomes a
 * comparison plus `Select` before Vectorize sees it. The tl/tir.infinity()
 * sentinel becomes a FloatImm here as well.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
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

// Collects the BufferLoad locations of an expression: the ordering step needs
// to know which destination's entry value a value reads.
class LoadCollector : public tirx::StmtExprVisitor {
public:
  struct Location {
    // A handle, not a borrowed node pointer: the ordering analysis keeps these
    // locations while it rewrites the tree, so their lifetime must not depend
    // on the input tree staying alive and unmodified.
    Buffer buffer;
    Array<PrimExpr> indices;
  };

  std::vector<Location> loads;

  void VisitExpr_(const BufferLoadNode *op) final {
    loads.push_back({op->buffer, op->indices});
    tirx::StmtExprVisitor::VisitExpr_(op);
  }
};

// Task set of one arm of a conditional: one bit per store. The bits are packed
// into 64-bit words so the pairing DP carries no arm-size limit -- the search
// is exponential in the arm size, but the lower-bound pruning collapses the
// shapes that appear in practice (distinct destinations need no branching at
// all).
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

  bool operator==(const TaskMask &other) const {
    return words_ == other.words_;
  }
  bool operator!=(const TaskMask &other) const { return !(*this == other); }
  bool operator<(const TaskMask &other) const { return words_ < other.words_; }

private:
  std::vector<uint64_t> words_;
};

// A statement in a conditional conversion plan. Besides stores, an entry
// condition may be materialised as a Bind. Such Binds are scheduling tasks, not
// expression aliases: they must remain at their program point and dominate
// every use, including after an enclosing conditional is folded.
struct PlanTask {
  Stmt stmt;
  /*! The statement as a store when it is one; undefined for a plain Bind. */
  BufferStore store;
};

using AlignedTask = std::pair<Optional<Stmt>, Optional<Stmt>>;

class ControlFlowMutator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMD_VF") {
      // Enclosing blocks are traversed to reach the SIMD_VF region; nothing
      // outside it is rewritten.
      return StmtExprMutator::VisitStmt_(op);
    }
    // Clearing must precede the fold: folding builds stores without the field.
    // Admission uses the same region scan Verify and Vectorize use, so a
    // `T.Parallel` that holds explicit `tl.simd.*` calls is not mistaken for an
    // automatic unit (the scan skips those loops; a plain "any Parallel loop"
    // probe would fold inside them).
    const pto::PtoRegionScan scan = pto::ScanPtoRegion(op->body);
    Stmt normalized =
        scan.units.empty() ? op->body : NormalizePredicates(op->body);
    // No region-level gate: every rewrite below is gated on `in_parallel_`, so
    // a hand-written VMI region (SIMD_VF without a unit) stays untouched.
    bool saved = in_simd_vf_;
    in_simd_vf_ = true;
    Stmt body = StmtExprMutator::VisitStmt(normalized);
    in_simd_vf_ = saved;
    SBlock block = GetRef<SBlock>(op);
    SBlockNode *node = block.CopyOnWrite();
    // A realized block's initializer is a statement position of its own: it
    // runs before the body and carries access predicates too, so it needs the
    // same clearing -- under the same condition as the body. A block without an
    // automatic unit is not rewritten, and its initializer stays untouched.
    if (op->init.defined() && !scan.units.empty()) {
      node->init = NormalizePredicates(op->init.value());
    }
    node->body = body;
    return block;
  }

  // Deliberately unused now that admission runs through the shared scan; kept
  // out to avoid a second, divergent definition of "this region has a unit". //
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
        // `node` is an expression position too: a UB load there carries an
        // access predicate Verify admitted, so the same clearing has to reach
        // it. Non-expression metadata (a Buffer, a Var) is left untouched.
        PrimExpr node_expr;
        if (op->node.as<PrimExprNode>()) {
          node_expr = Downcast<PrimExpr>(op->node);
        }
        const PrimExpr rewritten_node =
            node_expr.defined() ? VisitExpr(node_expr) : PrimExpr();
        Stmt result = StmtExprMutator::VisitStmt_(op);
        LeaveScope(&saved);
        if (rewritten_node.defined() && !rewritten_node.same_as(node_expr)) {
          AttrStmt attr = Downcast<AttrStmt>(result);
          AttrStmtNode *node = attr.CopyOnWrite();
          node->node = rewritten_node;
          return attr;
        }
        return result;
      }
      // The realize header (condition and iteration values) is evaluated in the
      // enclosing scope, before the block variables exist. It is a predicate
      // position like any other: a load there carries an access predicate that
      // Verify admitted, so the same clearing has to reach it here.
      Stmt VisitStmt_(const SBlockRealizeNode *op) final {
        // `predicate` is a plain PrimExpr here (never an Optional): a realize
        // always carries one, defaulting to `const_true`.
        const PrimExpr predicate = VisitExpr(op->predicate);
        bool changed = !predicate.same_as(op->predicate);
        Array<PrimExpr> iter_values;
        for (const PrimExpr &value : op->iter_values) {
          const PrimExpr rewritten = VisitExpr(value);
          if (!rewritten.same_as(value)) {
            changed = true;
          }
          iter_values.push_back(rewritten);
        }
        SBlock block = GetRef<SBlock>(op->block.get());
        // The block owns its own scope: its initializer runs before the body
        // and is an expression/statement position like any other, so it is
        // visited first and gets the same predicate clearing.
        if (block->init.defined()) {
          const Stmt init = VisitStmt(block->init.value());
          if (!init.same_as(block->init.value())) {
            block.CopyOnWrite()->init = init;
            changed = true;
          }
        }
        const Stmt body = VisitStmt(block->body);
        if (!body.same_as(block->body)) {
          block.CopyOnWrite()->body = body;
          changed = true;
        }
        if (!changed) {
          return GetRef<Stmt>(op);
        }
        SBlockRealize realize = GetRef<SBlockRealize>(op);
        SBlockRealizeNode *node = realize.CopyOnWrite();
        node->predicate = predicate;
        node->iter_values = iter_values;
        node->block = block;
        return realize;
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
    if (!in_simd_vf_ || op->kind != ForKind::kParallel ||
        pto::UsesExplicitSIMD(op->body)) {
      // An explicit `tl.simd.*` nest is hand-written: Verify and Vectorize skip
      // it, so Legalize must not fold conditionals inside it either. The loop
      // range is still recorded for a nested converting region.
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
      if (!bind->value.dtype().is_handle()) {
        // The guard-dependency analysis only reads which symbols a condition
        // depends on, so every scalar Bind is recorded here -- including a
        // value that may not be re-evaluated (a snapshot of memory): the
        // dependency set is a superset either way, which keeps the verdict
        // conservative.
        guard_bind_env_.Set(bind->var, bind->value);
        if (pto::IsSafelyReevaluableValue(bind->value)) {
          outer_binds_.Set(bind->var, bind->value);
        }
      }
    }
    return result;
  }

  // Folding evaluates both arms for every iteration of the unit, so a guard can
  // no longer be what makes an arm safe to evaluate. Division and modulo by a
  // value that may be zero are the operations where that matters: the source
  // skips them while the guard is false, the fold does not.
  bool ArmIsSafelyEvaluable(const Stmt &arm) {
    bool safe = true;
    PostOrderVisit(arm, [&](const ObjectRef &node) {
      if (!safe) {
        return;
      }
      const PrimExpr *divisor = nullptr;
      if (const auto *div = node.as<DivNode>()) {
        divisor = &div->b;
      } else if (const auto *mod = node.as<ModNode>()) {
        divisor = &mod->b;
      } else if (const auto *fdiv = node.as<FloorDivNode>()) {
        // Address arithmetic admits floordiv/floormod, so they are just as
        // eager as div/mod once the guard is folded away.
        divisor = &fdiv->b;
      } else if (const auto *fmod = node.as<FloorModNode>()) {
        divisor = &fmod->b;
      }
      if (divisor != nullptr && !predicate_analyzer_.CanProve(
                                    *divisor != make_zero(divisor->dtype()))) {
        // A zero literal must carry the divisor's own dtype: IntImm rejects a
        // floating-point dtype, and a float division is just as unsafe when the
        // divisor may be zero.
        safe = false;
      }
    });
    return safe;
  }

  // A guard may be the only reason an access stays inside its buffer (a guard
  // such as `i < 64` covering the first chunk of a two-chunk unit, or `row < 2`
  // on a runtime row index). The fold evaluates both arms, and the synthesized
  // store, for every iteration, so an access that leaves its buffer for the
  // iterations where the guard is false makes the rewrite illegal -- reads
  // included: `if i < 64: y[i] = a[i] else: y[i] = 0` reads `a[64..127]` after
  // the fold even though those lanes select the constant.
  bool AccessMayLeaveBounds(const Stmt &arm, const PrimExpr &condition) {
    bool leaves = false;
    PostOrderVisit(arm, [&](const ObjectRef &node) {
      if (leaves) {
        return;
      }
      if (const auto *store = node.as<BufferStoreNode>()) {
        leaves =
            IndicesMayLeaveBuffer(store->buffer, store->indices, condition);
      } else if (const auto *load = node.as<BufferLoadNode>()) {
        leaves = IndicesMayLeaveBuffer(load->buffer, load->indices, condition);
      }
    });
    return leaves;
  }

  // Judged per logical axis against the view shape. A physical-offset check
  // against the product of the shape is wrong twice over: a strided view
  // (`shape (2, 64)`, `strides (128, 1)`) legitimately stores at offsets
  // 128..191 inside a 192-element allocation, and a physical range does not say
  // which logical index left the view.
  bool IndicesMayLeaveBuffer(const Buffer &buffer,
                             const Array<PrimExpr> &indices,
                             const PrimExpr &condition) {
    if (indices.size() != buffer->shape.size()) {
      return false;
    }
    // A bound wider than the int32 coordinate range cannot come from a loop
    // coordinate: it comes from an unconstrained runtime symbol (or from an
    // overflowed bound computation). Such an axis normally follows the
    // program's own contract -- pinning the symbol would test one address, not
    // the ones it can take -- so it is skipped and the remaining axes are still
    // judged.
    const int64_t kRuntimeBound = int64_t{1} << 31;
    const std::set<const VarNode *> constrained = ConditionSymbols(condition);
    for (size_t axis = 0; axis < indices.size(); ++axis) {
      // Scalar `Bind` values in scope are known expressions, not opaque
      // symbols: expand them so `r = row + 1` is judged as `row + 1`. The
      // region-entry snapshot (`predicate_env_`) covers the enclosing scope;
      // the fold-site Bind environment covers aliases defined inside the unit,
      // which is what makes `j = row; if row < 2: y[j, i] = 1` decidable. The
      // expansion reuses the shared resolver, so its depth and snapshot rules
      // are the access analyzer's, not a second copy of them.
      const PrimExpr base_index =
          predicate_env_.empty() ? indices[axis]
                                 : Substitute(indices[axis], predicate_env_);
      const IndexExpansion expansion = ExpandIndexAliases(base_index);
      const PrimExpr &index = expansion.expr;
      const arith::ConstIntBound bound =
          predicate_analyzer_.const_int_bound(index);
      if (bound->min_value <= -kRuntimeBound ||
          bound->max_value >= kRuntimeBound) {
        // Unprovable. Accepting is the caller's contract *only* while neither
        // the index nor the destination extent is what the folded condition
        // constrains: the fold evaluates the access for every lane, so an
        // address that fits only under the guard must not become
        // unconditional. `if 0 <= row < 2: y[row, i] = 1` on a (2, 64) buffer
        // is the index case; `if i < n: y[i + 32 * step]` on `(n,)` is the
        // extent case, whose extent check would otherwise be skipped by the
        // `continue` below.
        if (!expansion.resolved) {
          out_of_bounds_detail_ =
              "buffer `" + std::string(buffer->name) + "` axis " +
              std::to_string(axis) +
              " keeps an index alias that cannot be resolved; the guard may be "
              "what keeps that address in range";
          return true;
        }
        if (GuardCarriesTheAccess(buffer, axis, index, constrained)) {
          return true;
        }
        if (GuardCarriesTheExtent(buffer, axis, constrained)) {
          return true;
        }
        continue; // runtime-driven axis, not guarded: the program's contract
      }
      const auto *dim = buffer->shape[axis].as<IntImmNode>();
      if (dim == nullptr) {
        // A dynamic extent cannot bound the upper end, but a provable range
        // still carries a meaningful lower end.
        if (bound->min_value < 0) {
          out_of_bounds_detail_ = "buffer `" + std::string(buffer->name) +
                                  "` axis " + std::to_string(axis) +
                                  " is indexed below zero, range [" +
                                  std::to_string(bound->min_value) + ", " +
                                  std::to_string(bound->max_value) + "]";
          return true;
        }
        // The upper end cannot be compared with a symbolic extent. That is the
        // caller's contract when the extent is independent of the folded
        // condition -- but `if i < n: y[i] = 1` on `(n,)` uses the same `n`:
        // the fold stores every lane of the unit, so `n == 64` with 128 lanes
        // leaves the buffer even though the source never did. Accept only when
        // the extent provably covers the whole index range.
        const PrimExpr extent = buffer->shape[axis];
        if (predicate_analyzer_.CanProve(
                extent > make_const(extent.dtype(), bound->max_value))) {
          continue;
        }
        if (GuardCarriesTheExtent(buffer, axis, constrained)) {
          return true;
        }
        continue;
      }
      if (bound->min_value < 0 || bound->max_value >= dim->value) {
        out_of_bounds_detail_ =
            DescribeOutOfBounds(buffer, axis, dim->value, bound);
        return true;
      }
    }
    return false;
  }

  // True when the folded condition constrains a runtime symbol the access index
  // depends on, i.e. the guard is what keeps the address in range.
  bool GuardCarriesTheAccess(const Buffer &buffer, size_t axis,
                             const PrimExpr &index,
                             const std::set<const VarNode *> &constrained) {
    for (const VarNode *symbol : UnresolvedSymbols(index)) {
      if (constrained.count(symbol) == 0) {
        continue;
      }
      out_of_bounds_detail_ = "buffer `" + std::string(buffer->name) +
                              "` axis " + std::to_string(axis) +
                              " uses runtime symbol `" + symbol->name_hint +
                              "` that the folded condition constrains; the "
                              "guard is the only reason this access fits";
      return true;
    }
    return false;
  }

  // True when the folded condition constrains a symbol of the destination's
  // extent expression, i.e. the guard is what keeps the access inside the
  // buffer's logical extent.
  bool GuardCarriesTheExtent(const Buffer &buffer, size_t axis,
                             const std::set<const VarNode *> &constrained) {
    for (const VarNode *symbol : ConditionSymbols(buffer->shape[axis])) {
      if (constrained.count(symbol) == 0) {
        continue;
      }
      out_of_bounds_detail_ =
          "buffer `" + std::string(buffer->name) + "` axis " +
          std::to_string(axis) + " has symbolic extent `" + symbol->name_hint +
          "` that the folded condition is the only bound of; folding would "
          "store every lane of the unit";
      return true;
    }
    return false;
  }

  // Symbols inside `expr` whose own range the analyzer cannot bound: kernel
  // parameters and other unconstrained runtime values. A loop coordinate never
  // lands here, because every enclosing loop binds its range.
  std::set<const VarNode *> UnresolvedSymbols(const PrimExpr &expr) {
    std::set<const VarNode *> symbols;
    const int64_t kRuntimeBound = int64_t{1} << 31;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      const auto *var = node.as<VarNode>();
      if (var == nullptr || symbols.count(var) != 0) {
        return;
      }
      const arith::ConstIntBound bound =
          predicate_analyzer_.const_int_bound(GetRef<Var>(var));
      if (bound->min_value <= -kRuntimeBound ||
          bound->max_value >= kRuntimeBound) {
        symbols.insert(var);
      }
    });
    return symbols;
  }

  // Every symbol the folded condition mentions. Collected conservatively: a
  // condition that constrains `row` through any expression makes the guard a
  // candidate reason for the access to fit.
  /*! Result of expanding the coordinate aliases of one index. */
  struct IndexExpansion {
    /*! The index with every resolvable alias expanded. */
    PrimExpr expr;
    /*! False when an alias could not be resolved (a memory snapshot, or a chain
     * past the shared resolver's depth/node budget). */
    bool resolved = true;
  };

  // Expands pure coordinate aliases through the lexical Bind environment the
  // fold has seen. The shared resolver owns the rules (depth, node budget, and
  // "a definition that reads memory is a snapshot, never an address"), so this
  // pass cannot drift from the access analyzer.
  IndexExpansion ExpandIndexAliases(const PrimExpr &expr) {
    IndexExpansion expansion{expr, true};
    if (guard_bind_env_.empty()) {
      return expansion;
    }
    const Optional<PrimExpr> resolved =
        pto::ResolveIndexAliases(expr, guard_bind_env_);
    if (!resolved.defined()) {
      expansion.resolved = false;
      return expansion;
    }
    expansion.expr = resolved.value();
    return expansion;
  }

  // The coordinates the folded condition *compares*, i.e. the symbols whose
  // range the guard narrows. A symbol that only appears inside a buffer access
  // does not count: `a[K] > 0` tests the loaded value, it does not bound `K`.
  class CoordinateSymbols : public tirx::ExprVisitor {
  public:
    explicit CoordinateSymbols(std::set<const VarNode *> *out) : out_(out) {}
    // The value of a load is not a constraint on any coordinate it indexes.
    void VisitExpr_(const tirx::BufferLoadNode *) final {}
    void VisitExpr_(const tirx::VarNode *op) final { out_->insert(op); }

  private:
    std::set<const VarNode *> *out_;
  };

  std::set<const VarNode *> ConditionSymbols(const PrimExpr &condition) {
    std::set<const VarNode *> symbols;
    std::set<const VarNode *> expanded;
    std::function<void(const PrimExpr &)> collect = [&](const PrimExpr &expr) {
      if (const auto *and_node = expr.as<AndNode>()) {
        collect(and_node->a);
        collect(and_node->b);
        return;
      }
      if (const auto *or_node = expr.as<OrNode>()) {
        collect(or_node->a);
        collect(or_node->b);
        return;
      }
      if (const auto *not_node = expr.as<NotNode>()) {
        collect(not_node->a);
        return;
      }
      PrimExpr lhs;
      PrimExpr rhs;
      if (pto::MatchRelational(expr, &lhs, &rhs) != nullptr) {
        CoordinateSymbols collector(&symbols);
        collector(lhs);
        collector(rhs);
        return;
      }
      // A bare scalar bool (`if flag`) is itself the constraint; analyse its
      // definition the same way so `flag = row < 2` keeps constraining `row`.
      if (const auto *var = expr.as<VarNode>()) {
        symbols.insert(var);
        if (expanded.insert(var).second) {
          auto it = guard_bind_env_.find(GetRef<Var>(var));
          if (it != guard_bind_env_.end()) {
            collect((*it).second);
          }
        }
        return;
      }
      CoordinateSymbols collector(&symbols);
      collector(expr);
    };
    collect(condition);
    return symbols;
  }

  std::string DescribeOutOfBounds(const Buffer &buffer, size_t axis,
                                  int64_t extent,
                                  const arith::ConstIntBound &bound) {
    return "buffer `" + std::string(buffer->name) + "` axis " +
           std::to_string(axis) + " extent " + std::to_string(extent) +
           ", index range [" + std::to_string(bound->min_value) + ", " +
           std::to_string(bound->max_value) + "]";
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
    // Folding evaluates both arms for every iteration, so an arm may only hold
    // operations that are safe to evaluate even where its guard is false.
    if (!ArmIsSafelyEvaluable(then_case) ||
        (else_case.defined() && !ArmIsSafelyEvaluable(else_case.value()))) {
      Fail("a conditional arm divides or takes a modulo by a divisor that is "
           "not provably nonzero; folding would evaluate that arm "
           "unconditionally, so keep the branch or make the divisor provably "
           "nonzero");
    }
    // The same applies to memory, reads and writes alike: a guard is allowed to
    // be the only thing keeping an access in bounds, and the fold drops it.
    if (AccessMayLeaveBounds(then_case, op->condition) ||
        (else_case.defined() &&
         AccessMayLeaveBounds(else_case.value(), op->condition))) {
      Fail("a guarded access can leave its buffer for iterations where the "
           "guard is false; folding would evaluate that read or write "
           "unconditionally (" +
           out_of_bounds_detail_ + ")");
    }
    // The condition travels into the value choices, so it needs the same
    // rewriting as the arms (Min/Max -> Select, infinity -> FloatImm).
    // If it reads a destination written by this If, multiple output stores must
    // share one entry-value snapshot instead of re-evaluating the condition
    // after an earlier store has changed it.
    PrimExpr condition = VisitExpr(op->condition);
    const bool condition_conflicts =
        ConditionConflictsWithWrites(condition, then_case, else_case);
    std::vector<AlignedTask> aligned;
    if (!AlignArmDestinations(then_case, else_case, &aligned)) {
      aligned = IdentityAlignment(then_case, else_case);
    }
    return EmitAlignedPairs(condition, condition_conflicts, aligned);
  }

  // Min/Max -> Select so Vectorize only needs cmp + vsel. Rewritten only
  // inside a T.Parallel unit in a converting region: scalar expressions in
  // sibling serial code within the same SIMD_VF keep their Min/Max shape.
  // Min/Max stay nodes when the direct VMI instruction is usable: integers
  // always (integer comparison has no NaN or signed-zero ambiguity), floating
  // point only under `tl.pto.relaxed_minmax`, whose default accepts the
  // instruction's NaN and signed-zero semantics. The supported set is the VMI
  // element set for the instruction: 8/16/32-bit integers, f16/bf16/f32 and
  // FP8; wider floats keep the comparison plus Select form.
  bool KeepMinMaxNode(const DataType &dtype) const {
    if (!pto::IsSupportedVectorBinaryOp("vmin", dtype)) {
      return false;
    }
    return dtype.is_int() || dtype.is_uint() || relaxed_minmax_;
  }

  PrimExpr VisitExpr_(const MinNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    if (KeepMinMaxNode(op->dtype)) {
      return Min(a, b);
    }
    return Select(a < b, a, b);
  }
  PrimExpr VisitExpr_(const MaxNode *op) final {
    if (!in_parallel_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    if (KeepMinMaxNode(op->dtype)) {
      return Max(a, b);
    }
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
          return FloatImm(op->dtype, std::numeric_limits<double>::infinity());
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  // Conservative storage test for *ordering*: two accesses may conflict unless
  // they name different known origins (a parameter or an allocation). Views
  // over the same storage conflict even when their logical indices differ,
  // because A[1,i] and Av[0,i] can denote the same physical address. The
  // judgment is the same one Verify applies (pto::MayConflictStorage), so the
  // two passes cannot classify a pair differently.
  bool MayConflict(const Buffer &lhs_buffer, const Buffer &rhs_buffer) {
    return pto::MayConflictStorage(lhs_buffer, rhs_buffer, storage_origins_);
  }

  bool SameArmLocation(const Buffer &lhs_buffer,
                       const Array<PrimExpr> &lhs_indices,
                       const Buffer &rhs_buffer,
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
    if (!lhs_buffer->data.same_as(rhs_buffer->data) ||
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
      out->push_back({stmt, GetRef<BufferStore>(store)});
      return true;
    }
    if (stmt.as<BindNode>() != nullptr) {
      out->push_back({stmt, BufferStore()});
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
      if (!task.store.defined()) {
        continue;
      }
      for (const LoadCollector::Location &load : condition_loads.loads) {
        if (MayConflict(task.store->buffer, load.buffer)) {
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
        (else_case.defined() &&
         !CollectPlanTasks(else_case.value(), &else_tasks))) {
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
                            const Optional<Stmt> &else_case,
                            std::vector<AlignedTask> *aligned) {
    std::vector<PlanTask> then_stores;
    std::vector<PlanTask> else_stores;
    if (!CollectPlanTasks(then_case, &then_stores)) {
      return false;
    }
    if (else_case.defined() &&
        !CollectPlanTasks(else_case.value(), &else_stores)) {
      return false;
    }
    if (then_stores.empty() && else_stores.empty()) {
      return false;
    }

    const size_t n = then_stores.size();
    const size_t m = else_stores.size();

    // Precedence masks preserve RAW/WAR/WAW and Bind definition/use edges.
    // A condition Bind reads its source buffers but writes none; its SSA
    // variable still has to dominate every store that references it.
    const auto collect_reads = [](const PlanTask &task) {
      LoadCollector reads;
      if (task.store.defined()) {
        reads(task.store->value);
        for (const PrimExpr &index : task.store->indices) {
          reads(index);
        }
      } else {
        reads(task.stmt);
      }
      return reads.loads;
    };
    const auto build_predecessors =
        [this, &collect_reads](const std::vector<PlanTask> &seq) {
          const size_t k = seq.size();
          std::vector<TaskMask> pred(k, TaskMask(k));
          // One read footprint and one used-Var set per task: the RAW/WAR and
          // Bind-definition checks below reuse them instead of walking seq[j]
          // again for every earlier task. The dependency graph is unchanged.
          std::vector<std::vector<LoadCollector::Location>> reads(k);
          std::vector<std::set<const VarNode *>> used_vars(k);
          for (size_t i = 0; i < k; ++i) {
            reads[i] = collect_reads(seq[i]);
            PostOrderVisit(seq[i].stmt, [&](const ObjectRef &node) {
              if (const auto *var = node.as<VarNode>()) {
                used_vars[i].insert(var);
              }
            });
          }
          for (size_t i = 0; i < k; ++i) {
            const auto &reads_i = reads[i];
            for (size_t j = i + 1; j < k; ++j) {
              const auto &reads_j = reads[j];
              bool dep = false;
              if (const auto *bind_i = seq[i].stmt.as<BindNode>()) {
                dep = used_vars[j].count(bind_i->var.get()) != 0;
              }
              if (seq[i].store.defined() && seq[j].store.defined() &&
                  MayConflict(seq[i].store->buffer, seq[j].store->buffer)) {
                dep = true; // WAW
              }
              if (seq[i].store.defined()) {
                for (const LoadCollector::Location &load : reads_j) {
                  if (MayConflict(seq[i].store->buffer, load.buffer)) {
                    dep = true; // earlier write, later read
                  }
                }
              }
              if (seq[j].store.defined()) {
                for (const LoadCollector::Location &load : reads_i) {
                  if (MayConflict(seq[j].store->buffer, load.buffer)) {
                    dep = true; // earlier read, later overwrite
                  }
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
    std::vector<PlanTask> representatives;
    const auto same_destination = [this](const PlanTask &lhs,
                                         const PlanTask &rhs) {
      return lhs.store.defined() && rhs.store.defined() &&
             SameArmLocation(lhs.store->buffer, lhs.store->indices,
                             rhs.store->buffer, rhs.store->indices);
    };
    const auto label_index =
        [this, &label_masks, &representatives, n, m,
         &same_destination](const PlanTask &task) -> size_t {
      for (size_t k = 0; k < label_masks.size(); ++k) {
        if (same_destination(representatives[k], task)) {
          return k;
        }
      }
      label_masks.push_back({TaskMask(n), TaskMask(m)});
      representatives.push_back(task);
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
    const auto lower_bound = [&label_masks, full_t,
                              full_e](const TaskMask &done_t,
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

    // The exact search is exponential in the number of unpaired stores: a
    // pathological arm (many independent then-only stores that the lower bound
    // still hopes to pair) can enumerate a combinatorial number of states. Cap
    // it and let the caller fall back to the identity order, which preserves
    // each arm's sequence.
    search_states_ = 0;
    budget_exhausted_ = false;
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
      if (++search_states_ > kAlignmentSearchBudget) {
        budget_exhausted_ = true;
        return kUnreachableSteps;
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
          if (!same_destination(then_stores[i], else_stores[j])) {
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

    std::vector<AlignedTask> then_seq;
    TaskMask done_t(n);
    TaskMask done_e(m);
    while (done_t != full_t || done_e != full_e) {
      if (budget_exhausted_) {
        return false; // caller uses IdentityAlignment
      }
      const int total = best_steps(done_t, done_e);
      bool emitted = false;
      // Prefer a pair: it keeps the position count minimal and avoids a
      // destination read-back on the missing side.
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
          if (!same_destination(then_stores[i], else_stores[j])) {
            continue;
          }
          if (1 + best_steps(done_t.With(i), done_e.With(j)) == total) {
            then_seq.emplace_back(then_stores[i].stmt, else_stores[j].stmt);
            done_t.Set(i);
            done_e.Set(j);
            emitted = true;
          }
        }
      }
      for (size_t i = 0; i < n && !emitted; ++i) {
        if (!done_t.Test(i) && !pred_t[i].AnyOutside(done_t) &&
            1 + best_steps(done_t.With(i), done_e) == total) {
          then_seq.emplace_back(then_stores[i].stmt, Optional<Stmt>());
          done_t.Set(i);
          emitted = true;
        }
      }
      for (size_t j = 0; j < m && !emitted; ++j) {
        if (!done_e.Test(j) && !pred_e[j].AnyOutside(done_e) &&
            1 + best_steps(done_t, done_e.With(j)) == total) {
          then_seq.emplace_back(Optional<Stmt>(), else_stores[j].stmt);
          done_e.Set(j);
          emitted = true;
        }
      }
      if (!emitted) {
        return false; // unreachable: a single store move always exists
      }
    }

    aligned->insert(aligned->end(), then_seq.begin(), then_seq.end());
    return true;
  }

  Stmt EmitAlignedPairs(const PrimExpr &condition, bool condition_conflicts,
                        const std::vector<AlignedTask> &aligned) {
    size_t store_count = 0;
    for (const AlignedTask &task : aligned) {
      if (task.first.as<BufferStoreNode>() != nullptr ||
          task.second.as<BufferStoreNode>() != nullptr) {
        ++store_count;
      }
    }

    // One final store is safe even when its condition reads the destination:
    // there is no later store that could re-evaluate the condition after a
    // write. Multiple stores share one entry snapshot only when the condition
    // aliases a write. Conditions over unchanged inputs need no extra Bind.
    const bool snapshot = condition_conflicts && store_count > 1;
    Var condition_var;
    PrimExpr effective_condition = condition;
    Array<Stmt> flat;
    if (snapshot) {
      condition_var =
          Var("cond_snapshot_" + std::to_string(condition_serial_++),
              condition.dtype());
      effective_condition = condition_var;
      flat.push_back(Bind(condition_var, condition));
    }

    for (const AlignedTask &task : aligned) {
      const auto *then_store = task.first.as<BufferStoreNode>();
      const auto *else_store = task.second.as<BufferStoreNode>();
      if (then_store == nullptr && else_store == nullptr) {
        if (task.first.defined()) {
          flat.push_back(task.first.value());
        }
        if (task.second.defined()) {
          flat.push_back(task.second.value());
        }
        continue;
      }
      const BufferStoreNode *shape =
          then_store != nullptr ? then_store : else_store;
      if (then_store != nullptr && else_store != nullptr &&
          !SameArmLocation(then_store->buffer, then_store->indices,
                           else_store->buffer, else_store->indices)) {
        Fail("the two arms do not write the same destination after alignment");
      }
      const PrimExpr then_value =
          then_store != nullptr ? then_store->value
                                : BufferLoad(shape->buffer, shape->indices);
      const PrimExpr else_value =
          else_store != nullptr ? else_store->value
                                : BufferLoad(shape->buffer, shape->indices);
      flat.push_back(BufferStore(
          shape->buffer, Select(effective_condition, then_value, else_value),
          shape->indices));
    }
    return flat.empty() ? Stmt(Evaluate(0)) : Stmt(SeqStmt::Flatten(flat));
  }
  // tl.pto.relaxed_minmax (default true): floating-point Min/Max may use the
  // direct VMI instruction, accepting its NaN and signed-zero semantics.
  bool relaxed_minmax_ = true;
  bool in_simd_vf_ = false;
  bool in_parallel_ = false;
  int condition_serial_ = 0;
  // Exact alignment search guard rails (see AlignArmDestinations): the search
  // is exponential, so it is capped and falls back to the identity order.
  static constexpr int64_t kAlignmentSearchBudget = 200000;
  static constexpr int kUnreachableSteps = 1 << 20;
  std::string out_of_bounds_detail_;
  int64_t search_states_ = 0;
  bool budget_exhausted_ = false;
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
  // Scalar `Bind` definitions seen by the fold, used to expand a condition that
  // is just a Var into the symbols its definition depends on.
  Map<Var, PrimExpr> guard_bind_env_;
  std::vector<std::pair<Var, Range>> outer_ranges_;
};

PrimFunc LowerControlFlow(PrimFunc f, bool relaxed_minmax) {
  auto *n = f.CopyOnWrite();
  ControlFlowMutator mutator;
  mutator.relaxed_minmax_ = relaxed_minmax;
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
  mutator.storage_origins_ = pto::CollectStorageOrigins(n->body, n->buffer_map,
                                                        no_alias, non_restrict);
  n->body = mutator(std::move(n->body));
  return f;
}

} // namespace

namespace transform {

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    // Floating-point Min/Max may use the direct VMI instruction unless the
    // caller opts into the NaN- and signed-zero-preserving Select lowering.
    const bool relaxed_minmax =
        ctx->GetConfig<Bool>("tl.pto.relaxed_minmax", Bool(true))
            .value()
            ->value;
    return LowerControlFlow(std::move(f), relaxed_minmax);
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
