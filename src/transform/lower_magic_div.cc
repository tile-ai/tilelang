/*!
 * \file lower_magic_div.cc
 * \brief Lower divisions by launch-invariant dynamic shapes to host-precomputed
 *        magic-number multiply-shift sequences (CUTLASS FastDivmod style).
 *
 * For each FloorDiv/FloorMod/Div/Mod site whose divisor is a launch-invariant
 * positive int32 expression and whose dividend is a provably non-negative
 * int32 expression, rewrite:
 *
 *   floordiv(x, d)  ->  tl.magic_div(x, d_i, m_i, s_i)
 *   floormod(x, d)  ->  x - tl.magic_div(x, d_i, m_i, s_i) * d_i
 *
 * where (m_i, s_i) are fresh variables bound on the host before
 * SplitHostDevice, which threads them into the device parameter list like any
 * other symbolic variable. The intrinsic stays opaque until device codegen,
 * where it expands to a mul-high + shift sequence with a d == 1 fallback.
 *
 * Magic constants (valid for 0 <= x < 2^31, d >= 2):
 *   k = ceil_log2(d), p = 31 + k, M = ceil(2^p / d), s = p - 32
 *   q = umulhi(uint32(x), M) >> s
 * d == 1 is handled by a device-side select (host binds m = s = 0).
 *
 * Performance note: index expressions in this pipeline can be fully inlined
 * and huge. This pass deliberately avoids canonical_simplify / CanProve on
 * whole expressions; all checks are either structural (factor-spine walks,
 * sum-term decomposition) or O(n) interval evaluation.
 */

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using arith::IRMutatorWithAnalyzer;

namespace {

// Decompose a product into its factors (Mul spine walk).
std::vector<PrimExpr> CollectFactors(const PrimExpr &e) {
  std::vector<PrimExpr> factors;
  PrimExpr cur = e;
  while (const MulNode *mul = cur.as<MulNode>()) {
    factors.push_back(mul->b);
    cur = mul->a;
  }
  factors.push_back(cur);
  return factors;
}

// Structural multiset equality of two factor lists. canonical_simplify does
// not commute free products (e.g. `sn * n` vs `n * sn` stay distinct), so
// product equality up to reassociation must be checked this way. A miss only
// loses an optimization opportunity; a match is always sound.
bool FactorMultisetEqual(const std::vector<PrimExpr> &a,
                         const std::vector<PrimExpr> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  ExprDeepEqual deep_equal;
  std::vector<bool> used(b.size(), false);
  for (const PrimExpr &fa : a) {
    bool found = false;
    for (size_t j = 0; j < b.size(); j++) {
      if (!used[j] && deep_equal(fa, b[j])) {
        used[j] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

// Step 0: canonicalize `x - floordiv(x, d) * e` into `floormod(x, d)` when
// CanonicalSimplify(e - d) == 0 (i.e. the multiplied term is the divisor up to
// reassociation). The identity holds for any d != 0, so no positivity gate is
// required here; downstream logic decides whether the floormod is eligible
// for magic-number lowering.
class FloorModCanonicalizer : public IRMutatorWithAnalyzer {
public:
  explicit FloorModCanonicalizer(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  Stmt Apply(const Stmt &body) { return VisitStmt(body); }

  PrimExpr VisitExpr_(const SubNode *op) final {
    PrimExpr expr = IRMutatorWithAnalyzer::VisitExpr_(op);
    const SubNode *sub = expr.as<SubNode>();
    if (sub == nullptr || !sub->a.dtype().is_int() || !sub->b.as<MulNode>()) {
      return expr;
    }
    // Decompose the right operand into a product of factors (spine walk,
    // cheap even for huge expressions) and look for a FloorDiv factor whose
    // dividend matches the left operand: x - floordiv(x, d) * rest.
    std::vector<PrimExpr> factors = CollectFactors(sub->b);
    ExprDeepEqual deep_equal;
    for (size_t i = 0; i < factors.size(); i++) {
      const FloorDivNode *fd = factors[i].as<FloorDivNode>();
      if (fd == nullptr || !deep_equal(fd->a, sub->a)) {
        continue;
      }
      // The remaining factors must equal the divisor fd->b up to
      // reassociation (factor multiset equality).
      std::vector<PrimExpr> rest;
      for (size_t j = 0; j < factors.size(); j++) {
        if (j != i) {
          rest.push_back(factors[j]);
        }
      }
      if (rest.empty()) {
        continue; // b == floordiv alone is not a remainder pattern
      }
      if (FactorMultisetEqual(rest, CollectFactors(fd->b))) {
        return floormod(sub->a, fd->b);
      }
    }
    return expr;
  }
};

struct MagicDivisorEntry {
  PrimExpr divisor; // original divisor expression (host-side value for d_i)
  Var d_var;        // tl_magic_d_i : int32
  Var m_var;        // tl_magic_m_i : int32 (uint32 bit pattern)
  Var s_var;        // tl_magic_s_i : int32
};

class MagicDivRewriter : public IRMutatorWithAnalyzer {
public:
  MagicDivRewriter(arith::Analyzer *analyzer, const PrimFunc &func)
      : IRMutatorWithAnalyzer(analyzer) {
    MarkBufferMapShapes(func);
    for (const Var &param : func->params) {
      if (param->dtype.is_int() && param->dtype.lanes() == 1) {
        invariant_vars_.insert(param.get());
      }
    }
    auto collect_vars = [&](const PrimExpr &e) {
      PostOrderVisit(e, [&](const ObjectRef &n) {
        if (const VarNode *v = n.as<VarNode>()) {
          invariant_vars_.insert(v);
        }
      });
    };
    for (const auto &kv : func->buffer_map) {
      const Buffer &buf = kv.second;
      for (const PrimExpr &e : buf->shape) {
        collect_vars(e);
      }
      for (const PrimExpr &e : buf->strides) {
        collect_vars(e);
      }
      collect_vars(buf->elem_offset);
    }
  }

  Stmt Apply(const Stmt &body) {
    CollectConditionDivMods(body);
    return VisitStmt(body);
  }

  // The div/mod overrides gate on the *current* (not yet visited) node so the
  // dividend stays in its original, most analyzable form (e.g. floormod)
  // before nested sites inside it are rewritten. Rewritten expressions are
  // then visited to handle nested sites in their operands.
  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    if (PrimExpr r = TryRewriteDivMod(op, /*is_mod=*/false); r.defined()) {
      return VisitExpr(r);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (PrimExpr r = TryRewriteDivMod(op, /*is_mod=*/true); r.defined()) {
      return VisitExpr(r);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const DivNode *op) final {
    if (PrimExpr r = TryRewriteDivMod(op, /*is_mod=*/false); r.defined()) {
      return VisitExpr(r);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const ModNode *op) final {
    if (PrimExpr r = TryRewriteDivMod(op, /*is_mod=*/true); r.defined()) {
      return VisitExpr(r);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    // Keep a complete mixed-radix unflatten chain visible until
    // FlattenBuffer.  Replacing its quotient/remainder pieces with opaque
    // magic calls prevents the flattened store offset from simplifying back
    // to the original linear index.
    if (!IsLinearIndexUnflatten(op->indices, op->buffer->shape)) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }

    PrimExpr value = VisitExpr(op->value);
    Optional<PrimExpr> predicate = op->predicate;
    if (predicate.defined()) {
      predicate = VisitExpr(predicate.value());
    }
    if (value.same_as(op->value) && predicate.same_as(op->predicate)) {
      return GetRef<Stmt>(op);
    }
    return BufferStore(op->buffer, value, op->indices, predicate, op->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (!IsLinearIndexUnflatten(op->indices, op->buffer->shape)) {
      return IRMutatorWithAnalyzer::VisitExpr_(op);
    }
    Optional<PrimExpr> predicate = op->predicate;
    if (predicate.defined()) {
      predicate = VisitExpr(predicate.value());
    }
    if (predicate.same_as(op->predicate)) {
      return GetRef<PrimExpr>(op);
    }
    return BufferLoad(op->buffer, op->indices, predicate, op->span);
  }

  bool HasMagicDivisors() const { return !entries_.empty(); }

  // Bind the magic constants on the host: one libtvm_runtime helper call per
  // constant (multiplier / shift) for CUTLASS FastDivmod
  // (k = ceil_log2(d), p = 31 + k, M = ceil(2^p / d), s = p - 32; 0 for
  // d == 1 where the device expansion selects x).
  Stmt WrapWithHostBinds(const Stmt &body) {
    ICHECK(!entries_.empty());
    std::vector<Stmt> seq;
    for (size_t i = 0; i < entries_.size(); i++) {
      const MagicDivisorEntry &e = entries_[i];
      std::string suffix = std::to_string(i);
      seq.emplace_back(Bind(e.d_var, e.divisor));
      // Opaque extern calls: downstream analyzer-based passes see no
      // arithmetic chain (an earlier unrolled ceil_log2 ladder of arithmetic
      // binds caused multi-minute slowdowns in rewrite/canonical simplify
      // passes on large kernels), and the binds are not substituted into the
      // device body (m/s must stay separate kernel params: the runtime arg
      // packer rejects uint64, so a packed (M<<32)|s param is not an option).
      seq.emplace_back(
          Bind(e.m_var, Cast(DataType::Int(32),
                             Call(DataType::UInt(32), builtin::call_extern(),
                                  {StringImm("TileLangHostFastDivmodU32Mul"),
                                   Cast(DataType::UInt(32), e.d_var)}))));
      seq.emplace_back(
          Bind(e.s_var, Call(DataType::Int(32), builtin::call_extern(),
                             {StringImm("TileLangHostFastDivmodU32Shift"),
                              Cast(DataType::UInt(32), e.d_var)})));
    }
    seq.push_back(body);
    return SeqStmt(seq);
  }

private:
  bool IsMatchingDiv(const PrimExpr &expr, const PrimExpr &dividend,
                     const std::vector<PrimExpr> &divisor_factors) const {
    PrimExpr lhs;
    PrimExpr rhs;
    if (const FloorDivNode *div = expr.as<FloorDivNode>()) {
      lhs = div->a;
      rhs = div->b;
    } else if (const DivNode *div = expr.as<DivNode>()) {
      lhs = div->a;
      rhs = div->b;
    } else {
      return false;
    }
    return ExprDeepEqual()(lhs, dividend) &&
           FactorMultisetEqual(CollectFactors(rhs), divisor_factors);
  }

  bool IsMatchingMod(const PrimExpr &expr, const PrimExpr &dividend,
                     const std::vector<PrimExpr> &divisor_factors) const {
    PrimExpr lhs;
    PrimExpr rhs;
    if (const FloorModNode *mod = expr.as<FloorModNode>()) {
      lhs = mod->a;
      rhs = mod->b;
    } else if (const ModNode *mod = expr.as<ModNode>()) {
      lhs = mod->a;
      rhs = mod->b;
    } else {
      return false;
    }
    return ExprDeepEqual()(lhs, dividend) &&
           FactorMultisetEqual(CollectFactors(rhs), divisor_factors);
  }

  bool IsLinearIndexUnflatten(const Array<PrimExpr> &indices,
                              const Array<PrimExpr> &shape) const {
    if (indices.size() < 2 || indices.size() != shape.size()) {
      return false;
    }

    const auto *outer_div = indices[0].as<FloorDivNode>();
    if (outer_div == nullptr) {
      return false;
    }
    PrimExpr linear_index = outer_div->a;
    PrimExpr remainder;
    for (size_t i = 0; i + 1 < indices.size(); ++i) {
      std::vector<PrimExpr> stride_factors;
      for (size_t j = i + 1; j < shape.size(); ++j) {
        std::vector<PrimExpr> shape_factors = CollectFactors(shape[j]);
        stride_factors.insert(stride_factors.end(), shape_factors.begin(),
                              shape_factors.end());
      }
      const PrimExpr &dividend = i == 0 ? linear_index : remainder;
      if (!IsMatchingDiv(indices[i], dividend, stride_factors)) {
        return false;
      }
      if (i + 1 == indices.size() - 1) {
        return IsMatchingMod(indices.back(), dividend, stride_factors);
      }
      const auto *next_div = indices[i + 1].as<FloorDivNode>();
      if (next_div == nullptr ||
          !IsMatchingMod(next_div->a, dividend, stride_factors)) {
        return false;
      }
      remainder = next_div->a;
    }
    return false;
  }

  bool ProvePositive(const PrimExpr &e) {
    try {
      if (analyzer_->CanProve(e > 0, arith::ProofStrength::kSymbolicBound)) {
        return true;
      }
    } catch (const std::exception &) {
    }
    return analyzer_->const_int_bound(e)->min_value >= 1;
  }

  // Cheap structural proof for x >= 0 that avoids heavyweight symbolic
  // analysis on large expressions. Covers the shapes of bit-twiddled
  // permutation indices: floormod with a positive divisor, term-wise
  // non-negative sums, and the `a - floormod(a, c)` remainder pattern.
  bool ProveNonNegative(const PrimExpr &e) { return ProveNonNegative(e, 0); }

  bool ProveNonNegative(const PrimExpr &e, int depth) {
    try {
      if (analyzer_->const_int_bound(e)->min_value >= 0) {
        return true;
      }
    } catch (const std::exception &) {
    }
    if (depth >= 16) {
      return false;
    }
    if (const FloorModNode *fm = e.as<FloorModNode>()) {
      // floormod with a positive divisor is always non-negative.
      if (ProvePositive(fm->b)) {
        return true;
      }
    }
    if (const AddNode *add = e.as<AddNode>()) {
      return ProveNonNegative(add->a, depth + 1) &&
             ProveNonNegative(add->b, depth + 1);
    }
    if (const SubNode *sub = e.as<SubNode>()) {
      // a - floormod(a, c) >= 0 when a >= 0 and c > 0.
      if (const FloorModNode *fm = sub->b.as<FloorModNode>()) {
        ExprDeepEqual deep_equal;
        if (deep_equal(sub->a, fm->a) && ProveNonNegative(fm->a, depth + 1) &&
            ProvePositive(fm->b)) {
          return true;
        }
      }
    }
    return false;
  }

  // A divisor is launch-invariant when it is a positive int32 expression
  // whose only variables are scalar parameters or buffer shape/stride/offset
  // symbols (thread indices and loop variables are never in that set).
  bool IsInvariantDivisor(const PrimExpr &d) {
    if (d.dtype() != DataType::Int(32)) {
      return false;
    }
    bool ok = true;
    bool has_var = false;
    PostOrderVisit(d, [&](const ObjectRef &n) {
      if (n.as<CallNode>() || n.as<BufferLoadNode>()) {
        ok = false;
        return;
      }
      if (const VarNode *v = n.as<VarNode>()) {
        has_var = true;
        if (!invariant_vars_.count(v)) {
          ok = false;
        }
      }
    });
    if (!ok || !has_var) {
      return false;
    }
    return ProvePositive(d);
  }

  template <typename NodeT>
  PrimExpr TryRewriteDivMod(const NodeT *node, bool is_mod) {
    if (skip_nodes_.count(node)) {
      return PrimExpr();
    }
    const PrimExpr &x = node->a;
    const PrimExpr &d = node->b;
    if (x.dtype() != DataType::Int(32) || !IsInvariantDivisor(d) ||
        !ProveNonNegative(x)) {
      return PrimExpr();
    }
    size_t idx = GetOrCreateEntry(d);
    const MagicDivisorEntry &e = entries_[idx];
    // floormod is lowered to its own opaque intrinsic: exposing
    // x - q * d in TIR would give rewrite_simplify a distributable Sub/Mul
    // form and can send it into distribute/collect oscillation on large
    // expressions.
    const Op &op = is_mod ? tl::magic_mod() : tl::magic_div();
    return Call(DataType::Int(32), op, {x, e.d_var, e.m_var, e.s_var});
  }

  size_t GetOrCreateEntry(const PrimExpr &d) {
    // Dedup by factor-multiset identity: reassociated divisors
    // (ng*sn vs sn*ng) share one entry, while distinct same-named Vars
    // never merge. Entries are few, so a linear scan is fine.
    std::vector<PrimExpr> factors = CollectFactors(d);
    for (size_t i = 0; i < entries_.size(); i++) {
      if (FactorMultisetEqual(factors, CollectFactors(entries_[i].divisor))) {
        return i;
      }
    }
    size_t idx = entries_.size();
    std::string suffix = std::to_string(idx);
    MagicDivisorEntry e;
    e.divisor = d;
    e.d_var = Var("tl_magic_d_" + suffix, DataType::Int(32));
    e.m_var = Var("tl_magic_m_" + suffix, DataType::Int(32));
    e.s_var = Var("tl_magic_s_" + suffix, DataType::Int(32));
    entries_.push_back(e);
    return idx;
  }

  // Div/mod sites inside condition expressions (if guards, select
  // conditions) are NOT rewritten. Downstream analyzer-based passes run
  // rewrite_simplify over those conditions; floordiv/floormod forms are what
  // their rules expect, and opaque intrinsics there have caused extreme
  // slowdowns. Conditions are also exactly the guards whose provability the
  // floormod canonical form improves.
  void CollectConditionDivMods(const Stmt &body) {
    auto mark = [&](const PrimExpr &cond) {
      PostOrderVisit(cond, [&](const ObjectRef &c) {
        if (c->IsInstance<FloorDivNode>() || c->IsInstance<FloorModNode>() ||
            c->IsInstance<DivNode>() || c->IsInstance<ModNode>()) {
          skip_nodes_.insert(static_cast<const BaseExprNode *>(c.get()));
        }
      });
    };
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (const IfThenElseNode *ite = n.as<IfThenElseNode>()) {
        mark(ite->condition);
      } else if (const SelectNode *sel = n.as<SelectNode>()) {
        mark(sel->condition);
      } else if (const AssertStmtNode *asrt = n.as<AssertStmtNode>()) {
        mark(asrt->condition);
      } else if (const CallNode *call = n.as<CallNode>()) {
        if (call->op.same_as(builtin::if_then_else())) {
          mark(call->args[0]);
        }
      }
    });
  }

  std::unordered_set<const VarNode *> invariant_vars_;
  std::unordered_set<const BaseExprNode *> skip_nodes_;
  std::vector<MagicDivisorEntry> entries_;
};

namespace {

// Post-rewrite CSE for magic calls: calls with the same (x, d), including a
// div/mod pair duplicated by let-inlining, share one quotient per thread.
class MagicCallHoister {
public:
  static Stmt Apply(const Stmt &body) {
    // 1. Collect groups with the same (x, d) in post-order (inner first).
    // Div and mod share one quotient; the remainder is derived as x - q*d.
    struct Group {
      Call repr;
      std::vector<Call> div_calls;
      std::vector<Call> mod_calls;
      PrimExpr x;
      PrimExpr d;
      Var q;
      Var r;
    };
    std::vector<Group> groups;
    ExprDeepEqual deep_equal;
    PostOrderVisit(body, [&](const ObjectRef &n) {
      const CallNode *call = n.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      bool is_mod = call->op.same_as(tl::magic_mod());
      if (!is_mod && !call->op.same_as(tl::magic_div())) {
        return;
      }
      Call call_ref = GetRef<Call>(call);
      ICHECK_EQ(call->args.size(), 4U);
      for (Group &g : groups) {
        if (deep_equal(g.d, call->args[1]) && deep_equal(g.x, call->args[0])) {
          (is_mod ? g.mod_calls : g.div_calls).push_back(call_ref);
          return;
        }
      }
      Group g;
      g.repr = call_ref;
      g.x = call->args[0];
      g.d = call->args[1];
      g.q = Var("tl_magic_q_" + std::to_string(groups.size()), call->dtype);
      g.r = Var("tl_magic_r_" + std::to_string(groups.size()), call->dtype);
      (is_mod ? g.mod_calls : g.div_calls).push_back(call_ref);
      groups.push_back(std::move(g));
    });
    if (groups.empty()) {
      return body;
    }

    // 2. Find the innermost thread_extent AttrStmt containing all groups.
    std::unordered_set<Call, ObjectPtrHash, ObjectPtrEqual> targets;
    for (const Group &g : groups) {
      for (const Call &c : g.div_calls) {
        targets.insert(c);
      }
      for (const Call &c : g.mod_calls) {
        targets.insert(c);
      }
    }
    auto contains_all = [&](const Stmt &root) {
      std::unordered_set<const Object *> seen;
      PostOrderVisit(root, [&](const ObjectRef &n) {
        if (const auto *call = n.as<CallNode>();
            call != nullptr && targets.count(GetRef<Call>(call))) {
          seen.insert(n.get());
        }
      });
      return seen.size() == targets.size();
    };
    // Thread bindings are thread_extent AttrStmts at this pipeline stage;
    // pick the innermost one containing every group.
    Optional<AttrStmt> scope;
    std::function<void(const Stmt &)> walk = [&](const Stmt &cur_s) {
      if (const AttrStmtNode *a = cur_s.as<AttrStmtNode>()) {
        if (a->attr_key == tirx::attr::thread_extent && contains_all(a->body)) {
          scope = GetRef<AttrStmt>(a);
          // Walk continues deeper; the last hit is the innermost.
        }
        walk(a->body);
        return;
      }
      if (const ForNode *f = cur_s.as<ForNode>()) {
        walk(f->body);
        return;
      }
      if (const SeqStmtNode *s = cur_s.as<SeqStmtNode>()) {
        for (const Stmt &c : s->seq) {
          walk(c);
        }
      } else if (const SBlockNode *b = cur_s.as<SBlockNode>()) {
        walk(b->body);
      } else if (const IfThenElseNode *i = cur_s.as<IfThenElseNode>()) {
        walk(i->then_case);
        if (i->else_case.defined()) {
          walk(i->else_case.value());
        }
      }
    };
    walk(body);
    if (!scope.defined()) {
      return body; // no uniform thread scope covers every use; leave inline
    }

    // 3. Rewrite: replace each group call with its q var (bottom-up so nested
    // groups substitute first), and prepend the binds to the scope body.
    class Replacer : public StmtExprMutator {
    public:
      std::unordered_map<Call, PrimExpr, ObjectPtrHash, ObjectPtrEqual>
          replace_;
      AttrStmt scope_;
      explicit Replacer(AttrStmt scope) : scope_(std::move(scope)) {}

      PrimExpr Apply(const PrimExpr &e) { return VisitExpr(e); }
      Stmt Apply(const Stmt &s) { return VisitStmt(s); }

      PrimExpr VisitExpr_(const CallNode *op) final {
        auto it = replace_.find(GetRef<Call>(op));
        if (it != replace_.end()) {
          return it->second;
        }
        return StmtExprMutator::VisitExpr_(op);
      }

      Stmt VisitStmt_(const AttrStmtNode *op) final {
        if (GetRef<AttrStmt>(op).same_as(scope_)) {
          Stmt new_body = VisitStmt(op->body);
          std::vector<Stmt> seq;
          for (const auto &[var, value] : binds_) {
            seq.emplace_back(Bind(var, value));
          }
          seq.push_back(new_body);
          auto n = CopyOnWrite(op);
          n->body = SeqStmt(seq);
          return Stmt(n);
        }
        return StmtExprMutator::VisitStmt_(op);
      }

      std::vector<std::pair<Var, PrimExpr>> binds_;
    };

    Replacer replacer(scope.value());
    for (const Group &g : groups) {
      for (const Call &c : g.div_calls) {
        replacer.replace_[c] = g.q;
      }
      for (const Call &c : g.mod_calls) {
        replacer.replace_[c] = g.r;
      }
    }
    for (const Group &g : groups) {
      // Bind values are built by mutating the args only, so a group nested
      // inside x refers to the already-bound inner quotient/remainder.
      Array<PrimExpr> new_args;
      for (const PrimExpr &arg : g.repr->args) {
        new_args.push_back(replacer.Apply(arg));
      }
      if (g.div_calls.empty()) {
        replacer.binds_.emplace_back(g.r, Call(g.repr->dtype, tl::magic_mod(),
                                               new_args, g.repr->annotations,
                                               g.repr->span));
        continue;
      }
      replacer.binds_.emplace_back(g.q, Call(g.repr->dtype, tl::magic_div(),
                                             new_args, g.repr->annotations,
                                             g.repr->span));
      if (!g.mod_calls.empty()) {
        replacer.binds_.emplace_back(
            g.r, Call(g.repr->dtype, tl::magic_mod_from_quotient(),
                      {new_args[0], new_args[1], g.q}, g.repr->annotations,
                      g.repr->span));
      }
    }
    return replacer.Apply(body);
  }
};

} // namespace

} // namespace

tvm::transform::Pass LowerMagicDiv() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    if (!ctx->GetConfig<Bool>(kEnableMagicDiv, Bool(false)).value()) {
      return f;
    }
    Stmt body = f->body;
    {
      // The canonicalizer gets its own analyzer: IRMutatorWithAnalyzer feeds
      // Bind constraints into the shared analyzer, and a second visit of the
      // same binds with a differently-constrained context trips
      // ConstIntBoundAnalyzer's conflicting-update check.
      arith::Analyzer canon_analyzer;
      FloorModCanonicalizer canonicalizer(&canon_analyzer);
      body = canonicalizer.Apply(body);
    }
    arith::Analyzer analyzer;
    MagicDivRewriter rewriter(&analyzer, f);
    body = rewriter.Apply(body);
    if (!rewriter.HasMagicDivisors()) {
      return f;
    }
    f.CopyOnWrite()->body = rewriter.WrapWithHostBinds(body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerMagicDiv", {});
}

tvm::transform::Pass MagicCallHoist() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    if (!ctx->GetConfig<Bool>(kEnableMagicDiv, Bool(false)).value()) {
      return f;
    }
    f.CopyOnWrite()->body = MagicCallHoister::Apply(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MagicCallHoist", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.LowerMagicDiv", LowerMagicDiv);
  refl::GlobalDef().def("tl.transform.MagicCallHoist", MagicCallHoist);
}

} // namespace tl
} // namespace tvm
