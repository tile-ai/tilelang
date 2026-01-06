#include "layout_reducer.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ffi/base_details.h"
#include "tvm/ffi/object.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/var.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm::tl {

using namespace tir;

namespace {

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(Var var, PrimExpr val) : kind(kBindValue), var(var), value(val) {};
  Constr(Var var, Range range) : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      return var == value;
    case kBindRange:
      return And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  Constr Substitute(ffi::Map<Var, PrimExpr> subs) const {
    return Constr(tir::Substitute(ToGenericConstr(), subs));
  }
  void Populate(arith::Analyzer &analyzer) const {
    switch (kind) {
    case kConstr:
      analyzer.EnterConstraint(value);
      break;
    case kBindValue:
      analyzer.Bind(var, value);
      break;
    case kBindRange:
      analyzer.Bind(var, range);
      break;
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<Var, PrimExpr> subs) const {
    ConstrSet new_set;
    for (const auto &c : constrs_) {
      new_set.constrs_.push_back(c.Substitute(subs));
    }
    return new_set;
  }
  void Populate(arith::Analyzer &analyzer) const {
    for (const auto &c : constrs_) {
      c.Populate(analyzer);
    }
  }
  bool CanProve(const PrimExpr &expr) const {
    arith::Analyzer analyzer;
    Populate(analyzer);
    return analyzer.CanProve(expr);
  }
  template <typename... Args> void AddConstr(Args... args) {
    constrs_.push_back(Constr(args...));
  }
  void Extend(const ConstrSet &other) {
    for (const auto &c : other.constrs_) {
      constrs_.push_back(c);
    }
  }
  std::vector<Constr> constrs_;
};

struct ConstrVisitor : public StmtExprVisitor {
private:
  using Base = StmtExprVisitor;
  struct Guard {
    std::vector<Constr> &constrs;
    ~Guard() { constrs.pop_back(); }
  };
  template <typename... Args> Guard MakeGuard(const Args... args) {
    constr_stack_.push_back(Constr(args...));
    return Guard{constr_stack_};
  }

public:
  void VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                           const PrimExpr false_value) {
    {
      auto guard = MakeGuard(cond);
      Base::VisitExpr(true_value);
    }
    {
      auto guard = MakeGuard(Not(cond));
      Base::VisitExpr(false_value);
    }
  }
  void VisitStmt_(const LetStmtNode *op) override {
    auto guard = MakeGuard(op->var, op->value);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == tir::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
      Base::VisitStmt_(op);
    } else if (op->attr_key == tir::attr::thread_extent ||
               op->attr_key == tir::attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      Range dom = Range::FromMinExtent(make_zero(op->value.dtype()), op->value);
      auto guard = MakeGuard(iv->var, dom);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const AssertStmtNode *op) override {
    auto guard = MakeGuard(op->condition);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const IfThenElseNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      auto guard = MakeGuard(Not(op->condition));
      Base::VisitStmt(op->else_case.value());
    }
  }
  void VisitExpr_(const SelectNode *op) override {
    VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }
  void VisitExpr_(const CallNode *op) override {
    static auto op_if_then_else = Op::Get("tir.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      Base::VisitExpr_(op);
    }
  }
  void VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kParallel || op->kind == ForKind::kVectorized) {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  std::vector<Constr> constr_stack_;
};

struct ParallelLoopVerifier : public ConstrVisitor {
  std::vector<Var> parallel_loop_vars_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> reducers;

  void VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kParallel) {
      parallel_loop_vars_.push_back(op->loop_var);
      ConstrVisitor::VisitStmt_(op);
      parallel_loop_vars_.pop_back();
    } else {
      ConstrVisitor::VisitStmt_(op);
    }
  }
  void VisitStmt_(const BufferStoreNode *op) override {
    if (reducers.count(op->buffer->data)) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    ConstrSet cset{constr_stack_};
    std::vector<Var> other_thread_vars_;
    ffi::Map<Var, PrimExpr> subs;
    for (const auto &var : parallel_loop_vars_) {
      Var v_other_thread(var->name_hint + "<OTHER>", var->dtype);
      other_thread_vars_.push_back(v_other_thread);
      subs.Set(var, v_other_thread);
    }
    cset.Extend(cset.Substitute(subs));
    for (const auto &idx : op->indices) {
      cset.AddConstr(idx == tir::Substitute(idx, subs));
    }
    arith::Analyzer analyzer;
    cset.Populate(analyzer);
    // If we can prove the values are the same, then no data race can happen.
    if (analyzer.CanProve(op->value == tir::Substitute(op->value, subs))) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    ffi::Array<Var> failed_vars;
    PrimExpr failed_var_expr;
    for (auto [k, v] : subs) {
      if (!analyzer.CanProve(k == v)) {
        failed_vars.push_back(k);
        failed_var_expr =
            failed_var_expr.defined() ? And(failed_var_expr, k == v) : (k == v);
      }
    }
    if (!failed_vars.empty()) {
      LOG(FATAL) << "Potential data race detected: `" << op->buffer
                 << op->indices << "`"
                 << "is written by multiple threads of loop vars: "
                 << failed_vars << ", Counterexample:\n"
                 << analyzer.z3_prover.GetModel(failed_var_expr)
                 << "If you believe this is a false positive, pass "
                    "`PassKey.TL_DISABLE_DATA_RACE_CHECK` to pass key to "
                    "disable this check.";
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode *op) override {
    if (op->annotations.count(attr::kReducerInfo)) {
      auto map = op->annotations.Get(attr::kReducerInfo)
                     ->as<Map<Var, Map<String, String>>>();
      ICHECK(map) << "reducer_replication map is not defined";
      for (const auto &[var, info] : map.value()) {
        reducers.insert(var);
      }
    }
    return StmtExprVisitor::VisitStmt_(op);
  }
};

using namespace tir::transform;

tvm::transform::Pass VerifyParallelLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ParallelLoopVerifier verifier;
    verifier(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyParallelLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VerifyParallelLoop", VerifyParallelLoop);
}

} // namespace

} // namespace tvm::tl
