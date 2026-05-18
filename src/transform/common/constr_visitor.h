#ifndef TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
#define TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_

#include "support/check.h"
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include <ostream>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>

namespace tvm::tl {

struct Constr {

  enum Kind {
    kConstr,
    kBindValue,
    kBindRange,
  } kind;
  bool is_assume = false;
  tirx::Var var;
  PrimExpr value;
  Range range;

  Constr(PrimExpr constr, bool is_assume = false)
      : kind(kConstr), value(constr), is_assume(is_assume) {};
  Constr(tirx::Var var, PrimExpr val)
      : kind(kBindValue), var(var), value(val) {};
  Constr(tirx::Var var, Range range)
      : kind(kBindRange), var(var), range(range) {};

  Constr() = default;
  Constr(const Constr &other) = default;
  Constr(Constr &&other) = default;
  Constr &operator=(const Constr &other) = default;

  void format(std::ostream &os) const {
    os << "Constr(kind=";
    switch (kind) {
    case kConstr:
      os << "kConstr";
      os << ", is_assume=" << (is_assume ? "true" : "false");
      os << ", value=" << value;
      break;
    case kBindValue:
      os << "kBindValue";
      os << ", var=" << var->name_hint;
      os << ", value=" << value;
      break;
    case kBindRange:
      os << "kBindRange";
      os << ", var=" << var->name_hint;
      os << ", range=Range(min=" << range->min;
      os << ", extent=" << range->extent << ")";
      break;
    default:
      os << "Unknown";
    }
    os << ")";
  }

  PrimExpr ToGenericConstr() const {
    switch (kind) {
    case kConstr:
      return value;
    case kBindValue:
      return var == value;
    case kBindRange:
      return tirx::And(var >= range->min, var < (range->min + range->extent));
    }
    LOG(FATAL) << "Unreachable";
    return PrimExpr();
  }
  Constr Substitute(ffi::Map<tirx::Var, PrimExpr> subs) const {
    return Constr(tirx::Substitute(ToGenericConstr(), subs));
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
    default:
      LOG(FATAL) << "Unreachable";
    }
  }
};

struct ConstrSet {
  ConstrSet Substitute(ffi::Map<tirx::Var, PrimExpr> subs) const {
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

  /*! \brief Convert the constraint set to a conjunction (AND) of all
   * constraints */
  PrimExpr ToConjunction() const {
    if (constrs_.empty())
      return Bool(true);
    PrimExpr result = constrs_[0].ToGenericConstr();
    for (size_t i = 1; i < constrs_.size(); ++i) {
      result = tirx::And(result, constrs_[i].ToGenericConstr());
    }
    return result;
  }

  void format(std::ostream &os) const {
    os << "ConstrSet(size=" << constrs_.size() << ") {\n";
    for (size_t i = 0; i < constrs_.size(); ++i) {
      os << "  [" << i << "] ";
      constrs_[i].format(os);
      os << "\n";
    }
    os << "}";
  }

  std::vector<Constr> constrs_;
};

struct ConstrVisitor : public tirx::StmtExprVisitor {
private:
  using Base = tirx::StmtExprVisitor;

  struct Guard {
    std::vector<Constr> &constrs;
    ~Guard() { constrs.pop_back(); }
  };

protected:
  template <typename... Args> Guard MakeGuard(const Args... args) {
    constr_stack_.push_back(Constr(args...));
    return Guard{constr_stack_};
  }

public:
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;
  void VisitIfThenElseExpr(const PrimExpr cond, const PrimExpr true_value,
                           const PrimExpr false_value) {
    // Visit the condition first without any guard, as it is always evaluated
    // This ensures any buffer accesses in the condition are recorded
    Base::VisitExpr(cond);
    {
      auto guard = MakeGuard(cond);
      Base::VisitExpr(true_value);
    }
    {
      auto guard = MakeGuard(tirx::Not(cond));
      Base::VisitExpr(false_value);
    }
  }
  void VisitStmt_(const tirx::BindNode *op) override {
    auto guard = MakeGuard(op->var, op->value);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const tirx::AttrStmtNode *op) override {
    if (op->attr_key == tirx::attr::tilelang_assume) {
      auto expr = Downcast<PrimExpr>(op->node);
      auto guard = MakeGuard(expr, true);
      Base::VisitStmt_(op);
    } else if (op->attr_key == tirx::attr::thread_extent ||
               op->attr_key == s_tir::attr::virtual_thread) {
      tirx::IterVar iv = Downcast<tirx::IterVar>(op->node);
      Range dom =
          Range::FromMinExtent(tirx::make_zero(op->value.dtype()), op->value);
      auto guard = MakeGuard(iv->var, dom);
      Base::VisitStmt_(op);
    } else {
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tirx::AssertStmtNode *op) override {
    auto guard = MakeGuard(op->condition);
    Base::VisitStmt_(op);
  }
  void VisitStmt_(const tirx::IfThenElseNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      auto guard = MakeGuard(tirx::Not(op->condition));
      Base::VisitStmt(op->else_case.value());
    }
  }
  void VisitExpr_(const tirx::SelectNode *op) override {
    VisitIfThenElseExpr(op->condition, op->true_value, op->false_value);
  }
  void VisitExpr_(const tirx::CallNode *op) override {
    static auto op_if_then_else = Op::Get("tirx.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      VisitIfThenElseExpr(op->args[0], op->args[1], op->args[2]);
    } else {
      Base::VisitExpr_(op);
    }
  }
  void VisitStmt_(const tirx::ForNode *op) override {
    if (op->kind == tirx::ForKind::kParallel ||
        op->kind == tirx::ForKind::kVectorized) {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    } else {
      auto guard_1 =
          MakeGuard(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto guard_2 = MakeGuard(op->extent > 0);
      Base::VisitStmt_(op);
    }
  }
  void VisitStmt_(const tirx::WhileNode *op) override {
    {
      auto guard = MakeGuard(op->condition);
      Base::VisitStmt(op->body);
    }
  }
  ConstrSet GetConstrSet() const {
    return ConstrSet{.constrs_ = constr_stack_};
  }
  std::vector<Constr> constr_stack_;
};
} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_CONSTR_VISITOR_H_
