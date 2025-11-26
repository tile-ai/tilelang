#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <z3++.h>

#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ir/expr.h"
#include "tvm/node/structural_equal.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/expr_functor.h"

namespace tvm::arith {

using namespace tir;
using namespace ffi;

class Z3AnalyzerImpl : ExprFunctor<z3::expr(const PrimExpr&)> {
 public:
  Z3AnalyzerImpl() = default;
  bool force_memorize{false};
  static bool isValidDType(DataType dtype) {
    return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1;
  }
  void EnterWithScope() {
    solver.push();
    scope_stack.push(std::move(scope_updates));
    scope_updates.clear();
  }
  void ExitWithScope() {
    solver.pop();
    auto old_updates = std::move(scope_stack.top());
    scope_stack.pop();
    for (const auto& [e, v] : scope_updates) {
      if (v.has_value()) {
        expr_map[e] = v.value();
      } else {
        expr_map.erase(e);
      }
    }
  }
  void Bind(const Var& var, const PrimExpr& value, bool allow_override = false) {
    if (!isValidDType(var->dtype)) {
      return;
    }
    auto var_expr = getOrCreate(var.as<VarNode>(), true, allow_override);
    auto value_expr = VisitExpr(value);
    solver.add(var_expr == value_expr);
  }
  void Bind(const Var& var, const Range& range, bool allow_override = false) {
    if (!isValidDType(var->dtype)) {
      return;
    }
    auto var_expr = getOrCreate(var.as<VarNode>(), true, allow_override);
    auto min_expr = VisitExpr(range->min);
    auto extent_expr = VisitExpr(range->extent);
    solver.add(var_expr >= min_expr);
    solver.add(var_expr < (min_expr + extent_expr));
  }
  bool CanProve(const PrimExpr& expr) {
    if (!isValidDType(expr->dtype)) {
      return false;
    }
    auto converted = VisitExpr(Not(expr));
    z3::expr_vector vec(ctx);
    vec.push_back(converted);
    auto result = solver.check(vec);
    return result == z3::unsat;
  }

 private:
  using Base = ExprFunctor<z3::expr(const PrimExpr&)>;
  using ExprUpdates = std::vector<std::pair<const PrimExpr, std::optional<z3::expr>>>;
  z3::context ctx;
  z3::solver solver{ctx};
  std::unordered_map<const PrimExpr, z3::expr, StructuralEqual> expr_map;
  ExprUpdates scope_updates;
  std::stack<ExprUpdates> scope_stack;
  z3::expr VisitExpr(const PrimExpr& e) override {
    ICHECK(isValidDType(e->dtype)) << "Z3Analyzer only supports scalar integer expressions.";
    auto it = expr_map.find(e);
    if (it != expr_map.end()) {
      return it->second;
    }
    return Base::VisitExpr(e);
  }
  z3::expr getOrCreate(const PrimExprNode* op, bool memorize = false, bool override = false) {
    auto ref = ffi::GetRef<PrimExpr>(op);
    if (!override && expr_map.count(ref)) {
      return expr_map[ref];
    }
    auto dtype = op->dtype;
    std::stringstream ss;
    ss << ffi::GetRef<PrimExpr>(op);
    std::string name = ss.str();
    auto max_val = Downcast<IntImm>(max_value(dtype))->value;
    auto min_val = Downcast<IntImm>(min_value(dtype))->value;
    auto e = ctx.int_const(name.c_str());
    solver.add(e <= ctx.int_val(max_val));
    solver.add(e >= ctx.int_val(min_val));
    if (memorize || force_memorize) {
      if (expr_map.count(ref)) {
        scope_updates.push_back({ref, expr_map[ref]});
      } else {
        scope_updates.push_back({ref, std::nullopt});
      }
      expr_map[ref] = e;
    }
    return e;
  }
  z3::expr VisitExpr_(const VarNode* op) override { return getOrCreate(op, true); }
  z3::expr VisitExpr_(const BufferLoadNode* op) override { return getOrCreate(op); }
  z3::expr VisitExpr_(const ProducerLoadNode* op) override { return getOrCreate(op); }
  z3::expr VisitExpr_(const LetNode* op) override {
    if (isValidDType(op->var->dtype)) {
      auto var = VisitExpr(op->var);
      auto value = VisitExpr(op->value);
      solver.add(var == value);
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const CallNode* op) override { return getOrCreate(op); }
  z3::expr VisitInt(const PrimExpr& expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      return z3::ite(e, ctx.int_val(1), ctx.int_val(0));
    } else {
      return e;
    }
  }
  z3::expr VisitExpr_(const AddNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return a + b;
  }
  z3::expr VisitExpr_(const SubNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return a - b;
  }
  z3::expr VisitExpr_(const MulNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return a * b;
  }
  z3::expr VisitExpr_(const DivNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return a / b;
  }
  z3::expr VisitExpr_(const ModNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::mod(a, b);
  }
  z3::expr VisitExpr_(const FloorDivNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return a / b;
  }
  z3::expr VisitExpr_(const FloorModNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::mod(a, b);
  }
  z3::expr VisitExpr_(const MinNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a < b, a, b);
  }
  z3::expr VisitExpr_(const MaxNode* op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    return z3::ite(a > b, a, b);
  }
  z3::expr VisitIntCmp(z3::expr (*cmp_op)(const z3::expr&, const z3::expr&), const PrimExprNode* op,
                       const PrimExpr& a, const PrimExpr& b) {
    if (isValidDType(a->dtype) && isValidDType(b->dtype)) {
      auto left = VisitInt(a);
      auto right = VisitInt(b);
      return cmp_op(left, right);
    } else {
      return getOrCreate(op);
    }
  }
  z3::expr VisitExpr_(const EQNode* op) override {
    return VisitIntCmp(z3::operator==, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const NENode* op) override {
    return VisitIntCmp(z3::operator!=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LTNode* op) override {
    return VisitIntCmp(z3::operator<, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LENode* op) override {
    return VisitIntCmp(z3::operator<=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GTNode* op) override {
    return VisitIntCmp(z3::operator>, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GENode* op) override {
    return VisitIntCmp(z3::operator>=, op, op->a, op->b);
  }
  z3::expr VisitBool(const PrimExpr& e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      return expr != ctx.int_val(0);
    }
  }
  z3::expr VisitExpr_(const AndNode* op) override {
    auto a = VisitBool(op->a);
    auto b = VisitBool(op->b);
    return a && b;
  }
  z3::expr VisitExpr_(const OrNode* op) override {
    auto a = VisitBool(op->a);
    auto b = VisitBool(op->b);
    return a || b;
  }
  z3::expr VisitExpr_(const ReduceNode* op) override { return getOrCreate(op); }
  z3::expr VisitExpr_(const CastNode* op) override { return getOrCreate(op); }
  z3::expr VisitExpr_(const NotNode* op) override {
    auto a = VisitBool(op->a);
    return !a;
  }
  z3::expr VisitExpr_(const SelectNode* op) override {
    auto cond = VisitBool(op->condition);
    auto true_value = VisitInt(op->true_value);
    auto false_value = VisitInt(op->false_value);
    return z3::ite(cond, true_value, false_value);
  }
  z3::expr VisitExpr_(const RampNode* op) override {
    throw std::runtime_error("Z3Analyzer does not support RampNode.");
  }
  z3::expr VisitExpr_(const BroadcastNode* op) override {
    throw std::runtime_error("Z3Analyzer does not support BroadcastNode.");
  }
  z3::expr VisitExpr_(const ShuffleNode* op) override {
    throw std::runtime_error("Z3Analyzer does not support ShuffleNode.");
  }
  z3::expr VisitExpr_(const IntImmNode* op) override { return ctx.int_val(op->value); }
  z3::expr VisitExpr_(const FloatImmNode* op) override {
    throw std::runtime_error("Z3Analyzer does not support FloatImmNode.");
  }
  z3::expr VisitExpr_(const StringImmNode* op) override {
    throw std::runtime_error("Z3Analyzer does not support StringImmNode.");
  }
};

class Z3AnalyzerNode : public Object {
 public:
  Z3AnalyzerImpl impl;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("arith.Z3Analyzer", Z3AnalyzerNode, Object);
};

class Z3Analyzer: public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Z3Analyzer, ObjectRef, Z3AnalyzerNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  
}

}  // namespace tvm::arith