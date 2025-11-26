#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <z3++.h>

#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "dlpack/dlpack.h"
#include "tvm/ffi/cast.h"
#include "tvm/ffi/object.h"
#include "tvm/ffi/reflection/registry.h"
#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "tvm/node/structural_equal.h"
#include "tvm/node/structural_hash.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/expr_functor.h"

namespace tvm::tl {

using namespace tir;
using namespace ffi;

class Z3AnalyzerNode : ExprFunctor<z3::expr(const PrimExpr &)>, public Object {
public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.arith.Z3Analyzer", Z3AnalyzerNode,
                                    Object);
  inline static constexpr bool _type_mutable = true;
  bool build_bv = false;
  Z3AnalyzerNode(bool build_bv = false): build_bv(build_bv) {}
  using Base = ExprFunctor<z3::expr(const PrimExpr &)>;
  using ExprUpdates = std::vector<std::pair<const PrimExpr, std::optional<z3::expr>>>;
  using ExprMap = std::unordered_map<const PrimExpr, z3::expr, StructuralHash, StructuralEqual>;
  z3::context ctx;
  z3::solver solver{ctx};
  bool force_memorize {false};
  void EnterWithScope() {
    solver.push();
    scope_stack.push(std::move(scope_updates));
    scope_updates.clear();
  }
  void ExitWithScope() {
    solver.pop();
    auto old_updates = std::move(scope_stack.top());
    scope_stack.pop();
    for (const auto &[e, v] : scope_updates) {
      if (v.has_value()) {
        leaf_map.emplace(e, v.value());
      } else {
        leaf_map.erase(e);
      }
    }
  }
  static bool IsValidDType(DataType dtype) {
    return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1;
  }
  void Bind(const Var &var, const PrimExpr &value, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    auto var_expr = GetLeafExpr(var.as<VarNode>(), true, allow_override);
    auto value_expr = VisitExpr(value);
    solver.add(var_expr == value_expr);
  }
  void Bind(const Var &var, const Range &range, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    auto var_expr = GetLeafExpr(var.as<VarNode>(), true, allow_override);
    auto min_expr = VisitExpr(range->min);
    auto extent_expr = VisitExpr(range->extent);
    solver.add(var_expr >= min_expr);
    solver.add(var_expr < (min_expr + extent_expr));
  }
  void AddConstraint(const PrimExpr &constraint) {
    solver.add(VisitBool(constraint));
  }
  void AddAssume(const PrimExpr &constraint) {
    force_memorize = true;
    auto expr = VisitBool(constraint);
    force_memorize = false;
    solver.add(expr);
  }
  bool CanProve(const PrimExpr &expr) {
    if (!IsValidDType(expr->dtype)) {
      return false;
    }
    EnterWithScope();
    auto converted = !VisitBool(expr);
    auto result = z3::unknown;
    try {
      result = solver.check(1, &converted);
    } catch (const z3::exception &e) {
      std::string msg = e.msg();
      if(msg != "max. steps exceeded") {
        LOG(WARNING) << "Z3Analyzer::CanProve failed to add constraint: " << e.msg();
      }
    }
    ExitWithScope();
    return result == z3::unsat;
  }
  void SetMaxStep(unsigned max_step) {
    solver.set("max_steps", max_step);
  }
  void SetTimeoutMs(unsigned timeout_ms) {
    solver.set("timeout", timeout_ms);
  }
  ffi::String GetSMTLIB2() { return solver.to_smt2(); }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    auto set_param_impl = [](Z3AnalyzerNode * node, const String & param, const Any & value) {
      std::cerr << "TypeIndex: " << (int) value.type_index() << "\n";
      if(value.type_index() == TypeIndex::kTVMFFIBool) {
        return node->solver.set(param.c_str(), value.cast<bool>());
      }
      if(value.type_index() == TypeIndex::kTVMFFIInt) {
        return node->solver.set(param.c_str(), value.cast<unsigned>());
      }
      if(value.type_index() == TypeIndex::kTVMFFIFloat) {
        return node->solver.set(param.c_str(), value.cast<double>());
      }
      if(auto v = value.as<String>()) {
        return node->solver.set(param.c_str(), v->c_str());
      }
      LOG(FATAL) << "Z3Analyzer::SetParam only supports unsigned, double, bool, and string.";
    };
    auto bind_impl = [](Z3AnalyzerNode * self, const Var & var, const ObjectRef & obj, bool allow_override) {
      if(obj->IsInstance<PrimExprNode>()) {
        return self->Bind(var, Downcast<PrimExpr>(obj), allow_override);
      }
      if(obj->IsInstance<RangeNode>()) {
        return self->Bind(var, Downcast<Range>(obj), allow_override);
      }
      LOG(FATAL) << "Z3Analyzer::Bind only supports PrimExpr and Range.";
    };
    refl::ObjectDef<Z3AnalyzerNode>()
        .def("_SetParam", set_param_impl)
        .def("_Bind", bind_impl)
        .def("set_max_step", &Z3AnalyzerNode::SetMaxStep)
        .def("set_timeout_ms", &Z3AnalyzerNode::SetTimeoutMs)
        .def("add_constraint", &Z3AnalyzerNode::AddConstraint)
        .def("add_assume", &Z3AnalyzerNode::AddAssume)
        .def("can_prove", &Z3AnalyzerNode::CanProve)
        .def("get_smtlib2", &Z3AnalyzerNode::GetSMTLIB2)
        .def("enter_with_scope", &Z3AnalyzerNode::EnterWithScope)
        .def("exit_with_scope", &Z3AnalyzerNode::ExitWithScope);
  }
private:
  ExprUpdates scope_updates;
  std::stack<ExprUpdates> scope_stack;
  std::unordered_set<std::string> used_names;
  ExprMap leaf_map;
  std::string GetNewName(const std::string & name) {
    if(used_names.count(name) == 0) {
      used_names.insert(name);
      return name;
    }
    int idx = 1;
    std::string check_name = name + "$" + std::to_string(idx);
    while(used_names.count(check_name)) {
      idx ++;
      check_name = name + "$" + std::to_string(idx);
    }
    used_names.insert(check_name);
    return check_name;
  }
  z3::expr GetLeafExpr(const PrimExprNode *op, bool memorize = false, bool override = false) {
    auto ref = ffi::GetRef<PrimExpr>(op);
    if (!override && leaf_map.count(ref)) {
      return leaf_map.at(ref);
    }
    auto dtype = op->dtype;
    std::stringstream ss;
    ss << ref;
    std::string name = GetNewName(ss.str());
    z3::expr e(ctx);
    if(build_bv) {
      e = ctx.bv_const(name.c_str(), dtype.bits());
    } else {
      auto max_val = Downcast<IntImm>(max_value(dtype))->value;
      auto min_val = Downcast<IntImm>(min_value(dtype))->value;
      e = ctx.int_const(name.c_str());
      solver.add(e <= ctx.int_val(max_val));
      solver.add(e >= ctx.int_val(min_val));
    }
    if (memorize || force_memorize) {
      if (leaf_map.count(ref)) {
        scope_updates.emplace_back(ref, leaf_map.at(ref));
      } else {
        scope_updates.emplace_back(ref, std::nullopt);
      }
      leaf_map.emplace(ref, e);
    }
    return e;
  }
  z3::expr VisitInt(const PrimExpr &expr) {
    auto e = VisitExpr(expr);
    if (e.is_bool()) {
      auto bits = expr->dtype.bits();
      if(build_bv) {
        return z3::ite(e, ctx.bv_val(1, bits), ctx.bv_val(0, bits));
      } else {
        return z3::ite(e, ctx.int_val(1), ctx.int_val(0));
      }
    } else {
      return e;
    }
  }
  z3::expr VisitBool(const PrimExpr &e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      auto bits = e->dtype.bits();
      if(build_bv) {
        return expr != ctx.bv_val(0, bits);
      } else {
        return expr != ctx.int_val(0);
      }
    }
  }
  z3::expr VisitIntBiOp(
    z3::expr (*signed_op)(const z3::expr &, const z3::expr &),
    z3::expr (*unsigned_op)(const z3::expr &, const z3::expr &),
    const PrimExprNode *op, const PrimExpr &a, const PrimExpr &b
  ) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      auto left = VisitInt(a);
      auto right = VisitInt(b);
      try {
        if(build_bv) {
          if (a->dtype.is_int() && b->dtype.is_int()) {
            return signed_op(left, right);
          } else {
            return unsigned_op(left, right);
          }
        } else {
          return signed_op(left, right);
        }
      } catch(z3::exception &e) {
        LOG(FATAL) << "Z3Analyzer::VisitIntBiOp failed to add signed op: " << GetRef<PrimExpr>(op) << e.msg();
      }
    } else {
      return GetLeafExpr(op);
    }
  }
  z3::expr VisitBoolCmpOp(
    z3::expr (*signed_op)(const z3::expr &, const z3::expr &),
    z3::expr (*unsigned_op)(const z3::expr &, const z3::expr &),
    const PrimExprNode *op, const PrimExpr &a, const PrimExpr &b
  ) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      auto left = VisitInt(a);
      auto right = VisitInt(b);
      try {
        if(build_bv) {
          if (a->dtype.is_int() && b->dtype.is_int()) {
            return signed_op(left, right);
          } else {
            return unsigned_op(left, right);
          }
        } else {
          return signed_op(left, right);
        }
      } catch(z3::exception &e) {
        LOG(FATAL) << "Z3Analyzer::VisitIntBiOp failed to add signed op: " << GetRef<PrimExpr>(op) << e.msg();
      }
    } else {
      return GetLeafExpr(op);
    }
  }
  z3::expr VisitExpr_(const VarNode *op) override { return GetLeafExpr(op, true); }
  z3::expr VisitExpr_(const BufferLoadNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const ProducerLoadNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const LetNode *op) override {
    if (IsValidDType(op->var->dtype)) {
      auto var = VisitExpr(op->var);
      auto value = VisitExpr(op->value);
      solver.add(var == value);
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const AddNode *op) override {
    return VisitIntBiOp(z3::operator +, z3::operator +, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const SubNode *op) override {
    return VisitIntBiOp(z3::operator +, z3::operator +, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const MulNode *op) override {
    return VisitIntBiOp(z3::operator *, z3::operator *, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const DivNode *op) override {
    return VisitIntBiOp(z3::operator /, z3::udiv, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const ModNode *op) override {
    return VisitIntBiOp(z3::operator %, z3::urem, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const FloorDivNode *op) override {
    return VisitIntBiOp(z3::operator/, z3::udiv, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const FloorModNode *op) override {
    return VisitIntBiOp(z3::operator %, z3::urem, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const EQNode *op) override {
    return VisitBoolCmpOp(z3::operator==, z3::operator==, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const NENode *op) override {
    return VisitBoolCmpOp(z3::operator!=, z3::operator!=, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LTNode *op) override {
    return VisitBoolCmpOp(z3::operator<, z3::ult, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const LENode *op) override {
    return VisitBoolCmpOp(z3::operator<=, z3::ule, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GTNode *op) override {
    return VisitBoolCmpOp(z3::operator>, z3::ugt, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const GENode *op) override {
    return VisitBoolCmpOp(z3::operator>=, z3::uge, op, op->a, op->b);
  }
  z3::expr VisitExpr_(const MinNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    if(build_bv) {
      if(op->a->dtype.is_int() && op->b->dtype.is_int()) {
        return z3::ite(z3::slt(a, b), a, b);
      } else {
        return z3::ite(z3::ult(a, b), a, b);
      }
    } else {
      return z3::ite(a < b, a, b);
    }
  }
  z3::expr VisitExpr_(const MaxNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    if(build_bv) {
      if(op->a->dtype.is_int() && op->b->dtype.is_int()) {
        return z3::ite(z3::sgt(a, b), a, b);
      } else {
        return z3::ite(z3::ugt(a, b), a, b);
      }
    } else {
      return z3::ite(a > b, a, b);
    }
  }
  z3::expr VisitExpr_(const AndNode *op) override {
    auto a = VisitBool(op->a);
    auto b = VisitBool(op->b);
    return a && b;
  }
  z3::expr VisitExpr_(const OrNode *op) override {
    auto a = VisitBool(op->a);
    auto b = VisitBool(op->b);
    return a || b;
  }
  z3::expr VisitExpr_(const ReduceNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const CastNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode *op) override {
    auto cond = VisitBool(op->condition);
    auto true_value = VisitInt(op->true_value);
    auto false_value = VisitInt(op->false_value);
    return z3::ite(cond, true_value, false_value);
  }
  z3::expr VisitExpr_(const RampNode *op) override {
    LOG(FATAL) << "Z3Analyzer does not support RampNode.";
  }
  z3::expr VisitExpr_(const BroadcastNode *op) override {
    LOG(FATAL) << "Z3Analyzer does not support BroadcastNode.";
  }
  z3::expr VisitExpr_(const ShuffleNode *op) override {
    LOG(FATAL) << "Z3Analyzer does not support ShuffleNode.";
  }
  z3::expr VisitExpr_(const IntImmNode *op) override {
    if(build_bv) {
      return ctx.bv_val(op->value, op->dtype.bits());
    } else {
      return ctx.int_val(op->value);
    }
  }
  z3::expr VisitExpr_(const FloatImmNode *op) override {
    LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions.";
  }
  z3::expr VisitExpr_(const StringImmNode *op) override {
    LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions.";
  }
};


// class Z3AnalyzerNode : public Object, ExprFunctor<z3::expr(const PrimExpr &)> {
// public:
//   TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.arith.Z3Analyzer", Z3AnalyzerNode,
//                                     Object);
//   inline static constexpr bool _type_mutable = true;
//   Z3AnalyzerNode() = default;
//   bool force_memorize{false};
//   static bool isValidDType(DataType dtype) {
//     return (dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1;
//   }
//   void EnterWithScope() {
//     solver.push();
//     scope_stack.push(std::move(scope_updates));
//     scope_updates.clear();
//   }
//   void ExitWithScope() {
//     solver.pop();
//     auto old_updates = std::move(scope_stack.top());
//     scope_stack.pop();
//     for (const auto &[e, v] : scope_updates) {
//       if (v.has_value()) {
//         expr_map.emplace(e, v.value());
//       } else {
//         expr_map.erase(e);
//       }
//     }
//   }
//   void Bind(const Var &var, const PrimExpr &value,
//             bool allow_override = false) {
//     if (!isValidDType(var->dtype)) {
//       return;
//     }
//     auto var_expr = getOrCreate(var.as<VarNode>(), true, allow_override);
//     auto value_expr = VisitExpr(value);
//     solver.add(var_expr == value_expr);
//   }
//   void Bind(const Var &var, const Range &range, bool allow_override = false) {
//     if (!isValidDType(var->dtype)) {
//       return;
//     }
//     auto var_expr = getOrCreate(var.as<VarNode>(), true, allow_override);
//     auto min_expr = VisitExpr(range->min);
//     auto extent_expr = VisitExpr(range->extent);
//     solver.add(var_expr >= min_expr);
//     solver.add(var_expr < (min_expr + extent_expr));
//   }
//   void AddConstraint(const PrimExpr &constraint) {
//     solver.add(VisitBool(constraint));
//   }
//   void AddAssume(const PrimExpr &constraint) {
//     force_memorize = true;
//     auto expr = VisitBool(constraint);
//     force_memorize = false;
//     solver.add(expr);
//   }
//   bool CanProve(const PrimExpr &expr) {
//     if (!isValidDType(expr->dtype)) {
//       return false;
//     }
//     EnterWithScope();
//     auto converted = !VisitBool(expr);
//     auto result = z3::unknown;
//     try {
//       result = solver.check(1, &converted);
//     } catch (const z3::exception &e) {
//       std::string msg = e.msg();
//       if(msg != "max. steps exceeded") {
//         LOG(WARNING) << "Z3Analyzer::CanProve failed to add constraint: " << e.msg();
//       }
//     }
//     ExitWithScope();
//     return result == z3::unsat;
//   }
//   void SetMaxStep(unsigned max_step) {
//     solver.set("max_steps", max_step);
//   }
//   void SetTimeoutMs(unsigned timeout_ms) {
//     solver.set("timeout", timeout_ms);
//   }
//   ffi::String GetSMTLIB2() { return solver.to_smt2(); }
//   static void RegisterReflection() {
//     namespace refl = tvm::ffi::reflection;
//     auto set_param_impl = [](Z3AnalyzerNode * node, const String & param, const Any & value) {
//       std::cerr << "TypeIndex: " << (int) value.type_index() << "\n";
//       if(value.type_index() == TypeIndex::kTVMFFIBool) {
//         return node->solver.set(param.c_str(), value.cast<bool>());
//       }
//       if(value.type_index() == TypeIndex::kTVMFFIInt) {
//         return node->solver.set(param.c_str(), value.cast<unsigned>());
//       }
//       if(value.type_index() == TypeIndex::kTVMFFIFloat) {
//         return node->solver.set(param.c_str(), value.cast<double>());
//       }
//       if(auto v = value.as<String>()) {
//         return node->solver.set(param.c_str(), v->c_str());
//       }
//       LOG(FATAL) << "Z3Analyzer::SetParam only supports unsigned, double, bool, and string.";
//     };
//     auto bind_impl = [](Z3AnalyzerNode * self, const Var & var, const ObjectRef & obj, bool allow_override) {
//       if(obj->IsInstance<PrimExprNode>()) {
//         return self->Bind(var, Downcast<PrimExpr>(obj), allow_override);
//       }
//       if(obj->IsInstance<RangeNode>()) {
//         return self->Bind(var, Downcast<Range>(obj), allow_override);
//       }
//       LOG(FATAL) << "Z3Analyzer::Bind only supports PrimExpr and Range.";
//     };
//     refl::ObjectDef<Z3AnalyzerNode>()
//         .def("_SetParam", set_param_impl)
//         .def("_Bind", bind_impl)
//         .def("set_max_step", &Z3AnalyzerNode::SetMaxStep)
//         .def("set_timeout_ms", &Z3AnalyzerNode::SetTimeoutMs)
//         .def("add_constraint", &Z3AnalyzerNode::AddConstraint)
//         .def("add_assume", &Z3AnalyzerNode::AddAssume)
//         .def("can_prove", &Z3AnalyzerNode::CanProve)
//         .def("get_smtlib2", &Z3AnalyzerNode::GetSMTLIB2)
//         .def("enter_with_scope", &Z3AnalyzerNode::EnterWithScope)
//         .def("exit_with_scope", &Z3AnalyzerNode::ExitWithScope);
//   }
//   void SetParam(String param, bool value) {
//     solver.set(param.c_str(), value);
//   }

// private:
//   using Base = ExprFunctor<z3::expr(const PrimExpr &)>;
//   using ExprUpdates =
//       std::vector<std::pair<const PrimExpr, std::optional<z3::expr>>>;
//   using ExprMap = std::unordered_map<const PrimExpr, z3::expr, StructuralHash, StructuralEqual>;
//   z3::context ctx;
//   z3::solver solver{ctx};
//   std::unordered_set<std::string> used_names;
//   ExprMap expr_map;
//   ExprUpdates scope_updates;
//   std::stack<ExprUpdates> scope_stack;
//   z3::expr VisitExpr(const PrimExpr &e) override {
//     ICHECK(isValidDType(e->dtype))
//         << "Z3Analyzer only supports scalar integer expressions.";
//     auto it = expr_map.find(e);
//     if (it != expr_map.end()) {
//       return it->second;
//     }
//     return Base::VisitExpr(e);
//   }
//   z3::expr VisitInt(const PrimExpr &expr) {
//     auto e = VisitExpr(expr);
//     if (e.is_bool()) {
//       return z3::ite(e, ctx.int_val(1), ctx.int_val(0));
//     } else {
//       return e;
//     }
//   }
//   z3::expr VisitBool(const PrimExpr &e) {
//     auto expr = VisitExpr(e);
//     if (expr.is_bool()) {
//       return expr;
//     } else {
//       return expr != ctx.int_val(0);
//     }
//   }
//   z3::expr VisitIntBiOp(z3::expr (*cmp_op)(const z3::expr &, const z3::expr &),
//                         const PrimExprNode *op, const PrimExpr &a,
//                         const PrimExpr &b) {
//     if (isValidDType(a->dtype) && isValidDType(b->dtype)) {
//       auto left = VisitInt(a);
//       auto right = VisitInt(b);
//       return cmp_op(left, right);
//     } else {
//       return getOrCreate(op);
//     }
//   }
//   z3::expr VisitBoolBiOp(z3::expr (*cmp_op)(const z3::expr &, const z3::expr &),
//                          const PrimExprNode *op, const PrimExpr &a,
//                          const PrimExpr &b) {
//     auto left = VisitBool(a);
//     auto right = VisitBool(b);
//     return cmp_op(left, right);
//   }
//   z3::expr getOrCreate(const PrimExprNode *op, bool memorize = false,
//                        bool override = false) {
//     auto ref = ffi::GetRef<PrimExpr>(op);
//     if (!override && expr_map.count(ref)) {
//       return expr_map.at(ref);
//     }
//     auto dtype = op->dtype;
//     std::stringstream ss;
//     ss << ffi::GetRef<PrimExpr>(op);
//     std::string name = ss.str();
//     if(used_names.count(name)) {
//       int idx = 1;
//       std::string check_name = name + "$" + std::to_string(idx);
//       while(used_names.count(check_name)) {
//         idx ++;
//         check_name = name + "$" + std::to_string(idx);
//       }
//       name = check_name;
//     }
//     auto max_val = Downcast<IntImm>(max_value(dtype))->value;
//     auto min_val = Downcast<IntImm>(min_value(dtype))->value;
//     auto e = ctx.int_const(name.c_str());
//     solver.add(e <= ctx.int_val(max_val));
//     solver.add(e >= ctx.int_val(min_val));
//     if (memorize || force_memorize) {
//       if (expr_map.count(ref)) {
//         scope_updates.emplace_back(ref, expr_map.at(ref));
//       } else {
//         scope_updates.emplace_back(ref, std::nullopt);
//       }
//       expr_map.emplace(ref, e);
//     }
//     return e;
//   }
//   z3::expr VisitExpr_(const VarNode *op) override {
//     return getOrCreate(op, true);
//   }
//   z3::expr VisitExpr_(const BufferLoadNode *op) override {
//     return getOrCreate(op);
//   }
//   z3::expr VisitExpr_(const ProducerLoadNode *op) override {
//     return getOrCreate(op);
//   }
//   z3::expr VisitExpr_(const LetNode *op) override {
//     if (isValidDType(op->var->dtype)) {
//       auto var = VisitExpr(op->var);
//       auto value = VisitExpr(op->value);
//       solver.add(var == value);
//     }
//     return VisitExpr(op->body);
//   }
//   z3::expr VisitExpr_(const CallNode *op) override { return getOrCreate(op); }
//   z3::expr VisitExpr_(const AddNode *op) override {
//     return VisitIntBiOp(z3::operator+, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const SubNode *op) override {
//     return VisitIntBiOp(z3::operator-, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const MulNode *op) override {
//     return VisitIntBiOp(z3::operator*, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const DivNode *op) override {
//     return VisitIntBiOp(z3::operator/, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const ModNode *op) override {
//     return VisitIntBiOp(z3::mod, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const FloorDivNode *op) override {
//     return VisitIntBiOp(z3::operator/, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const FloorModNode *op) override {
//     return VisitIntBiOp(z3::mod, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const EQNode *op) override {
//     return VisitIntBiOp(z3::operator==, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const NENode *op) override {
//     return VisitIntBiOp(z3::operator!=, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const LTNode *op) override {
//     return VisitIntBiOp(z3::operator<, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const LENode *op) override {
//     return VisitIntBiOp(z3::operator<=, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const GTNode *op) override {
//     return VisitIntBiOp(z3::operator>, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const GENode *op) override {
//     return VisitIntBiOp(z3::operator>=, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const MinNode *op) override {
//     auto a = VisitInt(op->a);
//     auto b = VisitInt(op->b);
//     return z3::ite(a < b, a, b);
//   }
//   z3::expr VisitExpr_(const MaxNode *op) override {
//     auto a = VisitInt(op->a);
//     auto b = VisitInt(op->b);
//     return z3::ite(a > b, a, b);
//   }
//   z3::expr VisitExpr_(const AndNode *op) override {
//     return VisitBoolBiOp(z3::operator&&, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const OrNode *op) override {
//     return VisitBoolBiOp(z3::operator||, op, op->a, op->b);
//   }
//   z3::expr VisitExpr_(const ReduceNode *op) override { return getOrCreate(op); }
//   z3::expr VisitExpr_(const CastNode *op) override { return getOrCreate(op); }
//   z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
//   z3::expr VisitExpr_(const SelectNode *op) override {
//     auto cond = VisitBool(op->condition);
//     auto true_value = VisitInt(op->true_value);
//     auto false_value = VisitInt(op->false_value);
//     return z3::ite(cond, true_value, false_value);
//   }
//   z3::expr VisitExpr_(const RampNode *op) override {
//     LOG(FATAL) << "Z3Analyzer does not support RampNode.";
//   }
//   z3::expr VisitExpr_(const BroadcastNode *op) override {
//     LOG(FATAL) << "Z3Analyzer does not support BroadcastNode.";
//   }
//   z3::expr VisitExpr_(const ShuffleNode *op) override {
//     LOG(FATAL) << "Z3Analyzer does not support ShuffleNode.";
//   }
//   z3::expr VisitExpr_(const IntImmNode *op) override {
//     LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions.";
//   }
//   z3::expr VisitExpr_(const FloatImmNode *op) override {
//     LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions.";
//   }
//   z3::expr VisitExpr_(const StringImmNode *op) override {
//     LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions.";
//   }
// };

class Z3Analyzer : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Z3Analyzer, ObjectRef,
                                             Z3AnalyzerNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("tl.arith.Z3Analyzer", [](bool use_bool_vector) {
      return Z3Analyzer(tvm::ffi::make_object<Z3AnalyzerNode>(use_bool_vector));
    });
  Z3AnalyzerNode::RegisterReflection();
}

} // namespace tvm::tl