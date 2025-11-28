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
  bool use_int;
  int bv_padding_bits;
  z3::context ctx;
  // z3::solver solver{ctx};
  Z3AnalyzerNode(bool use_int=true, int bv_padding_bits=0): use_int(use_int), bv_padding_bits(bv_padding_bits) {
    // turn off model, this leads 30% faster performance
    ctx.set("model", false);
  }
  using Base = ExprFunctor<z3::expr(const PrimExpr &)>;
  using ExprUpdates = std::vector<std::pair<const PrimExpr, std::optional<z3::expr>>>;
  using ExprMap = std::unordered_map<const PrimExpr, z3::expr, StructuralHash, StructuralEqual>;
  bool force_memorize {false};
  void EnterWithScope() {
    constraint_stack.push(constraints.size());
    scope_stack.push(std::move(scope_updates));
    scope_updates.clear();
  }
  void ExitWithScope() {
    size_t old_size = constraint_stack.top();
    constraint_stack.pop();
    constraints.resize(old_size, z3::expr(ctx));
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
    constraints.push_back(var_expr == value_expr);
  }
  void Bind(const Var &var, const Range &range, bool allow_override = false) {
    if (!IsValidDType(var->dtype)) return;
    auto var_expr = GetLeafExpr(var.as<VarNode>(), true, allow_override);
    auto min_expr = VisitExpr(range->min);
    auto extent_expr = VisitExpr(range->extent);
    constraints.push_back(var_expr >= min_expr);
    constraints.push_back(var_expr < (min_expr + extent_expr));
  }
  void AddConstraint(const PrimExpr &constraint) {
    constraints.push_back(VisitBool(constraint));
  }
  void AddAssume(const PrimExpr &constraint) {
    force_memorize = true;
    auto expr = VisitBool(constraint);
    force_memorize = false;
    constraints.push_back(expr);
  }
  bool CanProve(const PrimExpr &expr) {
    if (!IsValidDType(expr->dtype)) {
      return false;
    }
    // EnterWithScope();
    auto result = z3::unknown;
    z3::solver solver{ctx};
    for (const auto &c : constraints) {
      solver.add(c);
    }
    auto converted = !VisitBool(expr);
    auto reuslt = solver.check(1, &converted);
    // try {
    //   result = solver.check(1, &converted);
    // } catch (const z3::exception &e) {
    //   std::string msg = e.msg();
    //   if(msg != "max. steps exceeded") {
    //     LOG(WARNING) << "Z3Analyzer::CanProve failed to add constraint: " << e.msg();
    //   }
    // }
    // ExitWithScope();
    return result == z3::unsat;
  }
  // void SetMaxStep(unsigned max_step) {
  //   solver.set("max_steps", max_step);
  // }
  // void SetTimeoutMs(unsigned timeout_ms) {
  //   solver.set("timeout", timeout_ms);
  // }
  // ffi::String GetSMTLIB2() { return solver.to_smt2(); }
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // auto set_param_impl = [](Z3AnalyzerNode * node, const String & param, const Any & value) {
    //   if(value.type_index() == TypeIndex::kTVMFFIBool) {
    //     return node->solver.set(param.c_str(), value.cast<bool>());
    //   }
    //   if(value.type_index() == TypeIndex::kTVMFFIInt) {
    //     return node->solver.set(param.c_str(), value.cast<unsigned>());
    //   }
    //   if(value.type_index() == TypeIndex::kTVMFFIFloat) {
    //     return node->solver.set(param.c_str(), value.cast<double>());
    //   }
    //   if(auto v = value.as<String>()) {
    //     return node->solver.set(param.c_str(), v->c_str());
    //   }
    //   LOG(FATAL) << "Z3Analyzer::SetParam only supports unsigned, double, bool, and string.";
    // };
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
        // .def("_SetParam", set_param_impl)
        .def("_Bind", bind_impl)
        // .def("set_max_step", &Z3AnalyzerNode::SetMaxStep)
        // .def("set_timeout_ms", &Z3AnalyzerNode::SetTimeoutMs)
        .def("add_constraint", &Z3AnalyzerNode::AddConstraint)
        .def("add_assume", &Z3AnalyzerNode::AddAssume)
        .def("can_prove", &Z3AnalyzerNode::CanProve)
        // .def("get_smtlib2", &Z3AnalyzerNode::GetSMTLIB2)
        .def("enter_with_scope", &Z3AnalyzerNode::EnterWithScope)
        .def("exit_with_scope", &Z3AnalyzerNode::ExitWithScope);
  }
private:
  ExprUpdates scope_updates;
  std::stack<ExprUpdates> scope_stack;
  std::unordered_set<std::string> used_names;
  std::vector<z3::expr> constraints;
  std::stack<size_t> constraint_stack;
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
  unsigned GetBVBits(const DataType & dtype) { return dtype.bits() + bv_padding_bits; }
  z3::expr CreateZ3Val(const DataType & dtype, int64_t val) {
    if(use_int) {
      return ctx.int_val(val);
    } else {
      return ctx.bv_val(val, GetBVBits(dtype));
    }
  }
  z3::expr CreateZ3Const(const DataType & dtype, const std::string &name) {
    if(use_int) {
      return ctx.int_const(name.c_str());
    } else {
      return ctx.bv_const(name.c_str(), GetBVBits(dtype));
    }
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
    z3::expr e = CreateZ3Const(op->dtype, name);
    auto max_val = Downcast<IntImm>(max_value(dtype))->value;
    auto min_val = Downcast<IntImm>(min_value(dtype))->value;
    constraints.push_back(e <= CreateZ3Val(op->dtype, max_val));
    constraints.push_back(e >= CreateZ3Val(op->dtype, min_val));
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
      auto dtype = expr->dtype;
      return z3::ite(e, CreateZ3Val(dtype, 1), CreateZ3Val(dtype, 0));
    } else {
      return e;
    }
  }
  z3::expr VisitBool(const PrimExpr &e) {
    auto expr = VisitExpr(e);
    if (expr.is_bool()) {
      return expr;
    } else {
      auto dtype = e->dtype;
      return expr != CreateZ3Val(dtype, 0);
    }
  }
  z3::expr VisitExpr_(const CastNode * op) override {
    if(!IsValidDType(op->value->dtype)) {
      return GetLeafExpr(op);
    }
    auto e = VisitInt(op->value);
    if(use_int) {
      return e;
    } else {
      auto in_bits = GetBVBits(op->value->dtype);
      auto out_bits = GetBVBits(op->dtype);
      if(in_bits == out_bits) {
        // if bit width is identical, return 
        return e;
      } else if(in_bits > out_bits) {
        // if trunc input expr, using expr.extract
        return e.extract(out_bits - 1, 0);
      } else {
        // extend input
        if(op->value->dtype.is_int()) {
          return z3::sext(e, out_bits);
        } else {
          return z3::zext(e, out_bits);
        }
      }
    }
  }
  using Z3BinOp = z3::expr(*)(const z3::expr &, const z3::expr &);
  z3::expr VisitArith(Z3BinOp signed_op, Z3BinOp unsigned_op, const PrimExprNode *op, const PrimExpr &a, const PrimExpr &b) {
    if (IsValidDType(a->dtype) && IsValidDType(b->dtype)) {
      auto left = VisitInt(a);
      auto right = VisitInt(b);
      if (use_int || (a->dtype.is_int() && b->dtype.is_int())) {
        return signed_op(left, right);
      } else {
        return unsigned_op(left, right);
      }
    } else {
      return GetLeafExpr(op);
    }
  }
  z3::expr VisitExpr_(const MinNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    if(use_int || (op->a->dtype.is_int() && op->b->dtype.is_int())) {
      return z3::ite(a < b, a, b);
    } else {
      return z3::ite(z3::ult(a, b), a, b);
    }
  }
  z3::expr VisitExpr_(const MaxNode *op) override {
    auto a = VisitInt(op->a);
    auto b = VisitInt(op->b);
    if(use_int || (op->a->dtype.is_int() && op->b->dtype.is_int())) {
      return z3::ite(a > b, a, b);
    } else {
      return z3::ite(z3::ugt(a, b), a, b);
    }
  }
  z3::expr VisitExpr_(const LetNode *op) override { 
    if (IsValidDType(op->var->dtype)) {
      auto var = VisitExpr(op->var);
      auto value = VisitExpr(op->value);
      constraints.push_back(var == value);
    }
    return VisitExpr(op->body);
  }
  z3::expr VisitExpr_(const VarNode *op) override { return GetLeafExpr(op, true); }
  z3::expr VisitExpr_(const BufferLoadNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const ProducerLoadNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const AddNode *op) override { return VisitArith(z3::operator +, z3::operator +, op, op->a, op->b); }
  z3::expr VisitExpr_(const SubNode *op) override { return VisitArith(z3::operator -, z3::operator -, op, op->a, op->b); }
  z3::expr VisitExpr_(const MulNode *op) override { return VisitArith(z3::operator *, z3::operator *, op, op->a, op->b); }
  z3::expr VisitExpr_(const DivNode *op) override { return VisitArith(z3::operator /, z3::udiv, op, op->a, op->b); }
  z3::expr VisitExpr_(const ModNode *op) override { return VisitArith(z3::operator %, z3::urem, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorDivNode *op) override { return VisitArith(z3::operator/, z3::udiv, op, op->a, op->b); }
  z3::expr VisitExpr_(const FloorModNode *op) override { return VisitArith(z3::operator %, z3::urem, op, op->a, op->b); }
  z3::expr VisitExpr_(const EQNode *op) override { return VisitArith(z3::operator==, z3::operator==, op, op->a, op->b); }
  z3::expr VisitExpr_(const NENode *op) override { return VisitArith(z3::operator!=, z3::operator!=, op, op->a, op->b); }
  z3::expr VisitExpr_(const LTNode *op) override { return VisitArith(z3::operator<, z3::ult, op, op->a, op->b); }
  z3::expr VisitExpr_(const LENode *op) override { return VisitArith(z3::operator<=, z3::ule, op, op->a, op->b); }
  z3::expr VisitExpr_(const GTNode *op) override { return VisitArith(z3::operator>, z3::ugt, op, op->a, op->b); }
  z3::expr VisitExpr_(const GENode *op) override { return VisitArith(z3::operator>=, z3::uge, op, op->a, op->b); }
  z3::expr VisitExpr_(const AndNode *op) override { return VisitBool(op->a) && VisitBool(op->b); }
  z3::expr VisitExpr_(const OrNode *op) override { return VisitBool(op->a) || VisitBool(op->b); }
  z3::expr VisitExpr_(const ReduceNode *op) override { return GetLeafExpr(op); }
  z3::expr VisitExpr_(const NotNode *op) override { return !VisitBool(op->a); }
  z3::expr VisitExpr_(const SelectNode *op) override { return z3::ite(VisitBool(op->condition), VisitInt(op->true_value), VisitInt(op->false_value)); }
  z3::expr VisitExpr_(const RampNode *op) override { LOG(FATAL) << "Z3Analyzer does not support RampNode."; }
  z3::expr VisitExpr_(const BroadcastNode *op) override { LOG(FATAL) << "Z3Analyzer does not support BroadcastNode."; }
  z3::expr VisitExpr_(const ShuffleNode *op) override { LOG(FATAL) << "Z3Analyzer does not support ShuffleNode."; }
  z3::expr VisitExpr_(const IntImmNode *op) override { return CreateZ3Val(op->dtype, op->value); }
  z3::expr VisitExpr_(const FloatImmNode *op) override { LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions."; }
  z3::expr VisitExpr_(const StringImmNode *op) override { LOG(FATAL) << "Z3Analyzer only supports scalar integer expressions."; }
};

class Z3Analyzer : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Z3Analyzer, ObjectRef,
                                             Z3AnalyzerNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
    .def("tl.arith.Z3Analyzer", [](bool use_int, int bv_padding_bits) {
      return Z3Analyzer(tvm::ffi::make_object<Z3AnalyzerNode>(use_int, bv_padding_bits));
    });
  Z3AnalyzerNode::RegisterReflection();
}

} // namespace tvm::tl