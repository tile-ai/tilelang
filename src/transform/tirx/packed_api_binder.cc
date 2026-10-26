#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/dtype.h"
#include "op/prim.h"
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file packed_api_binder.cc
 * \brief Helper utility to match and bind packed function arguments.
 */
#include "packed_api_binder.h"

#include "transform/common/packed_api.h"
#include <tvm/ffi/cast.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include <algorithm>

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using namespace tvm::prim;
using tvm::prim::cast;

using ffi::reflection::AccessPath;
using ffi::reflection::AccessStep;

namespace {
PrimType PackedScalarType(const PrimType &type) {
  TVM_FFI_ICHECK(!type.IsVoid()) << "Cannot pass void through packed API";
  TVM_FFI_ICHECK_EQ(type.lanes(), 1)
      << "Cannot pass vectors through packed API";
  if (type.MatchesCode(kDLBool, kDLUInt, kDLInt))
    return PrimType::Int(64);
  TVM_FFI_ICHECK(type.MatchesCode(kDLFloat, kDLBfloat));
  return PrimType::Float(64);
}
} // namespace

PackedAPIArguments DecodePackedAPIArguments(
    const ffi::String &name, const ffi::Array<Var> &parameters,
    const Var &packed_args, const Var &num_args, const PrimExpr &device_type,
    const PrimExpr &device_id, const ffi::Map<Var, PrimType> &storage_dtypes,
    const ffi::Array<Var> &nullable_parameters,
    const ffi::Array<Var> &required_metadata) {
  PackedArgumentDecoder builder(name, parameters, packed_args, num_args,
                                device_type, device_id, storage_dtypes,
                                nullable_parameters, required_metadata);
  builder.DecodeAllParams();
  auto decoded = builder.Finalize();
  PackedAPIArguments result;
  for (const auto &[variable, definition] : decoded.var_defs) {
    result.bindings.Set(ffi::GetRef<Var>(variable), definition.value);
  }
  result.definitions = ffi::Array<Stmt>(decoded.init_nest);
  result.assertions = ffi::Array<Stmt>(decoded.asserts);
  result.tensor_declarations = ffi::Array<Stmt>(decoded.decl_tensors);
  return result;
}

// ============================================================
// Constructor
// ============================================================

PackedArgumentDecoder::PackedArgumentDecoder(
    const ffi::String &func_name, const ffi::Array<Var> &params,
    const Var &v_packed_args, const Var &v_num_packed_args,
    const PrimExpr &device_type, const PrimExpr &device_id,
    const ffi::Map<Var, PrimType> &storage_dtypes,
    const ffi::Array<Var> &nullable_parameters,
    const ffi::Array<Var> &required_metadata)
    : func_name_(func_name), params_(params), storage_dtypes_(storage_dtypes),
      v_packed_args_(v_packed_args), device_type_(device_type),
      device_id_(device_id) {
  for (const Var &parameter : nullable_parameters) {
    TVM_FFI_CHECK(parameter.as<TensorVar>() &&
                      std::any_of(params.begin(), params.end(),
                                  [&](const Var &candidate) {
                                    return candidate.same_as(parameter);
                                  }),
                  ValueError)
        << "Nullable parameters must be tensor arguments of the packed "
           "function";
    nullable_parameters_.insert(parameter.get());
  }
  for (const Var &variable : required_metadata)
    required_metadata_.insert(variable.get());
  for (const auto &[parameter, storage] : storage_dtypes_) {
    auto tensor = parameter.as<TensorVar>();
    TVM_FFI_CHECK(tensor.has_value(), TypeError)
        << "Packed storage requires a tensor parameter";
    int logical_bits =
        tensor.value().dtype().bits() * tensor.value().dtype().lanes();
    int storage_bits = storage.bits() * storage.lanes();
    TVM_FFI_CHECK(logical_bits > 0 && storage_bits >= logical_bits &&
                      storage_bits % logical_bits == 0,
                  ValueError)
        << "Physical tensor storage must contain an integral number of logical "
           "elements";
    TVM_FFI_CHECK(storage_bits == logical_bits ||
                      !tensor.value().shape().empty(),
                  ValueError)
        << "Packed tensor storage requires an innermost dimension";
  }
  // Build function signature string
  std::ostringstream os;
  os << func_name << "(";
  for (size_t i = 0; i < params.size(); ++i) {
    if (i > 0)
      os << ", ";
    Var param = params[i];
    if (auto buf = param.as<TensorVar>()) {
      std::string buf_name = buf.value()->name;
      os << buf_name << ": Tensor([";
      for (size_t j = 0; j < buf.value().shape().size(); ++j) {
        if (j > 0)
          os << ", ";
        std::ostringstream shape_os;
        if (auto var = buf.value().shape()[j].as<PrimVar>()) {
          shape_os << ((*var)->name.empty() ? "v" : (*var)->name.c_str());
        } else {
          shape_os << buf.value().shape()[j];
        }
        os << shape_os.str();
      }
      os << "], " << buf.value().dtype()->dtype << ")";
      param_names_[static_cast<int>(i)] = buf_name;
    } else {
      os << param->name << ": ";
      if (const auto *prim_type = param->ty.as<PrimTypeNode>()) {
        os << prim_type->dtype;
      } else {
        os << param->ty;
      }
      param_names_[static_cast<int>(i)] = param->name;
    }
  }
  os << ")";
  func_signature_ = os.str();
  sig_imm_ = StringImm(func_signature_);

  // Emit argument count check (early check — must execute before any loads)
  int num_args = static_cast<int>(params.size());
  EmitAssert(v_num_packed_args.as_or_throw<PrimExpr>() == num_args,
             "TypeError", //
             "Expected ", std::to_string(num_args), " arguments",
             when_calling_imm_, sig_imm_, "`");

  // Emit null-pointer check for packed args (early check)
  if (num_args > 0) {
    EmitAssert(!Call(PrimType::Bool(), tirx::isnullptr_op(),
                     ffi::Array<Expr>{v_packed_args})
                    .as_or_throw<PrimExpr>(),
               "TypeError", //
               "args pointer is NULL", when_calling_imm_, sig_imm_, "`");
  }
}

// ============================================================
// EmitTypeIndexCheck
// ============================================================

void PackedArgumentDecoder::EmitTypeIndexCheck(
    int param_index, const PrimExpr &cond, const std::string &expected_type) {
  EmitAssert(cond, "TypeError", //
             "Mismatched type on argument #", std::to_string(param_index),
             when_calling_imm_, sig_imm_, "`,\n  expected ", expected_type);
}

// ============================================================
// RenderAccessPath
// ============================================================

ffi::String PackedArgumentDecoder::RenderAccessPath(
    const ffi::reflection::AccessPath &path) const {
  ffi::Array<AccessStep> steps = path->ToSteps();
  std::ostringstream os;
  bool first_printed = false;
  bool consumed_root_array_item = false;
  for (size_t i = 0; i < steps.size(); ++i) {
    const AccessStep &step = steps[i];
    if (step->kind == ffi::reflection::AccessKind::kAttr) {
      std::string field_name = step->key.cast<ffi::String>();
      if (!first_printed) {
        os << field_name;
        first_printed = true;
      } else {
        os << "." << field_name;
      }
    } else if (step->kind == ffi::reflection::AccessKind::kArrayItem) {
      int64_t index = step->key.cast<int64_t>();
      if (!first_printed && !consumed_root_array_item) {
        consumed_root_array_item = true;
        if (i + 1 < steps.size() &&
            steps[i + 1]->kind == ffi::reflection::AccessKind::kAttr) {
          continue;
        }
        auto it = param_names_.find(static_cast<int>(index));
        if (it != param_names_.end()) {
          os << it->second;
          first_printed = true;
        }
      } else {
        os << "[" << index << "]";
      }
    } else if (step->kind == ffi::reflection::AccessKind::kMapItem) {
      os << "[" << step->key << "]";
    }
  }
  return os.str();
}

// ============================================================
// GetParamIndex
// ============================================================

int PackedArgumentDecoder::GetParamIndex(
    const ffi::reflection::AccessPath &path) const {
  ffi::Array<AccessStep> steps = path->ToSteps();
  if (steps.size() >= 1 &&
      steps[0]->kind == ffi::reflection::AccessKind::kArrayItem) {
    return static_cast<int>(steps[0]->key.cast<int64_t>());
  }
  return -1;
}

// ============================================================
// BindScalar (scalar bind with ffi::reflection::AccessPath)
// ============================================================

bool PackedArgumentDecoder::BindScalar(const PrimExpr &arg,
                                       const PrimExpr &value,
                                       const ffi::reflection::AccessPath &path,
                                       bool with_lets) {
  TVM_FFI_ICHECK(arg.ty() == value.ty());
  if (defer_scalar_bindings_) {
    pending_scalar_bindings_.push_back({arg, value, path, current_guard_});
    return false;
  }
  if (arg.as<VarNode>()) {
    Var v_arg = arg.as_or_throw<Var>();
    auto it = var_defs_.find(v_arg.get());
    if (it == var_defs_.end()) {
      // First bind: define the variable
      if (with_lets) {
        var_defs_.emplace(v_arg.get(), VarDefInfo{arg, path});
        init_nest_.emplace_back(Bind(v_arg, value));
      } else {
        var_defs_.emplace(v_arg.get(), VarDefInfo{value, path});
      }
      return true;
    } else {
      // Duplicate bind: create rich assertion with both paths
      PrimExpr prev_value = it->second.value.as_or_throw<PrimExpr>();
      PrimExpr scond =
          GuardCondition(tl::arith::Simplify(analyzer_, prev_value == value));
      if (IsZero(scond)) {
        TVM_FFI_THROW(InternalError)
            << "Bind have an unmet assertion: " << prev_value << " == " << value
            << " at " << RenderAccessPath(path);
      }
      if (!IsOne(scond)) {
        ffi::String current_path_str = RenderAccessPath(path);
        ffi::String first_path_str =
            RenderAccessPath(it->second.first_def_path);
        int param_index = GetParamIndex(path);
        ffi::Array<StringImm> parts;
        parts.push_back(StringImm("Mismatched "));
        parts.push_back(StringImm(current_path_str));
        if (param_index >= 0) {
          parts.push_back(StringImm(" on argument #"));
          parts.push_back(StringImm(std::to_string(param_index)));
        }
        parts.push_back(when_calling_imm_);
        parts.push_back(sig_imm_);
        if (!first_path_str.empty() && first_path_str != current_path_str) {
          parts.push_back(StringImm("`,\n  expected to match "));
          parts.push_back(StringImm(first_path_str));
        } else {
          parts.push_back(StringImm("`,\n  expected matching value"));
        }
        asserts_.emplace_back(
            AssertStmt(scond, StringImm("ValueError"), parts));
      }
    }
  } else {
    // Non-Var expression (e.g. batch_size + 1): defer assertion to Finalize()
    // so display-var substitution can render human-readable names.
    PrimExpr scond =
        GuardCondition(tl::arith::Simplify(analyzer_, arg == value));
    if (IsZero(scond)) {
      TVM_FFI_THROW(InternalError)
          << "Bind have an unmet assertion: " << arg << " == " << value
          << " at " << RenderAccessPath(path);
    }
    if (!IsOne(scond)) {
      pending_const_asserts_.push_back({scond, path, arg});
    }
  }
  return false;
}

PrimExpr
PackedArgumentDecoder::GuardCondition(const PrimExpr &condition) const {
  if (IsOne(current_guard_))
    return condition;
  return Call(PrimType::Bool(), prim::if_then_else_op(),
              {current_guard_, condition, IntImm::Bool(true)})
      .as_or_throw<PrimExpr>();
}

Expr PackedArgumentDecoder::GuardValue(const Expr &value) const {
  if (IsOne(current_guard_))
    return value;
  Expr zero =
      value->ty.as<PrimTypeNode>()
          ? Expr(tl::MakeConst(value->ty.as_or_throw<PrimType>(), 0))
          : Expr(Call(value->ty, tirx::reinterpret_op(), {IntImm::Int64(0)}));
  return Call(value->ty, prim::if_then_else_op(),
              {current_guard_, value, zero});
}

void PackedArgumentDecoder::BindNullableMetadata() {
  defer_scalar_bindings_ = true;
  for (auto &shape : pending_int_exprs_) {
    current_guard_ = shape.guard;
    TryBindLinearIntExpr(shape.arg, shape.value, shape.path);
  }
  current_guard_ = IntImm::Bool(true);
  defer_scalar_bindings_ = false;

  auto bindings = std::move(pending_scalar_bindings_);
  std::vector<Var> variables;
  std::unordered_map<const VarNode *, std::vector<const PendingScalarBinding *>>
      candidates;
  for (const auto &binding : bindings) {
    if (auto variable = binding.argument.as<Var>()) {
      auto [iterator, inserted] =
          candidates.try_emplace(variable.value().get());
      if (inserted)
        variables.push_back(variable.value());
      iterator->second.push_back(&binding);
    }
  }
  for (const Var &variable : variables) {
    if (var_defs_.count(variable.get()))
      continue;
    const auto &sources = candidates.at(variable.get());
    PrimExpr value = tl::MakeConst(variable->ty.as_or_throw<PrimType>(), 0);
    PrimExpr available = IntImm::Bool(false);
    std::unordered_set<int> parameter_indices;
    for (auto iterator = sources.rbegin(); iterator != sources.rend();
         ++iterator) {
      const auto &source = **iterator;
      value = tvm::if_then_else(source.guard, source.value, value);
      available = available || source.guard;
      parameter_indices.insert(GetParamIndex(source.path));
    }
    if (required_metadata_.count(variable.get()) &&
        parameter_indices.size() > 1) {
      EmitAssert(available, "RuntimeError",
                 "at least one non-null buffer is required to bind ",
                 variable->name, when_calling_imm_, sig_imm_, "`");
    }
    BindScalar(variable.as_or_throw<PrimExpr>(), value, sources.front()->path,
               true);
  }
  for (const auto &binding : bindings) {
    current_guard_ = binding.guard;
    BindScalar(binding.argument, binding.value, binding.path, true);
  }
  for (const auto &shape : pending_int_exprs_) {
    current_guard_ = shape.guard;
    BindScalar(shape.arg, shape.value, shape.path, true);
  }
  current_guard_ = IntImm::Bool(true);
}

bool PackedArgumentDecoder::TryBindLinearIntExpr(
    const PrimExpr &arg, const PrimExpr &value,
    const ffi::reflection::AccessPath &path) {
  // Recognize factor * var + offset, allowing either operand order for Add and
  // Mul.
  PrimExpr var = arg;
  PrimExpr factor = IntImm(arg.ty(), 1);
  PrimExpr offset = IntImm(arg.ty(), 0);

  // Remove the constant offset, if present.
  if (const auto *add = var.as<AddNode>()) {
    if (add->b.as<IntImmNode>()) {
      var = add->a;
      offset = add->b;
    } else if (add->a.as<IntImmNode>()) {
      var = add->b;
      offset = add->a;
    } else {
      return false;
    }
  } else if (const auto *sub = var.as<SubNode>()) {
    if (!sub->b.as<IntImmNode>())
      return false;
    var = sub->a;
    offset = -sub->b;
  }

  // Remove the constant factor; the remainder must be a variable.
  if (const auto *mul = var.as<MulNode>()) {
    if (mul->b.as<IntImmNode>()) {
      var = mul->a;
      factor = mul->b;
    } else if (mul->a.as<IntImmNode>()) {
      var = mul->b;
      factor = mul->a;
    } else {
      return false;
    }
  }
  if (!var.as<VarNode>() || arg.same_as(var) || IsZero(factor) ||
      var_defs_.count(var.as<VarNode>())) {
    return false;
  }

  // Check divisibility before binding the inferred variable to the quotient.
  PrimExpr delta = value - offset;
  std::string shape_path = RenderAccessPath(path);
  std::string decoded_expr = shape_path;
  if (!IsZero(offset)) {
    std::ostringstream os;
    os << "(" << shape_path << " - " << offset.as_or_throw<IntImm>()->value
       << ")";
    decoded_expr = os.str();
  }
  if (!IsOne(factor)) {
    std::ostringstream os;
    os << factor.as_or_throw<IntImm>()->value;
    std::string factor_text = os.str();
    EmitAssert(truncmod(delta, factor) == 0, "ValueError", "Invalid ",
               shape_path, " on argument #",
               std::to_string(GetParamIndex(path)), when_calling_imm_, sig_imm_,
               "`,\n  expected ", decoded_expr, " to be divisible by ",
               factor_text);
    decoded_expr += " // " + factor_text;
  }
  BindScalar(var, truncdiv(delta, factor), path, true);
  shape_var_exprs_.emplace(var.as<VarNode>(), std::move(decoded_expr));
  return true;
}

bool PackedArgumentDecoder::BindPointer(const Var &arg, const Expr &value,
                                        const ffi::reflection::AccessPath &path,
                                        bool with_lets) {
  TVM_FFI_ICHECK(tl::StructuralEqual()(arg->ty, value->ty));
  auto it = var_defs_.find(arg.get());
  if (it == var_defs_.end()) {
    var_defs_.emplace(arg.get(), VarDefInfo{value, path});
    if (with_lets) {
      init_nest_.emplace_back(Bind(arg, value));
    }
    return true;
  }

  auto pointer_as_uint = [](const Expr &expr) {
    return Call(PrimType::UInt(64), tirx::reinterpret_op(), {expr})
        .as_or_throw<PrimExpr>();
  };
  PrimExpr prev_value = pointer_as_uint(it->second.value);
  PrimExpr current_value = pointer_as_uint(value);
  PrimExpr condition =
      tl::arith::Simplify(analyzer_, prev_value == current_value);
  if (IsZero(condition)) {
    TVM_FFI_THROW(InternalError)
        << "Bind has an unmet pointer assertion at " << RenderAccessPath(path);
  }
  if (!IsOne(condition)) {
    ffi::String current_path = RenderAccessPath(path);
    ffi::String first_path = RenderAccessPath(it->second.first_def_path);
    ffi::Array<StringImm> parts{StringImm("Mismatched "),
                                StringImm(current_path)};
    if (!first_path.empty() && first_path != current_path) {
      parts.push_back(StringImm("`,\n  expected to match "));
      parts.push_back(StringImm(first_path));
    }
    asserts_.emplace_back(
        AssertStmt(condition, StringImm("ValueError"), parts));
  }
  return false;
}

// ============================================================
// ExprPathRenderer (generic expr-to-string via ExprFunctor)
// ============================================================

/*!
 * \brief Render PrimExpr to string with variable names replaced by
 * ffi::reflection::AccessPath names.
 *
 * Uses ExprFunctor for generic dispatch over all expression types.
 * The default TIR printer sanitizes Var names (e.g. "B.shape[0]" ->
 * "B_shape_0_") and adds type annotations (e.g. T.int64(1)). This functor
 * preserves original path names and uses plain integer formatting for
 * human-readable error messages.
 */
class ExprPathRenderer : public ExprFunctor<std::string(const Expr &)> {
public:
  using FVarName = std::function<std::string(const VarNode *)>;
  explicit ExprPathRenderer(FVarName f_var_name)
      : f_var_name_(std::move(f_var_name)) {}

protected:
  std::string Dispatch_(const VarNode *op) final { return f_var_name_(op); }
  std::string Dispatch_(const IntImmNode *op) final {
    std::ostringstream os;
    os << op->value;
    return os.str();
  }
  std::string Dispatch_(const FloatImmNode *op) final {
    std::ostringstream os;
    os << op->value;
    return os.str();
  }
  std::string Dispatch_(const prim::CastNode *op) final {
    return Dispatch(op->value);
  }
  std::string Dispatch_(const prim::AddNode *op) final {
    return BinOp(op->a, " + ", op->b);
  }
  std::string Dispatch_(const prim::SubNode *op) final {
    return BinOp(op->a, " - ", op->b);
  }
  std::string Dispatch_(const prim::MulNode *op) final {
    return BinOp(op->a, " * ", op->b);
  }
  std::string Dispatch_(const prim::DivNode *op) final {
    return BinOp(op->a, " / ", op->b);
  }
  std::string Dispatch_(const prim::ModNode *op) final {
    return BinOp(op->a, " % ", op->b);
  }
  std::string Dispatch_(const prim::FloorDivNode *op) final {
    return FuncOp("floordiv", op->a, op->b);
  }
  std::string Dispatch_(const prim::FloorModNode *op) final {
    return FuncOp("floormod", op->a, op->b);
  }
  std::string Dispatch_(const prim::MinNode *op) final {
    return FuncOp("min", op->a, op->b);
  }
  std::string Dispatch_(const prim::MaxNode *op) final {
    return FuncOp("max", op->a, op->b);
  }
  // Fallback: use operator<< for unhandled expression types.
  std::string DispatchDefault_(const ffi::Object *op) final {
    std::ostringstream os;
    os << ffi::GetRef<Expr>(static_cast<const ExprNode *>(op));
    return os.str();
  }

private:
  std::string BinOp(const PrimExpr &a, const char *op, const PrimExpr &b) {
    return Dispatch(a) + op + Dispatch(b);
  }
  std::string FuncOp(const char *name, const PrimExpr &a, const PrimExpr &b) {
    return std::string(name) + "(" + Dispatch(a) + ", " + Dispatch(b) + ")";
  }
  FVarName f_var_name_;
};

// ============================================================
// RenderPendingAsserts
// ============================================================

void PackedArgumentDecoder::RenderPendingAsserts() {
  if (pending_const_asserts_.empty())
    return;

  ExprPathRenderer renderer([this](const VarNode *v) -> std::string {
    if (auto inferred = shape_var_exprs_.find(v);
        inferred != shape_var_exprs_.end()) {
      return "(" + inferred->second + ")";
    }
    auto it = var_defs_.find(v);
    if (it != var_defs_.end()) {
      ffi::String path = RenderAccessPath(it->second.first_def_path);
      if (!path.empty())
        return std::string(path);
    }
    return std::string(v->name);
  });

  for (auto &pending : pending_const_asserts_) {
    std::string display_str = renderer.Dispatch(pending.expected_expr);

    ffi::String path_str = RenderAccessPath(pending.path);
    int param_index = GetParamIndex(pending.path);
    ffi::Array<StringImm> parts;
    parts.push_back(StringImm("Invalid "));
    parts.push_back(StringImm(path_str));
    if (param_index >= 0) {
      parts.push_back(StringImm(" on argument #"));
      parts.push_back(StringImm(std::to_string(param_index)));
    }
    parts.push_back(when_calling_imm_);
    parts.push_back(sig_imm_);
    parts.push_back(StringImm("`,\n  expected "));
    parts.push_back(StringImm(display_str));
    asserts_.emplace_back(
        AssertStmt(pending.condition, StringImm("ValueError"), parts));
  }
  pending_const_asserts_.clear();
}

// ============================================================
// Finalize
// ============================================================

PackedArgumentDecoder::Result PackedArgumentDecoder::Finalize() {
  RenderPendingAsserts();
  Result result;
  result.var_defs = std::move(var_defs_);
  result.init_nest = std::move(init_nest_);
  result.asserts = std::move(asserts_);
  result.decl_tensors = std::move(decl_tensors_);
  return result;
}

// ============================================================
// Per-dtype type-check + value-load methods
// ============================================================

/*! \brief Load the i-th packed argument as the given type. */
Expr PackedArgumentDecoder::LoadTVMFFIAnyUnionValue(const Var &v_packed_args,
                                                    int param_index,
                                                    Type arg_type) {
  ffi::Array<Expr> call_args{v_packed_args, IntImm::Int32(param_index),
                             IntImm::Int32(tirx::kTVMFFIAnyUnionValue)};
  if (auto prim_type = arg_type.as<PrimType>()) {
    PrimType api_type = PackedScalarType(prim_type.value());
    PrimExpr res = Call(api_type, tirx::abi_field_get_op(), call_args)
                       .as_or_throw<PrimExpr>();
    if (api_type != prim_type.value()) {
      res = prim::Cast(prim_type.value(), res);
    }
    return res;
  }
  TVM_FFI_CHECK(arg_type.as<PtrTypeNode>(), TypeError)
      << "Packed union values must have primitive or pointer type, but got "
      << arg_type;
  return Call(std::move(arg_type), tirx::abi_field_get_op(), call_args);
}

Expr PackedArgumentDecoder::DecodeParamOpaqueHandle(
    int param_index, const PrimExpr &type_index) {
  // ── Type check: accept handle-like types ───────────────────
  bool is_tensor = params_[param_index]->ty.as<TensorTypeNode>() != nullptr;
  std::string expected_type = is_tensor ? "Tensor" : "pointer";
  PrimExpr object_type =
      is_tensor ? type_index == ffi::TypeIndex::kTVMFFITensor
                : type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin;
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFINone ||
                         type_index == ffi::TypeIndex::kTVMFFIOpaquePtr ||
                         type_index == ffi::TypeIndex::kTVMFFIDLTensorPtr ||
                         object_type,
                     expected_type);

  // ── Load value and apply tensor offset ─────────────────────
  const int64_t object_cell_offset = sizeof(TVMFFIObject);
  static_assert(sizeof(TVMFFIObject) == 24);
  Expr arg_value = LoadTVMFFIAnyUnionValue(v_packed_args_, param_index,
                                           PtrType::VoidPtrType());
  Expr handle_from_tensor =
      Call(PtrType::VoidPtrType(), tirx::handle_add_byte_offset_op(),
           {arg_value, IntImm::Int32(object_cell_offset)});
  return Call(PtrType::VoidPtrType(), prim::if_then_else_op(),
              {type_index == ffi::TypeIndex::kTVMFFITensor, handle_from_tensor,
               arg_value});
}

PrimExpr PackedArgumentDecoder::DecodeParamBool(int param_index,
                                                const PrimExpr &type_index) {
  // ── Type check: accept bool or int ─────────────────────────
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFIBool ||
                         type_index == ffi::TypeIndex::kTVMFFIInt,
                     "boolean");
  return prim::Cast(
      PrimType::Bool(),
      LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, PrimType::Int(64)));
}

PrimExpr PackedArgumentDecoder::DecodeParamInt(int param_index,
                                               const PrimExpr &type_index,
                                               PrimType dtype) {
  // ── Type check: accept int or bool ─────────────────────────
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFIInt ||
                         type_index == ffi::TypeIndex::kTVMFFIBool,
                     "int");
  return LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, dtype);
}

PrimExpr PackedArgumentDecoder::DecodeParamFloat(int param_index,
                                                 const PrimExpr &type_index,
                                                 PrimType dtype) {
  // ── Type check: accept float, int, or bool ─────────────────
  EmitTypeIndexCheck(param_index,
                     type_index == ffi::TypeIndex::kTVMFFIFloat ||
                         type_index == ffi::TypeIndex::kTVMFFIInt ||
                         type_index == ffi::TypeIndex::kTVMFFIBool,
                     "float");
  return prim::Select(
      type_index == ffi::TypeIndex::kTVMFFIFloat,
      /* true_value = */
      LoadTVMFFIAnyUnionValue(v_packed_args_, param_index, dtype),
      /* false_value = */
      prim::Cast(dtype, LoadTVMFFIAnyUnionValue(v_packed_args_, param_index,
                                                PrimType::Int(64))));
}

// ============================================================
// DecodeParam (primary public method)
// ============================================================

void PackedArgumentDecoder::DecodeParam(int param_index) {
  Var param = params_[param_index];

  // Extract type_index from packed_args
  Var type_index(param->name + ".type_index", PrimType::Int(32));
  init_nest_.push_back(
      Bind(type_index, Call(PrimType::Int(32), tirx::abi_field_get_op(),
                            {v_packed_args_, IntImm::Int32(param_index),
                             IntImm::Int32(tirx::kTVMFFIAnyTypeIndex)})
                           .as_or_throw<PrimExpr>()));

  ffi::reflection::AccessPath param_path =
      ffi::reflection::AccessPath::Root()->Extend(
          AccessStep::ArrayItem(param_index));

  if (param->ty.as<NoneTypeNode>()) {
    // A None parameter still occupies an ABI slot, even when the body ignores
    // it.
    EmitTypeIndexCheck(param_index,
                       type_index.as_or_throw<PrimExpr>() ==
                           ffi::TypeIndex::kTVMFFINone,
                       "None");
    return;
  }

  if (param->ty.as<TensorTypeNode>()) {
    Var handle(param->name + ".handle", PtrType::VoidPtrType());
    Expr handle_value = DecodeParamOpaqueHandle(
        param_index, type_index.as_or_throw<PrimExpr>());
    BindPointer(handle, handle_value, param_path, true);
    tensor_handles_.emplace(param.get(), handle);
    return;
  }

  if (param->ty.as<PtrTypeNode>()) {
    Expr handle_value = DecodeParamOpaqueHandle(
        param_index, type_index.as_or_throw<PrimExpr>());
    Expr pointer_value =
        Call(param->ty, tirx::reinterpret_op(), {handle_value});
    BindPointer(param, pointer_value, param_path, true);
    return;
  }

  PrimType dtype = param->ty.as_or_throw<PrimType>();

  // Type-check and load value via per-dtype dispatch
  PrimExpr arg_value{ffi::UnsafeInit{}};
  if (dtype.MatchesCode(DLDataTypeCode::kDLBool)) {
    arg_value =
        DecodeParamBool(param_index, type_index.as_or_throw<PrimExpr>());
  } else if (dtype.MatchesCode(DLDataTypeCode::kDLInt,
                               DLDataTypeCode::kDLUInt)) {
    arg_value =
        DecodeParamInt(param_index, type_index.as_or_throw<PrimExpr>(), dtype);
  } else {
    TVM_FFI_ICHECK(
        dtype.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat));
    arg_value = DecodeParamFloat(param_index,
                                 type_index.as_or_throw<PrimExpr>(), dtype);
  }

  // Bind scalar param to loaded value (defines vars before tensor binds
  // reference them)
  BindScalar(param.as_or_throw<PrimExpr>(), arg_value, param_path, true);
}

// ============================================================
// DecodeAllParams (primary public method)
// ============================================================

void PackedArgumentDecoder::DecodeAllParams() {
  int num_args = static_cast<int>(params_.size());

  // Decode scalar parameters before any tensor shape can reference them.
  for (int i = 0; i < num_args; ++i) {
    DecodeParam(i);
  }

  // Shape pass 1: Prefer bare-variable dimensions across all arguments.
  // Per-tensor safety checks still precede the corresponding field loads.
  std::vector<int> tensor_order;
  for (int index = 0; index < num_args; ++index) {
    if (!nullable_parameters_.count(params_[index].get()))
      tensor_order.push_back(index);
  }
  for (int index = 0; index < num_args; ++index) {
    if (nullable_parameters_.count(params_[index].get()))
      tensor_order.push_back(index);
  }
  for (int index : tensor_order) {
    Var param = params_[index];
    if (auto tensor = param.as<TensorVar>()) {
      Var handle = tensor_handles_.at(param.get());
      defer_scalar_bindings_ = nullable_parameters_.count(param.get());
      current_guard_ =
          defer_scalar_bindings_
              ? !Call(PrimType::Bool(), tirx::isnullptr_op(), {handle})
                     .as_or_throw<PrimExpr>()
              : PrimExpr(IntImm::Bool(true));
      ffi::reflection::AccessPath param_path =
          ffi::reflection::AccessPath::Root()
              ->Extend(AccessStep::ArrayItem(index))
              ->Attr(ffi::String(tensor.value()->name));
      Expr data =
          DecodeParamDLTensor(tensor.value(), device_type_, device_id_, handle,
                              func_name_ + "." + param->name, param_path);
      decl_tensors_.push_back(
          Bind(tensor.value(), Call(tensor.value().ty(), tirx::decl_tensor_op(),
                                    {data}, {}, {tensor.value().ty()})));
      PrimExpr size = IntImm(tensor.value().ty()->DefaultIndexType(), 1);
      for (const auto &extent : tensor.value().shape())
        size *= extent;
      Stmt alignment =
          Evaluate(Call(PrimType::Void(), tirx::assume_aligned_op(),
                        {tensor.value(),
                         IntImm::Int32(tensor.value().ty()->data_alignment)}));
      // Empty tensors and opaque device handles carry no data-address alignment
      // promise. DecodeParamDLTensor has already checked or bound the expected
      // device type.
      PrimExpr device_type = device_type_;
      PrimExpr direct_address = device_type <= IntImm::Int32(kDLCUDAHost) ||
                                device_type == IntImm::Int32(kDLCUDAManaged) ||
                                device_type == IntImm::Int32(kDLROCM) ||
                                device_type == IntImm::Int32(kDLROCMHost);
      PrimExpr aligned_storage = tl::arith::Simplify(
          analyzer_, current_guard_ && size != 0 && direct_address);
      if (IsOne(aligned_storage)) {
        decl_tensors_.push_back(alignment);
      } else if (!IsZero(aligned_storage)) {
        decl_tensors_.push_back(If(aligned_storage, alignment));
      }
    }
  }
  current_guard_ = IntImm::Bool(true);
  defer_scalar_bindings_ = false;
  if (!nullable_parameters_.empty()) {
    BindNullableMetadata();
    pending_int_exprs_.clear();
    return;
  }
  // Integer-expression pass 2: Infer only variables not defined by a bare
  // dimension.
  for (auto &shape : pending_int_exprs_) {
    shape.needs_check =
        !TryBindLinearIntExpr(shape.arg, shape.value, shape.path);
  }
  // Integer-expression pass 3: Check the remaining expressions after all
  // bindings are known.
  for (const auto &shape : pending_int_exprs_) {
    if (shape.needs_check) {
      BindScalar(shape.arg, shape.value, shape.path, true);
    }
  }
  pending_int_exprs_.clear();
}

// ============================================================
// DLTensorGetFieldPtr (helper for shape/strides pointer extraction)
// ============================================================

Var PackedArgumentDecoder::DLTensorGetFieldPtr(const Var &handle,
                                               int field_kind,
                                               const std::string &var_name) {
  Type pointer_type = PtrType(DefaultIndexType());
  Var ptr(var_name, pointer_type);
  init_nest_.emplace_back(Bind(
      ptr,
      GuardValue(Call(pointer_type, tirx::abi_field_get_op(),
                      {handle, IntImm::Int32(0), IntImm::Int32(field_kind)}))));
  return ptr;
}

// ============================================================
// LoadInt64ArrayElem (helper for shape/strides element access)
// ============================================================

PrimExpr PackedArgumentDecoder::LoadInt64ArrayElem(const Var &ptr, int index) {
  return Call(DefaultIndexType(), tirx::abi_field_get_op(),
              {ptr, IntImm::Int32(index), IntImm::Int32(tirx::kInt64ArrayElem)})
      .as_or_throw<PrimExpr>();
}

// ============================================================
// Strides validation subfunctions
// ============================================================

int PackedArgumentDecoder::StoragePackingFactor(const TensorVar &tensor) const {
  PrimType storage = storage_dtypes_.Get(tensor).value_or(
      ::tvm::tl::CustomDTypeStorage(tensor.dtype()));
  return storage.bits() * storage.lanes() /
         (tensor.dtype().bits() * tensor.dtype().lanes());
}

void PackedArgumentDecoder::BindCompactStrides(
    const TensorVar &tensor, const Var &strides_ptr,
    const PrimExpr &v_strides_is_null,
    const ffi::reflection::AccessPath &param_path) {
  PrimType stype = tensor.ty()->DefaultIndexType();
  PrimExpr expect_stride = IntImm(stype, 1);
  int packing_factor = StoragePackingFactor(tensor);
  ffi::Array<PrimExpr> conds;
  PrimExpr empty = IntImm::Bool(false);
  for (const PrimExpr &extent : tensor.shape())
    empty = logical_or(empty, extent == 0);
  for (size_t i = tensor.shape().size(); i != 0; --i) {
    size_t k = i - 1;
    PrimExpr svalue = tl::CastValue(LoadInt64ArrayElem(strides_ptr, k), stype);
    if (k + 1 < tensor.shape().size())
      svalue *= packing_factor;
    conds.push_back(tensor.shape()[k] == 1 || expect_stride == svalue);
    expect_stride = expect_stride * tensor.shape()[k];
  }
  if (conds.size() != 0) {
    int param_index = GetParamIndex(param_path);
    Stmt check = AssertStmt(
        foldl([](PrimExpr a, PrimExpr b,
                 Location loc) { return logical_and(a, b, loc); },
              IntImm::Bool(true), conds),
        StringImm("ValueError"),
        ffi::Array<StringImm>(
            {StringImm("Mismatched "), StringImm(tensor->name),
             StringImm(".strides on argument #"),
             StringImm(std::to_string(param_index)), when_calling_imm_,
             sig_imm_, StringImm("`,\n  expected to be compact array")}));
    check =
        If(logical_and(prim::Not(v_strides_is_null), prim::Not(empty)), check);
    if (!IsOne(current_guard_))
      check = If(current_guard_, check);
    asserts_.emplace_back(SeqStmt({check}));
  }
}

void PackedArgumentDecoder::BindRegularStrides(
    const TensorVar &tensor, const Var &strides_ptr, const Var &shape_ptr,
    const PrimExpr &v_strides_is_null,
    const ffi::reflection::AccessPath &param_path) {
  PrimExpr stride_from_shape = 1;
  int packing_factor = StoragePackingFactor(tensor);
  auto strides = MatchShapeStridePattern(tensor.ty()).value().get<1>();
  for (int k = strides.size() - 1; k >= 0; k--) {
    PrimType stride_type = strides[k].ty();
    PrimExpr explicit_stride =
        tl::CastValue(LoadInt64ArrayElem(strides_ptr, k), stride_type);
    ffi::reflection::AccessPath strides_k_path =
        param_path->Attr(ffi::String("strides"))->ArrayItem(k);
    PrimExpr decoded = tvm::if_then_else(
        v_strides_is_null, tl::CastValue(stride_from_shape, stride_type),
        explicit_stride);
    if (packing_factor > 1 && k == static_cast<int>(strides.size()) - 1) {
      EmitAssert(v_strides_is_null || explicit_stride == 1, "ValueError",
                 "Packed tensor storage requires unit innermost stride");
      BindScalar(strides[k], IntImm(stride_type, 1), strides_k_path, true);
    } else {
      BindScalar(strides[k], decoded * packing_factor, strides_k_path, true);
    }
    stride_from_shape *=
        tl::CastValue(LoadInt64ArrayElem(shape_ptr, k), tensor.shape()[k].ty());
  }
}

// ============================================================
// DecodeParamDLTensor (private)
// ============================================================

Expr PackedArgumentDecoder::DecodeParamDLTensor(
    const TensorVar &tensor, const PrimExpr &device_type,
    const PrimExpr &device_id, const Var &handle, const std::string &arg_name,
    ffi::reflection::AccessPath base_path) {
  auto pattern = MatchShapeStridePattern(tensor.ty());
  TVM_FFI_CHECK(pattern.has_value(), ValueError)
      << "Tensor parameters require a shape-aligned strided memory layout";
  PrimExpr elem_offset = pattern.value().get<2>();
  PrimType storage_dtype = storage_dtypes_.Get(tensor).value_or(
      ::tvm::tl::CustomDTypeStorage(tensor.dtype()));
  int packing_factor = StoragePackingFactor(tensor);
  const PrimType tvm_ndim_type = PrimType::Int(32);

  std::string buf_name = tensor->name;
  ffi::reflection::AccessPath param_path = base_path;
  int param_index = GetParamIndex(base_path);

  // ── Section: Null pointer check ──────────────────────────────
  EmitTypeIndexCheck(param_index,
                     !Call(PrimType::Bool(), tirx::isnullptr_op(), {handle})
                          .as_or_throw<PrimExpr>(),
                     "Tensor");

  // ── Section: ndim ────────────────────────────────────────────
  PrimExpr v_ndim =
      Call(tvm_ndim_type, tirx::abi_field_get_op(),
           {handle, IntImm::Int32(0), IntImm::Int32(tirx::kDLTensorNDim)})
          .as_or_throw<PrimExpr>();
  PrimExpr a_ndim =
      IntImm(tvm_ndim_type, static_cast<int64_t>(tensor.shape().size()));
  EmitAssert(a_ndim == v_ndim, "ValueError", //
             "Mismatched ", buf_name, ".ndim on argument #",
             std::to_string(param_index), when_calling_imm_, sig_imm_,
             "`,\n  expected ", std::to_string(tensor.shape().size()));

  // ── Section: dtype ───────────────────────────────────────────
  {
    PrimExpr code_matches =
        Call(PrimType::UInt(8), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0), IntImm::Int32(tirx::kDLTensorTypeCode)})
            .as_or_throw<PrimExpr>() ==
        IntImm(PrimType::UInt(8), storage_dtype.code());
    PrimExpr bits_matches =
        Call(PrimType::UInt(8), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0), IntImm::Int32(tirx::kDLTensorTypeBits)})
            .as_or_throw<PrimExpr>() ==
        IntImm(PrimType::UInt(8), storage_dtype.bits());
    PrimExpr lanes_matches = Call(PrimType::UInt(16), tirx::abi_field_get_op(),
                                  {handle, IntImm::Int32(0),
                                   IntImm::Int32(tirx::kDLTensorTypeLanes)})
                                 .as_or_throw<PrimExpr>() ==
                             IntImm(PrimType::UInt(16), storage_dtype.lanes());
    PrimExpr cond = code_matches && bits_matches && lanes_matches;
    std::ostringstream dtype_os;
    dtype_os << storage_dtype->dtype;
    EmitAssert(cond, "TypeError", "Mismatched ", buf_name,
               ".dtype on argument #", std::to_string(param_index),
               when_calling_imm_, sig_imm_, "`,\n  expected ", dtype_os.str());
  }

  // ── Section: shape ───────────────────────────────────────────
  Var shape_ptr =
      DLTensorGetFieldPtr(handle, tirx::kDLTensorShape, arg_name + "_shape");
  for (size_t k = 0; k < tensor.shape().size(); ++k) {
    ffi::reflection::AccessPath shape_k_path =
        param_path->Attr(ffi::String("shape"))->ArrayItem(k);
    PrimExpr shape = tensor.shape()[k];
    PrimExpr value =
        tl::CastValue(LoadInt64ArrayElem(shape_ptr, k), shape.ty());
    if (k + 1 == tensor.shape().size())
      value *= packing_factor;
    if (shape.as<VarNode>()) {
      BindScalar(shape, value, shape_k_path, true);
    } else {
      pending_int_exprs_.push_back(
          {shape, value, shape_k_path, true, current_guard_});
    }
  }

  // ── Section: strides ─────────────────────────────────────────
  Var strides_ptr = DLTensorGetFieldPtr(handle, tirx::kDLTensorStrides,
                                        arg_name + "_strides");
  PrimExpr v_strides_is_null =
      Call(PrimType::Bool(), tirx::isnullptr_op(), {strides_ptr})
          .as_or_throw<PrimExpr>();
  if (!tensor.ty()->layout.has_value()) {
    BindCompactStrides(tensor, strides_ptr, v_strides_is_null, param_path);
  } else {
    BindRegularStrides(tensor, strides_ptr, shape_ptr, v_strides_is_null,
                       param_path);
  }

  // ── Section: byte_offset ─────────────────────────────────────
  int element_bits = tensor.dtype().MatchesCode(kDLBool)
                         ? tensor.dtype().StorageBytes() * 8
                         : tensor.dtype().bits() * tensor.dtype().lanes();
  ffi::reflection::AccessPath byte_offset_path =
      param_path->Attr(ffi::String("byte_offset"));
  if (const auto *const_offset = elem_offset.as<IntImmNode>()) {
    TVM_FFI_CHECK((const_offset->value * element_bits) % 8 == 0, ValueError)
        << "Tensor element offset must be byte-aligned";
    BindScalar(
        IntImm(PrimType::UInt(64), const_offset->value * element_bits / 8),
        Call(PrimType::UInt(64), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0),
              IntImm::Int32(tirx::kDLTensorByteOffset)})
            .as_or_throw<PrimExpr>(),
        byte_offset_path, true);
  } else {
    PrimExpr byte_offset = Call(PrimType::UInt(64), tirx::abi_field_get_op(),
                                {handle, IntImm::Int32(0),
                                 IntImm::Int32(tirx::kDLTensorByteOffset)})
                               .as_or_throw<PrimExpr>();
    PrimExpr offset_bits = byte_offset * tl::MakeConst(PrimType::UInt(64), 8);
    EmitAssert(truncmod(offset_bits,
                        tl::MakeConst(PrimType::UInt(64), element_bits)) ==
                   tl::MakeConst(PrimType::UInt(64), 0),
               "ValueError",
               "Tensor byte offset must be divisible by its element size");
    PrimExpr runtime_origin = tl::CastValue(
        offset_bits / tl::MakeConst(PrimType::UInt(64), element_bits),
        elem_offset.ty());
    if (elem_offset.as<VarNode>()) {
      BindScalar(elem_offset, runtime_origin, byte_offset_path, true);
    } else {
      // Linear offsets bind in element units, after byte-offset conversion.
      pending_int_exprs_.push_back(
          {elem_offset, runtime_origin,
           param_path->Attr(ffi::String("elem_offset")), true, current_guard_});
    }
  }

  // ── Section: device ──────────────────────────────────────────
  {
    PrimExpr actual_device_type =
        Call(PrimType::Int(32), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0),
              IntImm::Int32(tirx::kDLTensorDeviceType)})
            .as_or_throw<PrimExpr>();
    // Use custom assertion for device_type to show human-readable device name
    if (const auto *const_dt = device_type_.as<IntImmNode>()) {
      PrimExpr cond = tl::arith::Simplify(
          analyzer_, IntImm::Int32(const_dt->value) == actual_device_type);
      if (!IsOne(cond)) {
        std::string device_name =
            runtime::DLDeviceType2Str(const_dt->value.as<int>().value());
        EmitAssert(cond, "ValueError", //
                   "Mismatched ", buf_name, ".device_type on argument #",
                   std::to_string(param_index), when_calling_imm_, sig_imm_,
                   "`,\n  expected ", device_name);
      }
    } else {
      ffi::reflection::AccessPath device_type_path =
          param_path->Attr(ffi::String("device_type"));
      BindScalar(device_type_, actual_device_type, device_type_path, true);
    }
    ffi::reflection::AccessPath device_id_path =
        param_path->Attr(ffi::String("device_id"));
    BindScalar(
        device_id_,
        Call(PrimType::Int(32), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0), IntImm::Int32(tirx::kDLTensorDeviceId)})
            .as_or_throw<PrimExpr>(),
        device_id_path, true);
  }

  // ── Section: data pointer ────────────────────────────────────
  {
    ffi::reflection::AccessPath data_path =
        param_path->Attr(ffi::String("data"));
    Expr raw_data =
        Call(PtrType::VoidPtrType(), tirx::abi_field_get_op(),
             {handle, IntImm::Int32(0), IntImm::Int32(tirx::kDLTensorData)});
    Expr typed_data =
        Call(tensor.ty()->DataPtrType(), tirx::reinterpret_op(), {raw_data});
    {
      Expr vptr = typed_data;

      auto alloc_size = [&]() -> PrimExpr {
        PrimExpr product = IntImm(tensor.ty()->DefaultIndexType(), 1);
        for (const auto &dim : tensor.shape()) {
          product *= dim;
        }
        return product;
      }();
      // Data pointer null and alignment checks go to asserts_ because
      // alloc_size references tensor->shape which may contain
      // forward-referenced symbolic vars.
      PrimExpr empty_alloc = tl::CastValue(alloc_size == 0, PrimType::Bool());
      PrimExpr data_non_null =
          !Call(PrimType::Bool(), tirx::isnullptr_op(), {vptr})
               .as_or_throw<PrimExpr>();
      asserts_.emplace_back(AssertStmt(
          GuardCondition(empty_alloc || data_non_null), StringImm("ValueError"),
          ffi::Array<StringImm>(
              {StringImm(buf_name),
               StringImm(" data pointer is NULL on argument #"),
               StringImm(std::to_string(param_index)), when_calling_imm_,
               sig_imm_, StringImm("`,\n  expected non-NULL data pointer")})));

      {
        // Only direct-address devices expose the data address in
        // DLTensor::data. Opaque device handles (such as OpenCL buffer
        // descriptors) have no such alignment. Reuse the checked device type so
        // fixed-target guards simplify statically.
        PrimExpr device_type = device_type_;
        PrimExpr direct_address =
            device_type <= IntImm::Int32(kDLCUDAHost) ||
            device_type == IntImm::Int32(kDLCUDAManaged) ||
            device_type == IntImm::Int32(kDLROCM) ||
            device_type == IntImm::Int32(kDLROCMHost);
        // Keep this predicate consistent with ffi::IsDirectAddressDevice.
        // Check the allocation base, matching the assume_aligned on tensor
        // data.
        if (tensor.ty()->data_alignment > 1) {
          Expr handle = Call(PtrType::VoidPtrType(), tirx::reinterpret_op(),
                             ffi::Array<Expr>{vptr});
          PrimExpr ptr_as_int = Call(PrimType::UInt(64), tirx::reinterpret_op(),
                                     ffi::Array<Expr>{handle})
                                    .as_or_throw<PrimExpr>();
          PrimExpr align_cond =
              truncmod(ptr_as_int, IntImm(PrimType::UInt(64),
                                          tensor.ty()->data_alignment)) ==
              IntImm(PrimType::UInt(64), 0);
          asserts_.emplace_back(AssertStmt(
              GuardCondition(empty_alloc || !direct_address || align_cond),
              StringImm("ValueError"),
              ffi::Array<StringImm>(
                  {StringImm("Misaligned Tensor data on argument #"),
                   StringImm(std::to_string(param_index)), when_calling_imm_,
                   sig_imm_, StringImm("`,\n  expected data alignment="),
                   StringImm(std::to_string(tensor.ty()->data_alignment)),
                   StringImm(" bytes")})));
        }
      }
    }
    return GuardValue(typed_data);
  }
}

} // namespace tvm::tl::tirx_passes
