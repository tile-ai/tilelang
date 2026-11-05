/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_custom_datatypes.cc
 * \brief Lower host custom storage and native user datatype callbacks.
 *
 * Callback selection and missing-callback errors follow TVM
 * 4211874e9e4b8770fd20ad0d98ab80afd1df5028, target/datatype/registry.cc and
 * tirx/transform/lower_custom_datatypes.cc. All traversal and definition/use
 * rewriting use the native shared IR, without an old Buffer/object registry.
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

#include "op/native_utils.h"

namespace tvm::tl {
namespace {
using namespace tirx;

bool Registered(const PrimType &type) {
  if (static_cast<int>(type.code()) < 129)
    return false;
  static const auto get_registered =
      ffi::Function::GetGlobal("runtime._datatype_get_type_registered").value();
  return get_registered(static_cast<int>(type.code())).cast<bool>();
}

PrimType StorageType(const PrimType &type) {
  return Registered(type) ? PrimType::UInt(type.bits(), type.lanes()) : type;
}

std::string TypeName(const PrimType &type) {
  if (!Registered(type))
    return ffi::details::DLDataTypeCodeAsCStr(type.code());
  static const auto get_name =
      ffi::Function::GetGlobal("dtype.get_custom_type_name").value();
  return get_name(static_cast<int>(type.code())).cast<std::string>();
}

class CustomDatatypeLowerer : public StmtExprMutator {
public:
  explicit CustomDatatypeLowerer(std::string target)
      : target_(std::move(target)) {}
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  // Rebuild the complete TensorType/PtrType, including shape/layout references,
  // through the native structural protocol. A dtype's raw ABI code literals
  // and DataTypeImm values remain logical; they are not PrimType objects.
  ffi::UnchangedOr<ffi::Any>
  Mutate(ffi::AnyView value, InplaceMode mode = InplaceMode::kDisallow) final {
    if (const auto *type = value.as<PrimTypeObj>()) {
      auto original = ffi::GetRef<PrimType>(type);
      auto physical = StorageType(original);
      if (physical == original)
        return ffi::Unchanged();
      return physical;
    }
    return StmtExprMutator::Mutate(value, mode);
  }

  Var Define(const Var &var) {
    auto type =
        Mutate(var->ty).as_or_throw<ffi::UnchangedOr<Type>>().ValueOrUnchanged(
            var->ty);
    if (type.same_as(var->ty))
      return var;
    Var replacement = var.CopyWithType(type);
    VarRemapSet(var, replacement);
    return replacement;
  }

  ffi::UnchangedOr<Expr> Mutate_(const VarObj *var, InplaceMode mode) final {
    // The core mutator intentionally skips primitive definition types. Custom
    // scalar Bind/region/Let definitions need the same explicit remap as views.
    if (def_region_kind() != kTVMFFIDefRegionKindNone) {
      if (const auto *type = var->ty.as<PrimTypeObj>();
          type && Registered(ffi::GetRef<PrimType>(type))) {
        return Define(ffi::GetRef<Var>(var));
      }
    }
    return StmtExprMutator::Mutate_(var, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::CastObj *node,
                                     InplaceMode mode) final {
    PrimType destination = node->ty.as_or_throw<PrimType>();
    PrimType source = node->value.ty();
    if (!Registered(destination) && !Registered(source))
      return StmtExprMutator::Mutate_(node, mode);
    auto scratch = ffi::make_object<prim::CastObj>(*node);
    scratch->value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    auto lower = ffi::Function::GetGlobal("tvm.datatype.lower." + target_ +
                                          ".Cast." + TypeName(destination) +
                                          "." + TypeName(source));
    TVM_FFI_ICHECK(lower) << "Cast lowering function for target " << target_
                          << " destination type "
                          << static_cast<unsigned>(destination.code())
                          << " source type "
                          << static_cast<unsigned>(source.code())
                          << " not found";
    return Invoke(*lower, PrimExpr(scratch), StorageType(destination));
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const FloatImmObj *node,
                                     InplaceMode mode) final {
    PrimType type = node->ty.as_or_throw<PrimType>();
    if (!Registered(type))
      return StmtExprMutator::Mutate_(node, mode);
    auto lower = ffi::Function::GetGlobal("tvm.datatype.lower." + target_ +
                                          ".FloatImm." + TypeName(type));
    TVM_FFI_ICHECK(lower) << "FloatImm lowering function for target " << target_
                          << " type " << static_cast<unsigned>(type.code())
                          << " not found";
    return Invoke(*lower, ffi::GetRef<PrimExpr>(node), StorageType(type));
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *node, InplaceMode mode) final {
    const auto *type = node->ty.as<PrimTypeObj>();
    if (!type || !Registered(ffi::GetRef<PrimType>(type))) {
      // ABI and extern calls can carry an explicit primitive result without an
      // infer-type callback. Reinference would replace that type with
      // MissingType even when no argument changed. Structural mutation already
      // rewrites non-primitive result types and type arguments (e.g. tensor
      // views).
      return StmtExprMutator::Mutate_(node, mode);
    }
    auto scratch = ffi::make_object<CallObj>(*node);
    scratch->args = Mutate(node->args, mode)
                        .as_or_throw<ffi::UnchangedOr<ffi::Array<Expr>>>()
                        .ValueOrUnchanged(node->args);
    const auto *op = node->op.as<OpObj>();
    TVM_FFI_ICHECK(op) << "Lowering non-intrinsic Calls not implemented";
    PrimType logical = ffi::GetRef<PrimType>(type);
    auto lower = ffi::Function::GetGlobal(
        "tvm.datatype.lower." + target_ + ".Call.intrin." +
        std::string(op->name) + "." + TypeName(logical));
    TVM_FFI_ICHECK(lower) << "Intrinsic lowering function for target "
                          << target_ << ", intrinsic name " << op->name
                          << ", type " << static_cast<unsigned>(logical.code())
                          << " not found";
    return Invoke(*lower, PrimExpr(scratch), StorageType(logical));
  }

#define TILELANG_LOWER_CUSTOM_BINARY(OP)                                       \
  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::OP##Obj *node,                \
                                     InplaceMode mode) final {                 \
    PrimType logical = node->ty.as_or_throw<PrimType>();                       \
    if (!Registered(logical))                                                  \
      return StmtExprMutator::Mutate_(node, mode);                             \
    auto scratch = ffi::make_object<prim::OP##Obj>(*node);                     \
    scratch->a = Mutate(node->a, mode).ValueOrUnchanged(node->a);              \
    scratch->b = Mutate(node->b, mode).ValueOrUnchanged(node->b);              \
    auto lower = ffi::Function::GetGlobal("tvm.datatype.lower." + target_ +    \
                                          "." #OP "." + TypeName(logical));    \
    TVM_FFI_ICHECK(lower) << #OP " lowering function for target " << target_   \
                          << " type " << static_cast<unsigned>(logical.code()) \
                          << " not found";                                     \
    return Invoke(*lower, PrimExpr(scratch), StorageType(logical));            \
  }

  TILELANG_LOWER_CUSTOM_BINARY(Add)
  TILELANG_LOWER_CUSTOM_BINARY(Sub)
  TILELANG_LOWER_CUSTOM_BINARY(Mul)
  TILELANG_LOWER_CUSTOM_BINARY(Div)
  TILELANG_LOWER_CUSTOM_BINARY(Mod)
  TILELANG_LOWER_CUSTOM_BINARY(Min)
  TILELANG_LOWER_CUSTOM_BINARY(Max)
  TILELANG_LOWER_CUSTOM_BINARY(EQ)
  TILELANG_LOWER_CUSTOM_BINARY(NE)
  TILELANG_LOWER_CUSTOM_BINARY(LT)
  TILELANG_LOWER_CUSTOM_BINARY(LE)
  TILELANG_LOWER_CUSTOM_BINARY(GT)
  TILELANG_LOWER_CUSTOM_BINARY(GE)
#undef TILELANG_LOWER_CUSTOM_BINARY

  // These nodes select, bind, or arrange values; they do not define a custom
  // arithmetic operation. Their result tracks the lowered child's storage type.
#define TILELANG_RETYPE_CUSTOM_CONTAINER(OP)                                   \
  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::OP##Obj *node,                \
                                     InplaceMode mode) final {                 \
    auto result = StmtExprMutator::Mutate_(node, mode);                        \
    PrimType logical = node->ty.as_or_throw<PrimType>();                       \
    if (!Registered(logical))                                                  \
      return result;                                                           \
    auto expression =                                                          \
        std::move(result).ValueOrUnchanged(ffi::GetRef<PrimExpr>(node));       \
    auto copy =                                                                \
        ffi::make_object<prim::OP##Obj>(*expression.as<prim::OP##Obj>());      \
    copy->ty = StorageType(logical);                                           \
    return PrimExpr(copy);                                                     \
  }
  TILELANG_RETYPE_CUSTOM_CONTAINER(Select)
  TILELANG_RETYPE_CUSTOM_CONTAINER(Broadcast)
  TILELANG_RETYPE_CUSTOM_CONTAINER(Shuffle)
  TILELANG_RETYPE_CUSTOM_CONTAINER(Let)
#undef TILELANG_RETYPE_CUSTOM_CONTAINER

private:
  PrimExpr Invoke(const ffi::Function &callback, const PrimExpr &scratch,
                  const PrimType &expected) {
    // scratch preserves the logical result type but has lowered children, as
    // the old one-argument callback contract requires. It stays local to this
    // call and is never published, verified, serialized, or cached as final IR.
    ffi::Any value = callback(scratch);
    const auto *expr = value.as<ExprObj>();
    TVM_FFI_CHECK(expr && expr->ty.as<PrimTypeObj>(), TypeError)
        << "Custom datatype lowering callback must return a primitive native "
           "Expr";
    PrimExpr result = value.cast<PrimExpr>();
    TVM_FFI_CHECK(result.ty() == expected, TypeError)
        << "Custom datatype lowering callback returned " << result.ty()
        << ", expected physical storage type " << expected;
    return result;
  }

  std::string target_;
};

tirx::Function Lower(tirx::Function function) {
  auto target = function->GetAttr<Target>(tvm::attr::kTarget);
  TVM_FFI_ICHECK(target)
      << "LowerCustomDatatypes: Require the target attribute";
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  bool needed = false;
  VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &node) {
    if (const auto *type = node.as<PrimTypeObj>())
      needed |= Registered(ffi::GetRef<PrimType>(type));
  });
  if (!needed)
    return function;
  auto lowerer = ffi::make_object<CustomDatatypeLowerer>(
      std::string(target.value()->kind->name));
  auto params = function->params.Map([&](const Var &var) {
    TVM_FFI_CHECK(!var->ty.as<TensorTypeObj>(), TypeError)
        << "LowerCustomDatatypes must run after the logical packed ABI is "
           "constructed";
    return lowerer->Define(var);
  });
  auto body = lowerer->Mutate(function->body).ValueOrUnchanged(function->body);
  auto attrs = lowerer->Mutate(function->attrs)
                   .as_or_throw<ffi::UnchangedOr<DictAttrs>>()
                   .ValueOrUnchanged(function->attrs);
  VisitNativePostOrder(body, [&](const ffi::ObjectRef &node) {
    if (const auto *expression = node.as<ExprObj>()) {
      if (const auto *type = expression->ty.as<PrimTypeObj>())
        TVM_FFI_CHECK(!Registered(ffi::GetRef<PrimType>(type)), TypeError)
            << "Unlowered custom datatype expression before host codegen: "
            << expression->GetTypeKey();
    }
  });
  return tirx::Function(params, body, function->ret_type, attrs, function->loc);
}
} // namespace

namespace transform {
tvm::transform::Pass LowerCustomDatatypes() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        return Lower(std::move(function));
      },
      0, "tl.LowerCustomDatatypes");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerCustomDatatypes",
                                   transform::LowerCustomDatatypes);
}
} // namespace tvm::tl
