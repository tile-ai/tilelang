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
 * \file lower_tensor_layouts.cc
 * \brief Retain TileLang parameter metadata through native address lowering.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

namespace tvm::tl {

tvm::transform::Pass PrepareTensorLayouts();

namespace {

bool HasExactDefaultLayout(const tirx::TensorType &type) {
  const auto *layout = type->layout.as<tirx::TileLayoutObj>();
  if (!layout || !layout->replica.empty() ||
      layout->shard.size() != type->shape.size()) {
    return false;
  }
  auto expected = tirx::TileLayout::DefaultLayout(type->shape);
  prim::ExprDeepEqual equal;
  for (size_t i = 0; i < layout->shard.size(); ++i) {
    const auto &actual = layout->shard[i];
    const auto &dense = expected->shard[i];
    // Check every dimension, including unit and empty dimensions. Contiguity
    // proofs can ignore those strides, but packed metadata queries cannot.
    if (actual->axis != dense->axis || !equal(actual->extent, dense->extent) ||
        !equal(actual->stride, dense->stride)) {
      return false;
    }
  }
  for (const auto &[axis, offset] : layout->offset) {
    if (axis != tirx::Axis::Get("m") || !prim::IsZero(offset)) {
      return false;
    }
  }
  return true;
}

IRModule NormalizeCompactParameters(IRModule module) {
  std::unordered_map<const VarObj *, Var> compact_parameters;
  for (const auto &[symbol, base] : module->functions) {
    auto function = base.as<tirx::Function>();
    if (!function ||
        function.value()->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      continue;
    }
    auto flags = function.value()->GetAttr<ffi::Array<ffi::Any>>(
        "tl.explicit_stride_params");
    if (flags) {
      TVM_FFI_CHECK_EQ(flags->size(), function.value()->params.size(),
                       ValueError)
          << "Explicit stride parameter metadata must match the signature";
    }
    for (size_t i = 0; i < function.value()->params.size(); ++i) {
      const auto &parameter = function.value()->params[i];
      auto type = parameter->ty.as<tirx::TensorType>();
      bool explicit_stride = flags && (*flags)[i].cast<bool>();
      if (explicit_stride) {
        TVM_FFI_CHECK(type && type.value()->layout.has_value(), ValueError)
            << "Explicit stride metadata requires a tensor layout";
      } else if (type && HasExactDefaultLayout(type.value())) {
        // An unmarked native default layout, including one materialized by
        // PrepareTensorLayouts, retains native compact-argument semantics.
        // A source StridedTensor is marked explicitly even when its geometry
        // happens to be identical; it must retain its declared constraints.
        auto compact_type =
            ffi::make_object<tirx::TensorTypeObj>(*type.value().get());
        compact_type->layout = std::nullopt;
        compact_parameters.emplace(
            parameter.get(),
            Var(parameter->name, tirx::TensorType(std::move(compact_type)),
                parameter->loc));
      }
    }
  }
  if (compact_parameters.empty()) {
    return module;
  }
  return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
             std::move(module),
             [&compact_parameters](const Var &variable) -> Var {
               auto found = compact_parameters.find(variable.get());
               return found == compact_parameters.end() ? variable
                                                        : found->second;
             })
      .cast<IRModule>();
}

void CheckParameterGeometry(const Var &original, const Var &lowered) {
  auto old_type = original->ty.as<tirx::TensorType>();
  auto new_type = lowered->ty.as<tirx::TensorType>();
  TVM_FFI_ICHECK(old_type && new_type)
      << "LowerTensorLayouts changed a non-tensor parameter";
  TVM_FFI_ICHECK(
      old_type.value()->dtype == new_type.value()->dtype &&
      old_type.value()->scope == new_type.value()->scope &&
      old_type.value()->data_alignment == new_type.value()->data_alignment &&
      old_type.value()->shape.size() == new_type.value()->shape.size())
      << "LowerTensorLayouts changed a non-layout parameter contract";
  prim::ExprDeepEqual equal;
  for (size_t i = 0; i < old_type.value()->shape.size(); ++i) {
    TVM_FFI_ICHECK(
        equal(old_type.value()->shape[i], new_type.value()->shape[i]))
        << "LowerTensorLayouts changed a parameter's logical shape";
  }
}

} // namespace

tvm::transform::Pass LowerTensorLayouts() {
  auto pass = [](IRModule module, tvm::transform::PassContext) {
    module = NormalizeCompactParameters(std::move(module));
    ffi::Map<GlobalVar, ffi::Array<Var>> original_parameters;
    for (const auto &[symbol, base] : module->functions) {
      if (auto function = base.as<tirx::Function>();
          function &&
          !function.value()->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
        original_parameters.Set(symbol, function.value()->params);
      }
    }

    // Save the ABI before preparation materializes implicit compact layouts.
    // A unit-axis explicit stride can be irrelevant to addresses but still
    // supplies a scalar used by the program or constrains a packed argument.
    module = PrepareTensorLayouts()(std::move(module));
    module = tirx::transform::LowerTensorLayouts()(std::move(module));

    std::unordered_map<const VarObj *, Var> replacements;
    for (const auto &[symbol, original] : original_parameters) {
      const auto current =
          module->functions.at(symbol).as_or_throw<tirx::Function>();
      TVM_FFI_ICHECK_EQ(original.size(), current->params.size())
          << "LowerTensorLayouts changed the parameter count";
      for (size_t i = 0; i < original.size(); ++i) {
        const Var &lowered = current->params[i];
        if (!original[i].same_as(lowered)) {
          CheckParameterGeometry(original[i], lowered);
          replacements.emplace(lowered.get(), original[i]);
        }
      }
    }

    // Match input identities before descending through their types. The same
    // map covers definitions, body operands and metadata references. Native
    // flattened aliases and their already-computed physical indices stay as
    // produced by LowerTensorLayouts; no address is lowered for a second time.
    if (!replacements.empty()) {
      module =
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
              std::move(module),
              [&replacements](const Var &variable) -> Var {
                auto found = replacements.find(variable.get());
                return found == replacements.end() ? variable : found->second;
              })
              .cast<IRModule>();
    }
    auto functions = module->functions;
    for (const auto &[symbol, unused] : original_parameters) {
      auto function = functions.at(symbol).as_or_throw<tirx::Function>();
      function = WithoutAttr(std::move(function), "tl.explicit_stride_params");
      auto *node = function.CopyOnWrite();
      node->ty = node->func_type_annotation();
      functions.Set(symbol, std::move(function));
    }
    module.CopyOnWrite()->functions = std::move(functions);
    return module;
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.LowerTensorLayouts");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerTensorLayouts",
                                   LowerTensorLayouts);
}

} // namespace tvm::tl
