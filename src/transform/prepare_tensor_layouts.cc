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
 * \file prepare_tensor_layouts.cc
 * \brief Materialize implicit layouts before native definition-based lowering.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

tvm::transform::Pass PrepareTensorLayouts() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> tirx::Function {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      return function;
    }
    // LowerTensorLayouts derives parameter aliases before visiting the body.
    // With an implicit layout it can initially keep a tensor unchanged, then
    // create a different type only under a loop's local constraints. There is
    // no definition for that late-created alias. An explicit compact layout
    // makes it derive and cache the physical view at the original definition.
    // This adds no dimension predicates or proof policy. Existing layouts,
    // including strided views and nonzero origins, remain unchanged.
    return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
               std::move(function),
               [](const tirx::TensorType &type) -> tirx::TensorType {
                 if (type->layout.has_value()) {
                   return type;
                 }
                 // Empty compact tensors have no coordinates to regroup. Keep
                 // their implicit representation: native lowering may still
                 // visit accesses inside zero-trip loops, and explicit zero
                 // iterators would introduce division by zero in grouping.
                 for (const auto &extent : type->shape) {
                   if (prim::IsZero(extent)) {
                     return type;
                   }
                 }
                 auto explicit_type =
                     ffi::make_object<tirx::TensorTypeObj>(*type.get());
                 explicit_type->layout =
                     tirx::TileLayout::DefaultLayout(type->shape);
                 return tirx::TensorType(std::move(explicit_type));
               })
        .cast<tirx::Function>();
  };
  // Mapping the entire Function preserves definition identity in parameter
  // types, local declarations, shape/layout operands and every corresponding
  // use. No correspondence is inferred from tensor names or pointer spelling.
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.PrepareTensorLayouts");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.PrepareTensorLayouts",
                                   PrepareTensorLayouts);
}

} // namespace tvm::tl
